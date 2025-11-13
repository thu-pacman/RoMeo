#include <cstdio>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

#include "cute/tensor.hpp"

#include "cuda/utils.h"

using namespace cute;

template <typename smemALayout, typename smemBLayout, typename smemCLayout, int TM>
struct SharedStorage {
    array_aligned<__nv_fp8_e4m3, cosize_v<smemALayout>> smemA;
    array_aligned<__nv_fp8_e4m3, cosize_v<smemBLayout>> smemB;
    array_aligned<half_t, cosize_v<smemCLayout>> smemC;
    array_aligned<int, TM> smemTokenIds;
};

template <typename M, typename N, typename BM, typename BN>
__host__ __device__ auto make_scale_layout(M m, N n, BM bm, BN bn) {
    return make_layout(
        make_shape(
            make_shape(bm, m / bm),
            make_shape(bn, n / bn)
        ),
        make_stride(
            make_stride(0, n / bn),
            make_stride(0, 1)
        )
    );
}

template <typename E, typename M, typename N, typename BM, typename BN>
__host__ __device__ auto make_scale_layout(E e, M m, N n, BM bm, BN bn) {
    return make_layout(
        make_shape(
            e,
            make_shape(bm, m / bm),
            make_shape(bn, n / bn)
        ),
        make_stride(
            (m / bm) * (n / bn),
            make_stride(0, n / bn),
            make_stride(0, 1)
        )
    );
}

template <int M, int N, int NELM, int BS>
struct Array2DPartition {
    static constexpr int threads_per_row = N / NELM;
    static constexpr int rows_per_iter = BS / threads_per_row;
    static constexpr int niters = M / rows_per_iter;
    static constexpr int num_elements = NELM;
    
    int idx[niters], idy[niters];

    __host__ __device__ Array2DPartition(int tid) {
        CUTE_STATIC_ASSERT(N % NELM == 0);
        CUTE_STATIC_ASSERT(threads_per_row <= BS);
        CUTE_STATIC_ASSERT(M % rows_per_iter == 0);
        for (int i = 0; i < niters; i++) {
            idx[i] = tid / threads_per_row;
            idy[i] = (tid % threads_per_row) * NELM;
            tid += BS;
        }
    }
    
    __host__ __device__ constexpr int size() const {
        return niters;
    }
};


template <
    int N, int K, int E, int topk, bool has_moe_weight,
    int TileM, int TileN, int TileK, int NStage,
    typename MMA, typename G2SCopyA, typename G2SCopyB,
    typename SharedStorage, typename smemALayout, typename smemBLayout, typename smemCLayout
>
__global__ void gemm_kernel(
    int M, int *num_chunks_tensor,
    __nv_fp8_e4m3 *A, __nv_fp8_e4m3 *B, __nv_bfloat16 *C,
    float *scaleA, float *scaleB,
    int *token_ids, int *expert_ids,
    float *moe_weight
) {
    int idx = blockIdx.x, idy = blockIdx.y;

    int eid = expert_ids[idx];

    constexpr int BlockM = 1;
    constexpr int BlockN = TileN;
    constexpr int BlockK = TileK;

    int num_chunks = num_chunks_tensor[0];
    if (idx >= num_chunks) return;

    Tensor gA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(M, Int<K>{}), make_stride(Int<K>{}, _1{}))); // (M, K)
    Tensor gB = make_tensor(make_gmem_ptr(B), make_layout(make_shape(Int<E>{}, Int<N>{}, Int<K>{}), make_stride(Int<N * K>{}, Int<K>{}, _1{})));
    Tensor gC = make_tensor(make_gmem_ptr(C), make_layout(make_shape(M * topk, Int<N>{}), make_stride(Int<N>{}, _1{})));
    Tensor gScaleA = make_tensor(make_gmem_ptr(scaleA), make_scale_layout(M, Int<K>{}, Int<BlockM>{}, Int<BlockK>{}));
    Tensor gScaleB = make_tensor(make_gmem_ptr(scaleB), make_scale_layout(Int<E>{}, Int<N>{}, Int<K>{}, Int<BlockN>{}, Int<BlockK>{}));
    Tensor gTokenIds = make_tensor(make_gmem_ptr(token_ids), make_layout(make_shape(num_chunks, Int<TileM>{}), make_stride(Int<TileM>{}, _1{})));

    Tensor gA_tile = local_tile(gA, make_tile(M, Int<TileK>{}), make_coord(0, _)); // (M, TK, niters)
    Tensor gB_tile = local_tile(gB(eid, _, _), make_tile(Int<TileN>{}, Int<TileK>{}), make_coord(idy, _)); // (TN, TK, niters)
    Tensor gC_tile = local_tile(gC, make_tile(M * topk, Int<TileN>{}), make_coord(0, idy)); // (M * topk, TN)

    extern __shared__ char shared_memory[];
    SharedStorage &smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    __nv_fp8_e4m3 *smemA = smem.smemA.data(), *smemB = smem.smemB.data();
    half_t *smemC = smem.smemC.data();
    int *smemTokenIds = smem.smemTokenIds.data();

    Tensor sA = make_tensor(make_smem_ptr(smemA), smemALayout{});
    Tensor sB = make_tensor(make_smem_ptr(smemB), smemBLayout{});
    Tensor sC = make_tensor(make_smem_ptr(smemC), smemCLayout{});
    Tensor sTokenIds = make_tensor(make_smem_ptr(smemTokenIds), make_shape(Int<TileM>{}));

    G2SCopyB copyB;
    auto copyB_thread = copyB.get_slice(threadIdx.x);

    MMA mma;
    auto mma_thread = mma.get_slice(threadIdx.x);
    auto sC_tile_thread = mma_thread.partition_C(sC);
    auto local_C = make_fragment_like(sC_tile_thread);

    for (int i = threadIdx.x; i < TileM; i += blockDim.x) {
        sTokenIds(i) = gTokenIds(idx, i);
    }
    __syncthreads();

    int num_iters = K / TileK;

    auto gB_tile_g2s_thread = copyB_thread.partition_S(gB_tile);
    auto sB_tile_g2s_thread = copyB_thread.partition_D(sB);

    auto sA_tile_mma_thread = mma_thread.partition_A(sA); // (MMA, MMA_M, MMA_K, NStage)
    auto sB_tile_mma_thread = mma_thread.partition_B(sB);

    auto local_A = mma_thread.make_fragment_A(sA_tile_mma_thread);
    auto local_B = mma_thread.make_fragment_B(sB_tile_mma_thread);

    CUTE_STATIC_ASSERT(TileN == BlockN);
    CUTE_STATIC_ASSERT(TileK == BlockK);

    Array2DPartition<TileM, TileN, 16 / sizeof(half_t), size(mma)> sCPartition(threadIdx.x); // 16 byte per thread
    Array2DPartition<TileM, TileK, 16 / sizeof(__nv_fp8_e4m3), size(mma)> sAPartition(threadIdx.x);
    
    int CtokenIds[sCPartition.size()];
    for (int i = 0; i < sCPartition.size(); i++) {
        CtokenIds[i] = sTokenIds(sCPartition.idx[i]) / topk;
    }
    int AtokenIds[sAPartition.size()];
    for (int i = 0; i < sAPartition.size(); i++) {
        AtokenIds[i] = sTokenIds(sAPartition.idx[i]) / topk;
    }

    half2 frag_acc[sCPartition.size()][sCPartition.num_elements / 2];
    for (int i = 0; i < sCPartition.size(); i++) {
        for (int j = 0; j < sCPartition.num_elements / 2; j++) {
            frag_acc[i][j] = half2{0.0f, 0.0f};
        }
    }

    for (int comp_k = 0, load_k = 0; comp_k < num_iters; comp_k++) {
        for (; load_k < num_iters && load_k < (comp_k + NStage); load_k++) {
            int shared_idx = load_k % NStage;

            for (int i = 0; i < sAPartition.size(); i++) {
                int id = AtokenIds[i];
                if (id == M) continue;
                int j = sAPartition.idy[i];
                uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(&sA(sAPartition.idx[i], j, shared_idx)));
                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(smem), "l"(&gA_tile(id, j, load_k))
                );
            }

            copy(copyB, gB_tile_g2s_thread(_, _, _, load_k), sB_tile_g2s_thread(_, _, _, shared_idx));
            cp_async_fence();
        }
        if (comp_k + NStage > num_iters) {
            cp_async_wait<0>(); // NStage - comp_k - 1
        } else {
            cp_async_wait<NStage - 1>();
        }
        __syncthreads();
        int shared_idx = comp_k % NStage;
        clear(local_C);
        warpgroup_arrive();
        gemm(mma, local_A(_, _, _, shared_idx), local_B(_, _, _, shared_idx), local_C);
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        // reg to shared
        for (int i = 0; i < local_C.size(); i += 2) {
            *reinterpret_cast<uint32_t*>(&(sC_tile_thread[i])) = *reinterpret_cast<uint32_t*>(&(local_C[i]));
        }
        __syncthreads();

        // mul scale and accumulate
        float scale_b = gScaleB(eid, idy * BlockN, comp_k * BlockK);
        for (int i = 0; i < sCPartition.size(); i++) {
            int id = CtokenIds[i];
            if (id == M) continue;
            float scale_a = gScaleA(id, comp_k * BlockK);
            half2 frag[sCPartition.num_elements / 2];
            *reinterpret_cast<uint128_t*>(frag) = *reinterpret_cast<uint128_t*>(&sC(sCPartition.idx[i], sCPartition.idy[i]));
            half2 scale = __float2half2_rn(scale_a * scale_b);
            for (int t = 0; t < sCPartition.num_elements / 2; t++) {
                frag_acc[i][t] = frag_acc[i][t] + frag[t] * scale;
            }
        }
        __syncthreads();
        
    }

    // write back
    for (int i = 0; i < sCPartition.size(); i++) {
        int id = sTokenIds(sCPartition.idx[i]);
        if (id >= M * topk) continue;
        int j = sCPartition.idy[i];
        half2 moe_w;
        if (has_moe_weight) {
            moe_w = __float2half2_rn(moe_weight[id]);
        }
        for (int t = 0; t < sCPartition.num_elements / 2; t++) {
            half2 res = frag_acc[i][t];
            if (has_moe_weight) {
                res = res * moe_w;
            }
            __nv_bfloat162 &gc = *reinterpret_cast<__nv_bfloat162*>(&gC_tile(id, j + t * 2));
            gc = __floats2bfloat162_rn((float)res.x, (float)res.y);
        }
    }
}

template <
    int N, int K, int E, int topk, bool has_moe_weight,
    int TileM, int TileN, int TileK, int NStage
>
void call_gemm(
    int M, int *num_chunks_tensor, int max_num_chunks,
    __nv_fp8_e4m3 *A, __nv_fp8_e4m3 *B, __nv_bfloat16 *C,
    float *scaleA, float *scaleB,
    int *token_ids, int *expert_ids,
    void *_moe_weight, // optional
    cudaStream_t stream
) {
    float *moe_weight = reinterpret_cast<float*>(_moe_weight);
    
    using MMA = decltype(make_tiled_mma(
        SM90_64x128x32_F16E4M3E4M3_SS_TN<>{}
    ));

    constexpr int num_total_threads = size(MMA{});

    using smemALayout = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<__nv_fp8_e4m3>{},
        make_shape(Int<TileM>{}, Int<TileK>{}, Int<NStage>{})
    ));
    using smemBLayout = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<__nv_fp8_e4m3>{},
        make_shape(Int<TileN>{}, Int<TileK>{}, Int<NStage>{})
    ));
    using smemCLayout = decltype(composition(
        Swizzle<3,3,3>{},
        make_right_layout(
            make_shape(Int<TileM>{}, Int<TileN>{})
        )
    ));
    
    using sharedStorage = SharedStorage<smemALayout, smemBLayout, smemCLayout, TileM>;

    constexpr int num_thread_per_row = TileK / 16;
    constexpr int num_thread_rows = num_total_threads / num_thread_per_row;

    using G2SCopyA = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, __nv_fp8_e4m3>{},
        make_right_layout(Shape<Int<num_thread_rows>, Int<num_thread_per_row>>{}),
        make_right_layout(Shape<_1, _16>{}) // load 8 fp8 per thread (128bit)
    ));

    using G2SCopyB = G2SCopyA;

    dim3 block(size(MMA{}));
    dim3 grid(max_num_chunks, N / TileN);

    // assert(M % TileM == 0);
    assert(N % TileN == 0);
    assert(K % TileK == 0);

    auto smem_size = sizeof(sharedStorage);
    
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        gemm_kernel <
            N, K, E, topk, has_moe_weight,
            TileM, TileN, TileK, NStage,
            MMA, G2SCopyA, G2SCopyB,
            sharedStorage, smemALayout, smemBLayout, smemCLayout
        >,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));

    gemm_kernel <
        N, K, E, topk, has_moe_weight,
        TileM, TileN, TileK, NStage,
        MMA, G2SCopyA, G2SCopyB,
        sharedStorage, smemALayout, smemBLayout, smemCLayout
    > <<< grid, block, smem_size, stream >>> (
        M, num_chunks_tensor,
        A, B, C,
        scaleA, scaleB,
        token_ids, expert_ids,
        moe_weight
    );
}