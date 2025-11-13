#include <cstdio>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

#include "cute/tensor.hpp"

#include "cuda/utils.h"

using namespace cute;

template <typename smemALayout, typename smemBLayout, typename smemCLayout, int TileM, int TileN, int NStage>
struct SharedStorage {
    array_aligned<int8_t, cosize_v<smemALayout> / 2> smemA;
    array_aligned<int8_t, cosize_v<smemBLayout> / 2> smemB;
    array_aligned<half_t, cosize_v<smemCLayout>> smemC;
    array_aligned<half_t, TileM * NStage> smemScaleA;
    array_aligned<half_t, TileN * NStage> smemScaleB;
};

template <
    int N, int K, int GROUP_K,
    int TileM, int TileN, int TileK, int NStage, int WarpM, int WarpN, int WarpK,
    typename MMA, typename G2SCopyA, typename G2SCopyB, typename S2RCopyA, typename S2RCopyB, typename S2GCopyC,
    typename SharedStorage, typename smemALayout, typename smemBLayout, typename smemCLayout
>
__global__ void gemm_kernel(
    int M,
    int4b_t *A, int4b_t *B, half_t *C,
    half_t *A_scale, half_t *B_scale
) {
    int idx = blockIdx.x, idy = blockIdx.y;

    Tensor gA = make_tensor(make_gmem_ptr<int4b_t>(A), make_layout(make_shape(M, Int<K>{}), make_stride(Int<K>{}, _1{}))); // (M, K)
    Tensor gB = make_tensor(make_gmem_ptr<int4b_t>(B), make_layout(make_shape(Int<N>{}, Int<K>{}), make_stride(Int<K>{}, _1{}))); // (N, K)
    Tensor gC = make_tensor(make_gmem_ptr<half_t>(C), make_layout(make_shape(M, Int<N>{}), make_stride(Int<N>{}, _1{}))); // (M, N)
    Tensor gScaleA = make_tensor(make_gmem_ptr<half_t>(A_scale), make_layout(make_shape(M, Int<K / GROUP_K>{}), make_stride(Int<K / GROUP_K>{}, _1{}))); // (M, K/GK)
    Tensor gScaleB = make_tensor(make_gmem_ptr<half_t>(B_scale), make_layout(make_shape(Int<N>{}, Int<K / GROUP_K>{}), make_stride(Int<K / GROUP_K>{}, _1{}))); // (N, K/GK)

    auto gA_block = local_tile(gA, make_tile(Int<TileM>{}, Int<TileK>{}), make_coord(idx, _)); // (TM, TK, niters)
    auto gB_block = local_tile(gB, make_tile(Int<TileN>{}, Int<TileK>{}), make_coord(idy, _)); // (TN, TK, niters)
    auto gC_block = local_tile(gC, make_tile(Int<TileM>{}, Int<TileN>{}), make_coord(idx, idy)); // (TM, TN)
    auto gC_tile = local_tile(gC_block, make_tile(Int<WarpM>{}, Int<WarpN>{}), make_coord(_, _)); // (WM, WN, iter_M, iter_N)
    auto gScaleA_block = local_tile(gScaleA, make_tile(Int<TileM>{}, Int<K / GROUP_K>{}), make_coord(idx, 0)); // (TM, K / GK)
    auto gScaleB_block = local_tile(gScaleB, make_tile(Int<TileN>{}, Int<K / GROUP_K>{}), make_coord(idy, 0)); // (TN, K / GK)

    extern __shared__ char shared_memory[];
    SharedStorage &smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    int4b_t *smemA = reinterpret_cast<int4b_t*>(smem.smemA.data());
    int4b_t *smemB = reinterpret_cast<int4b_t*>(smem.smemB.data());
    half_t *smemC = smem.smemC.data();
    half_t *smemScaleA = smem.smemScaleA.data();
    half_t *smemScaleB = smem.smemScaleB.data();

    Tensor sA = make_tensor(make_smem_ptr<int4b_t>(smemA), smemALayout{}); // (TM, TK, NStage)
    Tensor sB = make_tensor(make_smem_ptr<int4b_t>(smemB), smemBLayout{}); // (TN, TK, NStage)
    Tensor sC = make_tensor(make_smem_ptr<half_t>(smemC), smemCLayout{}); // (WM, WN)
    Tensor sScaleA = make_tensor(make_smem_ptr(smemScaleA), make_layout(
        make_shape(Int<TileM>{}, Int<TileN>{}, Int<NStage>{}),
        make_stride(_1{}, _0{}, Int<TileM>{})
    )); // (TM, TN, NStage)
    Tensor sScaleB = make_tensor(make_smem_ptr(smemScaleB), make_layout(
        make_shape(Int<TileM>{}, Int<TileN>{}, Int<NStage>{}),
        make_stride(_0{}, _1{}, Int<TileN>{})
    )); // (TM, TN, NStage)

    auto sA_tile = local_tile(sA, make_tile(Int<WarpM>{}, Int<WarpK>{}), make_coord(_, _)); // (WM, WK, iter_M, iter_K, NStage)
    auto sB_tile = local_tile(sB, make_tile(Int<WarpN>{}, Int<WarpK>{}), make_coord(_, _)); // (WN, WK, iter_N, iter_K, NStage)
    auto sScaleA_tile = local_tile(sScaleA, make_tile(Int<WarpM>{}, Int<WarpN>{}), make_coord(_, _)); // (WM, WN, iter_M, iter_N, NStage)
    auto sScaleB_tile = local_tile(sScaleB, make_tile(Int<WarpM>{}, Int<WarpN>{}), make_coord(_, _)); // (WM, WN, iter_M, iter_N, NStage)

    MMA mma;
    auto mma_thread = mma.get_slice(threadIdx.x);
    Tensor local_A = mma_thread.partition_fragment_A(sA_tile(_, _, _, 0, 0)); // (MMA, MMA_WM, MMA_WK, iter_M)
    Tensor local_B = mma_thread.partition_fragment_B(sB_tile(_, _, _, 0, 0)); // (MMA, MMA_WN, MMA_WK, iter_N)
    Tensor local_C = partition_fragment_C(mma_thread, gC_tile.shape()); // (MMA, MMA_WM, MMA_WN, iter_M, iter_N)
    Tensor local_C_acc = make_tensor<half_t>(partition_shape_C(mma_thread, gC_tile.shape())); // (MMA, MMA_WM, MMA_WN, iter_M, iter_N)

    // global gA_block/gB_block -> shared sA/sB
    G2SCopyA copyA;
    auto copyA_thread = copyA.get_slice(threadIdx.x);
    auto gA_block_g2s_thread = copyA_thread.partition_S(gA_block); // (COPY, COPY_TM, COPY_TK, niters)
    auto sA_g2s_thread = copyA_thread.partition_D(sA); // (COPY, COPY_TM, COPY_TK, NStage)
    G2SCopyB copyB;
    auto copyB_thread = copyB.get_slice(threadIdx.x);
    auto gB_block_g2s_thread = copyB_thread.partition_S(gB_block); // (COPY, COPY_TN, COPY_TK, niters)
    auto sB_g2s_thread = copyB_thread.partition_D(sB); // (COPY, COPY_TN, COPY_TK, NStage)

    // shared sA_tile/sB_tile -> reg local_A/local_B
    S2RCopyA copyA_s2r;
    auto copyA_s2r_thread = copyA_s2r.get_slice(threadIdx.x);
    auto sA_tile_s2r_thread = copyA_s2r_thread.partition_S(sA_tile); // (COPY, COPY_WM, COPY_WK, iter_M, iter_K, NStage)
    auto local_A_s2r_view = copyA_s2r_thread.retile_D(local_A); // (COPY, COPY_WM, COPY_WK, iter_M)
    S2RCopyB copyB_s2r;
    auto copyB_s2r_thread = copyB_s2r.get_slice(threadIdx.x);
    auto sB_tile_s2r_thread = copyB_s2r_thread.partition_S(sB_tile); // (COPY, COPY_WN, COPY_WK, iter_N, iter_K, NStage)
    auto local_B_s2r_view = copyB_s2r_thread.retile_D(local_B); // (COPY, COPY_WN, COPY_WK, iter_N)

    // reg local_C -> local_C_acc
    // cast and scale
    auto sScaleA_thread = mma_thread.partition_C(sScaleA_tile); // (MMA, MMA_WM, MMA_WN, iter_M, iter_N, NStage)
    auto sScaleB_thread = mma_thread.partition_C(sScaleB_tile); // (MMA, MMA_WM, MMA_WN, iter_M, iter_N, NStage)

    // local_C_acc -> shared sC
    // direct copy
    auto sC_thread = mma_thread.partition_C(sC); // (MMA, MMA_WM, MMA_WN)

    // shared sC -> global gC_tile
    S2GCopyC copyC;
    auto copyC_thread = copyC.get_slice(threadIdx.x);
    auto sC_s2g_thread = copyC_thread.partition_S(sC); // (COPY, COPY_WM, COPY_WN)
    auto gC_tile_s2g_thread = copyC_thread.partition_D(gC_tile); // (COPY, COPY_WM, COPY_WN, iter_M, iter_N)

    constexpr int niters = K / TileK;
    constexpr int iter_m = TileM / WarpM;
    constexpr int iter_n = TileN / WarpN;
    constexpr int iter_k = TileK / WarpK;
    
    // Copy Scales to shared memory
    CUTE_STATIC_ASSERT(GROUP_K % TileK == 0);
    CUTE_STATIC_ASSERT(size(mma) >= TileM);
    CUTE_STATIC_ASSERT(size(mma) >= TileN);
    auto load_shared_scales = [&] (int load_k) {
        int k_idx = load_k * TileK;
        if ((k_idx + TileK) % GROUP_K == 0) { // fetch at the last iteration of each group
            int group_id = k_idx / GROUP_K;
            int scale_shared_idx = group_id % NStage;
            if (threadIdx.x < TileM) {
                sScaleA(threadIdx.x, 0, scale_shared_idx) = gScaleA_block(threadIdx.x, group_id);
            }
            if (threadIdx.x < TileN) {
                sScaleB(0, threadIdx.x, scale_shared_idx) = gScaleB_block(threadIdx.x, group_id);
            }
            __syncthreads();
        }
    };

    clear(local_C_acc);

    auto do_wmma = [&] (int k_idx) {
        for (int k = 0; k < iter_k; k++) {
            for (int i = 0; i < iter_m; i++) {
                copy(copyA_s2r, sA_tile_s2r_thread(_, _, _, i, k, k_idx), local_A_s2r_view(_, _, _, i));
            }
            for (int j = 0; j < iter_n; j++) {
                copy(copyB_s2r, sB_tile_s2r_thread(_, _, _, j, k, k_idx), local_B_s2r_view(_, _, _, j));
            }
            for (int i = 0; i < iter_m; i++) {
                for (int j = 0; j < iter_n; j++) {
                    gemm(mma, local_C(_, _, _, i, j), local_A(_, _, _, i), local_B(_, _, _, j), local_C(_, _, _, i, j));
                }
            }
        }
    };

    clear(local_C);

    auto scale_acc = [&] (int block_k) {
        int scale_shared_idx = ((block_k * TileK) / GROUP_K) % NStage;
        for (int i = 0; i < iter_m; i++) {
            for (int j = 0; j < iter_n; j++) {
                for (int _2 = 0; _2 < size(get<2>(local_C.shape())); _2++) {
                    for (int _1 = 0; _1 < size(get<1>(local_C.shape())); _1++) {
                        for (int _0 = 0; _0 < size(get<0>(local_C.shape())); _0 += 2) {
                            half c[2];
                            half_t s1 = sScaleA_thread(_0, _1, _2, i, j, scale_shared_idx) * sScaleB_thread(_0, _1, _2, i, j, scale_shared_idx);
                            half_t s2 = sScaleA_thread(_0 + 1, _1, _2, i, j, scale_shared_idx) * sScaleB_thread(_0 + 1, _1, _2, i, j, scale_shared_idx);
                            c[0] = local_C(_0, _1, _2, i, j) * s1;
                            c[1] = local_C(_0 + 1, _1, _2, i, j) * s2;
                            half2 &acc = *reinterpret_cast<half2*>(&local_C_acc(_0, _1, _2, i, j));
                            acc = acc + *reinterpret_cast<half2*>(c);
                        }
                    }
                }
            }
        }
        clear(local_C);
    };

    auto load_block = [&] (int load_k) {
        int shared_idx = load_k % NStage;
        copy(copyA, gA_block_g2s_thread(_, _, _, load_k), sA_g2s_thread(_, _, _, shared_idx));
        copy(copyB, gB_block_g2s_thread(_, _, _, load_k), sB_g2s_thread(_, _, _, shared_idx));
        cp_async_fence();
        load_shared_scales(load_k);
    };

    // Pipeline
    // 1. `NStage` loads of blocks A and B
    // 2. `niters - NStage` steady states
    // 2.1 Wait last block
    // 2.2 Compute block `i`
    // 2.3 Load next block `i + NStage`
    // 3. `NStage` pipeline tail
    // 3.1 Wait last block
    // 3.2 Compute block `i`

    for (int i = 0; i < NStage; i++) {
        load_block(i);
    }

    constexpr int acc_interval = GROUP_K / TileK;

    for (int i = 0; i < niters - NStage; i += acc_interval) {
        for (int j = 0; j < acc_interval && i + j < niters - NStage; j++) {
            cp_async_wait<NStage - 1>();
            __syncthreads();
            do_wmma((i + j) % NStage);
            if (j == acc_interval - 1) {
                scale_acc(i + j);
            }
            load_block(i + j + NStage);
        }
    }

    auto pipe_tail = [&] <int i>() {
        cp_async_wait<i>();
        __syncthreads();
        do_wmma((niters - 1 - i) % NStage);
        if ((niters - i) % acc_interval == 0) {
            scale_acc(niters - 1 - i);
        }
    };

    CUTE_STATIC_ASSERT(NStage <= 3);
    if constexpr (NStage >= 3) pipe_tail.template operator()<2>();
    if constexpr (NStage >= 2) pipe_tail.template operator()<1>();
    if constexpr (NStage >= 1) pipe_tail.template operator()<0>();

    for (int i = 0; i < iter_m; i++) {
        for (int j = 0; j < iter_n; j++) {
            copy(AutoVectorizingCopy{}, local_C_acc(_, _, _, i, j), sC_thread);
            __syncthreads();
            copy(copyC, sC_s2g_thread, gC_tile_s2g_thread(_, _, _, i, j));
            __syncthreads();
        }
    }
}

template <
    int N, int K, int GROUP_K,
    int TileM, int TileN, int TileK, int NStage,
    int WarpM, int WarpN, int WarpK
>
int call_gemm(
    int M,
    uint8_t *A, uint8_t *B, half_t *C,
    half_t *A_scale, half_t *B_scale,
    cudaStream_t stream
) {
    using MMA = decltype(make_tiled_mma(
        SM80_16x8x64_S32S4S4S32_TN_SATURATE{},
        Layout<Shape<_2, _4, _1>>{},
        Tile<Int<WarpM>, Int<WarpN>, Int<WarpK>>{}
    ));

    constexpr int num_total_threads = size(MMA{});

    using smemALayout = decltype(composition(
        get_best_swizzle<TileK, 4>(),
        Layout<
            Shape<Int<TileM>, Int<TileK>, Int<NStage>>,
            Stride<Int<TileK>, _1, Int<TileM * TileK>>
        >{}
    ));
    using smemBLayout = decltype(composition(
        get_best_swizzle<TileK, 4>(),
        Layout<
            Shape<Int<TileN>, Int<TileK>, Int<NStage>>,
            Stride<Int<TileK>, _1, Int<TileN * TileK>>
        >{}
    ));
    using smemCLayout = decltype(composition(
        get_best_swizzle<WarpN, 16>(),
        Layout<
            Shape<Int<WarpM>, Int<WarpN>>,
            Stride<Int<WarpN>, _1>
        >{}
    ));
    
    using sharedStorage = SharedStorage<smemALayout, smemBLayout, smemCLayout, TileM, TileN, NStage>;

    constexpr int num_thread_per_row = TileK / 32;
    constexpr int num_thread_rows = num_total_threads / num_thread_per_row;

    using G2SCopyA = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, int4b_t>{},
        make_right_layout(Shape<Int<num_thread_rows>, Int<num_thread_per_row>>{}),
        make_right_layout(Shape<_1, _32>{}) // load 32 int4b per thread (128bit)
    ));
    using G2SCopyB = G2SCopyA;

    using S2RCopyA = decltype(make_tiled_copy_A(
        Copy_Atom<SM75_U32x4_LDSM_N, int4b_t>{},
        MMA{}
    ));
    using S2RCopyB = decltype(make_tiled_copy_B(
        Copy_Atom<SM75_U32x4_LDSM_N, int4b_t>{},
        MMA{}
    ));

    using S2GCopyC = decltype(make_tiled_copy(
        Copy_Atom<AutoVectorizingCopy, half_t>{},
        make_right_layout(Shape<Int<num_total_threads / (WarpN / 8)>, Int<WarpN / 8>>{}),
        make_right_layout(Shape<_1, _8>{}) // load 8 half per thread (128bit)
    ));

    dim3 block(size(MMA{}));
    dim3 grid(M / TileM, N / TileN);

    assert(M % TileM == 0);
    assert(N % TileN == 0);
    assert(K % TileK == 0);
    assert(K / TileK >= NStage - 1);

    auto smem_size = sizeof(sharedStorage);
    
    if (cudaFuncSetAttribute(
        gemm_kernel <
            N, K, GROUP_K,
            TileM, TileN, TileK, NStage, WarpM, WarpN, WarpK,
            MMA, G2SCopyA, G2SCopyB, S2RCopyA, S2RCopyB, S2GCopyC,
            sharedStorage, smemALayout, smemBLayout, smemCLayout
        >,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ) != cudaSuccess) return 1;

    gemm_kernel <
        N, K, GROUP_K,
        TileM, TileN, TileK, NStage, WarpM, WarpN, WarpK,
        MMA, G2SCopyA, G2SCopyB, S2RCopyA, S2RCopyB, S2GCopyC,
        sharedStorage, smemALayout, smemBLayout, smemCLayout
    > <<< grid, block, smem_size, stream >>> (
        M,
        reinterpret_cast<int4b_t*>(A),
        reinterpret_cast<int4b_t*>(B),
        C,
        A_scale,
        B_scale
    );

    return 0;
}