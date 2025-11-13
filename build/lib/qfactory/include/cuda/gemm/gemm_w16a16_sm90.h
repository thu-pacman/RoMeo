#include <cstdio>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

#include "cute/tensor.hpp"

#include "cuda/utils.h"

using namespace cute;

template <typename smemALayout, typename smemBLayout>
struct SharedStorage {
    array_aligned<half_t, cosize_v<smemALayout>> smemA;
    array_aligned<half_t, cosize_v<smemBLayout>> smemB;
};

template <int remain>
__device__ void pipeline_tail(auto f) {
    cp_async_wait<remain>();
    __syncthreads();
    f(remain);
    if constexpr (remain > 0) {
        pipeline_tail<remain - 1>(f);
    }
}

template <
    int N, int K,
    int TileM, int TileN, int TileK, int NStage,
    typename MMA, typename G2SCopyA, typename G2SCopyB,
    typename SharedStorage, typename smemALayout, typename smemBLayout
>
__global__ void gemm_kernel(
    int M,
    half_t *A, half_t *B, half_t *C
) {
    int idx = blockIdx.x, idy = blockIdx.y;

    Tensor gA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(M, Int<K>{}), make_stride(Int<K>{}, _1{}))); // (M, K)
    Tensor gB = make_tensor(make_gmem_ptr(B), make_layout(make_shape(Int<N>{}, Int<K>{}), make_stride(Int<K>{}, _1{})));
    Tensor gC = make_tensor(make_gmem_ptr(C), make_layout(make_shape(M, Int<N>{}), make_stride(Int<N>{}, _1{})));

    Tensor gA_tile = local_tile(gA, make_tile(Int<TileM>{}, Int<TileK>{}), make_coord(idx, _)); // (TM, TK, niters)
    Tensor gB_tile = local_tile(gB, make_tile(Int<TileN>{}, Int<TileK>{}), make_coord(idy, _)); // (TN, TK, niters)
    Tensor gC_tile = local_tile(gC, make_tile(Int<TileM>{}, Int<TileN>{}), make_coord(idx, idy)); // (TM, TN)

    extern __shared__ char shared_memory[];
    SharedStorage &smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    half_t *smemA = smem.smemA.data(), *smemB = smem.smemB.data();

    Tensor sA = make_tensor(make_smem_ptr(smemA), smemALayout{});
    Tensor sB = make_tensor(make_smem_ptr(smemB), smemBLayout{});

    G2SCopyA copyA;
    G2SCopyB copyB;
    auto copyA_thread = copyA.get_slice(threadIdx.x);
    auto copyB_thread = copyB.get_slice(threadIdx.x);

    int num_iters = K / TileK;

    auto gA_tile_g2s_thread = copyA_thread.partition_S(gA_tile);
    auto sA_tile_g2s_thread = copyA_thread.partition_D(sA);
    auto gB_tile_g2s_thread = copyB_thread.partition_S(gB_tile);
    auto sB_tile_g2s_thread = copyB_thread.partition_D(sB);

    MMA mma;
    auto mma_thread = mma.get_slice(threadIdx.x);
    auto sA_tile_mma_thread = mma_thread.partition_A(sA); // (MMA, MMA_M, MMA_K, NStage)
    auto sB_tile_mma_thread = mma_thread.partition_B(sB);
    auto gC_tile_thread = mma_thread.partition_C(gC_tile);
    auto local_A = mma_thread.make_fragment_A(sA_tile_mma_thread);
    auto local_B = mma_thread.make_fragment_B(sB_tile_mma_thread);
    auto local_C = partition_fragment_C(mma, Shape<Int<TileM>, Int<TileN>>{});
    auto local_C_fp16 = make_tensor<half_t>(shape(local_C));

    clear(local_C);

    auto do_wgmma = [&] (int k_idx) {
        warpgroup_arrive();
        gemm(mma, local_C, local_A(_, _, _, k_idx), local_B(_, _, _, k_idx), local_C);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
    };

    for (int comp_k = 0, load_k = 0; comp_k + NStage <= num_iters; comp_k++) {
        for (; load_k < num_iters && load_k < (comp_k + NStage); load_k++) {
            int shared_idx = load_k % NStage;

            copy(copyA, gA_tile_g2s_thread(_, _, _, load_k), sA_tile_g2s_thread(_, _, _, shared_idx));
            copy(copyB, gB_tile_g2s_thread(_, _, _, load_k), sB_tile_g2s_thread(_, _, _, shared_idx));
            cp_async_fence();
        }
        cp_async_wait<NStage - 1>();
        __syncthreads();
        do_wgmma(comp_k % NStage);
    }

    if constexpr (NStage > 1) {
        pipeline_tail<NStage - 2>([&] (int remain_k) {
            do_wgmma((num_iters - remain_k - 1) % NStage);
        });
    }

    for (int i = 0; i < local_C.size(); i++) {
        local_C_fp16(i) = (half_t) local_C(i);
    }
    copy(local_C_fp16, gC_tile_thread);
}

template <
    int N, int K,
    int TileM, int TileN, int TileK, int NStage
>
int call_gemm(
    int M,
    half_t *A, half_t *B, half_t *C,
    cudaStream_t stream
) {
    using MMA = decltype(make_tiled_mma(
        SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{}
    ));

    constexpr int num_total_threads = size(MMA{});

    using smemALayout = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<half_t>{},
        make_shape(Int<TileM>{}, Int<TileK>{}, Int<NStage>{})
    ));
    using smemBLayout = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<half_t>{},
        make_shape(Int<TileN>{}, Int<TileK>{}, Int<NStage>{})
    ));
    
    using sharedStorage = SharedStorage<smemALayout, smemBLayout>;

    constexpr int num_thread_per_row = TileK / 8;
    constexpr int num_thread_rows = num_total_threads / num_thread_per_row;

    using G2SCopyA = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half_t>{},
        make_right_layout(Shape<Int<num_thread_rows>, Int<num_thread_per_row>>{}),
        make_right_layout(Shape<_1, _8>{}) // load 8 fp16 per thread (128bit)
    ));

    using G2SCopyB = G2SCopyA;

    dim3 block(size(MMA{}));
    dim3 grid(M / TileM, N / TileN);

    assert(M % TileM == 0);
    assert(N % TileN == 0);
    assert(K % TileK == 0);
    assert(K / TileK >= NStage - 1);

    auto smem_size = sizeof(sharedStorage);
    
    if(cudaFuncSetAttribute(
        gemm_kernel <
            N, K,
            TileM, TileN, TileK, NStage,
            MMA, G2SCopyA, G2SCopyB,
            sharedStorage, smemALayout, smemBLayout
        >,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ) != cudaSuccess) return 1;

    gemm_kernel <
        N, K,
        TileM, TileN, TileK, NStage,
        MMA, G2SCopyA, G2SCopyB,
        sharedStorage, smemALayout, smemBLayout
    > <<< grid, block, smem_size, stream >>> (
        M,
        A, B, C
    );

    return 0;
}