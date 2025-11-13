#include <cstdio>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

#include "cute/tensor.hpp"

#include "cuda/utils.h"

using namespace cute;

template <typename smemALayout, typename smemBLayout, typename smemCLayout, int TileM, int TileN>
struct SharedStorage {
    array_aligned<int8_t, cosize_v<smemALayout> / 2> smemA_4bit;
    array_aligned<int8_t, cosize_v<smemBLayout> / 2> smemB_4bit;
    array_aligned<__nv_bfloat16, cosize_v<smemCLayout>> smemC;
    array_aligned<__nv_bfloat16, TileM> smemScaleA;
    array_aligned<__nv_bfloat16, TileN> smemScaleB;
};

template <
    typename MMA_t, typename G2SCopyA_t, typename G2SCopyB_t, typename S2RCopyA_t, typename S2RCopyB_t, typename S2GCopyC_t,
    typename smemALayout_t, typename smemBLayout_t, typename smemCLayout_t
>
struct KernelTraits {
    using MMA = MMA_t;
    using G2SCopyA = G2SCopyA_t;
    using G2SCopyB = G2SCopyB_t;
    using S2RCopyA = S2RCopyA_t;
    using S2RCopyB = S2RCopyB_t;
    using S2GCopyC = S2GCopyC_t;
    using smemALayout = smemALayout_t;
    using smemBLayout = smemBLayout_t;
    using smemCLayout = smemCLayout_t;
};

template <
    int N, int K, int LDC,
    int TileM, int TileN, int TileK, int NStage, int WarpM, int WarpN, int WarpK,
    typename KTraits, typename SharedStorage
>
__global__ void gemm_kernel(
    int M,
    uint8_t *A, uint8_t *B, __nv_bfloat16 *C,
    __nv_bfloat16 *A_scale, __nv_bfloat16 *B_scale
) {
    int idx = blockIdx.x, idy = blockIdx.y;

    Tensor gA_4bit = make_tensor(make_gmem_ptr<int4b_t>(A), make_layout(make_shape(M, Int<K>{}), make_stride(Int<K>{}, _1{}))); // (M, K)
    Tensor gB_4bit = make_tensor(make_gmem_ptr<int4b_t>(B), make_layout(make_shape(Int<N>{}, Int<K>{}), make_stride(Int<K>{}, _1{}))); // (N, K)
    Tensor gC = make_tensor(make_gmem_ptr<__nv_bfloat16>(C), make_layout(make_shape(M, Int<N>{}), make_stride(Int<LDC>{}, _1{}))); // (M, N)
    Tensor gScaleA = make_tensor(make_gmem_ptr<__nv_bfloat16>(A_scale), make_layout(make_shape(M)));
    Tensor gScaleB = make_tensor(make_gmem_ptr<__nv_bfloat16>(B_scale), make_layout(make_shape(Int<N>{})));

    auto gA_block = local_tile(gA_4bit, make_tile(Int<TileM>{}, Int<TileK>{}), make_coord(idx, _)); // (TM, TK, niters)
    auto gB_block = local_tile(gB_4bit, make_tile(Int<TileN>{}, Int<TileK>{}), make_coord(idy, _)); // (TN, TK, niters)
    auto gC_block = local_tile(gC, make_tile(Int<TileM>{}, Int<TileN>{}), make_coord(idx, idy)); // (TM, TN)
    auto gC_tile = local_tile(gC_block, make_tile(Int<WarpM>{}, Int<WarpN>{}), make_coord(_, _)); // (WM, WN, iter_M, iter_N)
    auto gScaleA_block = local_tile(gScaleA, make_tile(Int<TileM>{}), make_coord(idx)); // (TM)
    auto gScaleB_block = local_tile(gScaleB, make_tile(Int<TileN>{}), make_coord(idy)); // (TN)

    extern __shared__ char shared_memory[];
    SharedStorage &smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr<int4b_t>(smem.smemA_4bit.data()), typename KTraits::smemALayout{}); // (TM, TK, NStage)
    Tensor sB = make_tensor(make_smem_ptr<int4b_t>(smem.smemB_4bit.data()), typename KTraits::smemBLayout{}); // (TN, TK, NStage)
    Tensor sC = make_tensor(make_smem_ptr<__nv_bfloat16>(smem.smemC.data()), typename KTraits::smemCLayout{}); // (WM, WN)
    Tensor sScaleA = make_tensor(make_smem_ptr<__nv_bfloat16>(smem.smemScaleA.data()), make_layout(
        make_shape(Int<TileM>{}, Int<TileN>{}),
        make_stride(_1{}, _0{})
    )); // (TM, TN)
    Tensor sScaleB = make_tensor(make_smem_ptr<__nv_bfloat16>(smem.smemScaleB.data()), make_layout(
        make_shape(Int<TileM>{}, Int<TileN>{}),
        make_stride(_0{}, _1{})
    )); // (TM, TN)
    
    using MMA = typename KTraits::MMA;
    using G2SCopyA = typename KTraits::G2SCopyA;
    using G2SCopyB = typename KTraits::G2SCopyB;
    using S2RCopyA = typename KTraits::S2RCopyA;
    using S2RCopyB = typename KTraits::S2RCopyB;
    using S2GCopyC = typename KTraits::S2GCopyC;

    auto sA_tile = local_tile(sA, make_tile(Int<WarpM>{}, Int<WarpK>{}), make_coord(_, _)); // (WM, WK, iter_M, iter_K, NStage)
    auto sB_tile = local_tile(sB, make_tile(Int<WarpN>{}, Int<WarpK>{}), make_coord(_, _)); // (WN, WK, iter_N, iter_K, NStage)
    auto sScaleA_tile = local_tile(sScaleA, make_tile(Int<WarpM>{}, Int<WarpN>{}), make_coord(_, _)); // (WM, WN, iter_M, iter_N)
    auto sScaleB_tile = local_tile(sScaleB, make_tile(Int<WarpM>{}, Int<WarpN>{}), make_coord(_, _)); // (WM, WN, iter_M, iter_N)

    MMA mma;
    auto mma_thread = mma.get_slice(threadIdx.x);
    Tensor local_A = mma_thread.partition_fragment_A(sA_tile(_, _, _, 0, 0)); // (MMA, MMA_WM, MMA_WK, iter_M)
    Tensor local_B = mma_thread.partition_fragment_B(sB_tile(_, _, _, 0, 0)); // (MMA, MMA_WN, MMA_WK, iter_N)
    Tensor local_C = partition_fragment_C(mma_thread, gC_tile.shape()); // (MMA, MMA_WM, MMA_WN, iter_M, iter_N)

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

    // reg local_C -> shared sC
    // direct copy and scale
    auto sC_thread = mma_thread.partition_C(sC); // (MMA, MMA_WM, MMA_WN)
    auto sScaleA_thread = mma_thread.partition_C(sScaleA_tile); // (MMA, MMA_WM, MMA_WN, iter_M, iter_N)
    auto sScaleB_thread = mma_thread.partition_C(sScaleB_tile); // (MMA, MMA_WM, MMA_WN, iter_M, iter_N)

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
    CUTE_STATIC_ASSERT(size(mma) >= TileM);
    CUTE_STATIC_ASSERT(size(mma) >= TileN);
    if (threadIdx.x < TileM) {
        sScaleA(threadIdx.x, 0) = gScaleA_block(threadIdx.x);
    }
    if (threadIdx.x < TileN) {
        sScaleB(0, threadIdx.x) = gScaleB_block(threadIdx.x);
    }
    __syncthreads();

    clear(local_C);

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

    auto load_block = [&] (int load_k) {
        int shared_idx = load_k % NStage;
        copy(copyA, gA_block_g2s_thread(_, _, _, load_k), sA_g2s_thread(_, _, _, shared_idx));
        copy(copyB, gB_block_g2s_thread(_, _, _, load_k), sB_g2s_thread(_, _, _, shared_idx));
        cp_async_fence();
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

    for (int i = 0; i < niters - NStage; i++) {
        cp_async_wait<NStage - 1>();
        __syncthreads();
        do_wmma(i % NStage);
        __syncthreads();
        load_block(i + NStage);
    }

    auto pipe_tail = [&] <int i>() {
        cp_async_wait<i>();
        __syncthreads();
        do_wmma((niters - 1 - i) % NStage);
    };

    CUTE_STATIC_ASSERT(NStage <= 3);
    if constexpr (NStage >= 3) pipe_tail.template operator()<2>();
    if constexpr (NStage >= 2) pipe_tail.template operator()<1>();
    if constexpr (NStage >= 1) pipe_tail.template operator()<0>();

    for (int i = 0; i < iter_m; i++) {
        for (int j = 0; j < iter_n; j++) {
            // copy(AutoVectorizingCopy{}, local_C(_, _, _, i, j), sC_thread);
            for (int _2 = 0; _2 < size(get<2>(local_C.shape())); _2++) {
                for (int _1 = 0; _1 < size(get<1>(local_C.shape())); _1++) {
                    for (int _0 = 0; _0 < size(get<0>(local_C.shape())); _0 += 2) {
                        __nv_bfloat16 c[2];
                        __nv_bfloat16 s1 = sScaleA_thread(_0, _1, _2, i, j) * sScaleB_thread(_0, _1, _2, i, j);
                        __nv_bfloat16 s2 = sScaleA_thread(_0 + 1, _1, _2, i, j) * sScaleB_thread(_0 + 1, _1, _2, i, j);
                        c[0] = (float) local_C(_0, _1, _2, i, j) * (float) s1;
                        c[1] = (float) local_C(_0 + 1, _1, _2, i, j) * (float) s2;
                        *reinterpret_cast<half2*>(&sC_thread(_0, _1, _2)) = *reinterpret_cast<half2*>(c);
                    }
                }
            }
            __syncthreads();
            copy(copyC, sC_s2g_thread, gC_tile_s2g_thread(_, _, _, i, j));
            __syncthreads();
        }
    }
}

template <
    int N, int K, int LDC,
    int TileM, int TileN, int TileK, int NStage,
    int WarpM, int WarpN, int WarpK
>
int call_gemm(
    int M,
    uint8_t *A, uint8_t *B, __nv_bfloat16 *C,
    __nv_bfloat16 *A_scale, __nv_bfloat16 *B_scale,
    cudaStream_t stream
) {
    using MMA_4bit = decltype(make_tiled_mma(
        SM80_16x8x64_S32S4S4S32_TN_SATURATE{},
        Layout<Shape<_2, _4, _1>>{},
        Tile<Int<WarpM>, Int<WarpN>, Int<WarpK>>{}
    ));

    constexpr int num_total_threads = size(MMA_4bit{});

    using smemALayout_4bit = decltype(composition(
        get_best_swizzle<TileK, 4>(),
        Layout<
            Shape<Int<TileM>, Int<TileK>, Int<NStage>>,
            Stride<Int<TileK>, _1, Int<TileM * TileK>>
        >{}
    ));
    using smemBLayout_4bit = decltype(composition(
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
    using sharedStorage = SharedStorage<smemALayout_4bit, smemBLayout_4bit, smemCLayout, TileM, TileN>;
    
    using G2SCopyA_4bit = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, int4b_t>{},
        make_right_layout(Shape<Int<num_total_threads / (TileK / 32)>, Int<TileK / 32>>{}),
        make_right_layout(Shape<_1, _32>{}) // load 32 int4b per thread (128bit)
    ));
    using G2SCopyB_4bit = G2SCopyA_4bit;

    using S2RCopyA_4bit = decltype(make_tiled_copy_A(
        Copy_Atom<SM75_U32x4_LDSM_N, int4b_t>{},
        MMA_4bit{}
    ));
    using S2RCopyB_4bit = decltype(make_tiled_copy_B(
        Copy_Atom<SM75_U32x4_LDSM_N, int4b_t>{},
        MMA_4bit{}
    ));

    using S2GCopyC = decltype(make_tiled_copy(
        Copy_Atom<AutoVectorizingCopy, __nv_bfloat16>{},
        make_right_layout(Shape<Int<num_total_threads / (WarpN / 8)>, Int<WarpN / 8>>{}),
        make_right_layout(Shape<_1, _8>{}) // load 8 half per thread (128bit)
    ));

    using KTraits = KernelTraits<
        MMA_4bit, G2SCopyA_4bit, G2SCopyB_4bit, S2RCopyA_4bit, S2RCopyB_4bit, S2GCopyC,
        smemALayout_4bit, smemBLayout_4bit, smemCLayout
    >;

    dim3 block(num_total_threads);
    dim3 grid(M / TileM, N / TileN);

    assert(M % TileM == 0);
    assert(N % TileN == 0);
    assert(K % TileK == 0);
    assert(K / TileK >= NStage - 1);

    auto smem_size = sizeof(sharedStorage);
    
    if (cudaFuncSetAttribute(
        gemm_kernel <
            N, K, LDC,
            TileM, TileN, TileK, NStage, WarpM, WarpN, WarpK,
            KTraits, sharedStorage
        >,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ) != cudaSuccess) return 1;

    gemm_kernel <
        N, K, LDC,
        TileM, TileN, TileK, NStage, WarpM, WarpN, WarpK,
        KTraits, sharedStorage
    > <<< grid, block, smem_size, stream >>> (
        M,
        A,
        B,
        C,
        A_scale,
        B_scale
    );

    return 0;
}