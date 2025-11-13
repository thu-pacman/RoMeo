#include <cstdio>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

#include "cute/tensor.hpp"

#include "cuda/utils.h"

using namespace cute;

template <typename smemALayout, typename smemBLayout, typename smemCLayout, int TempSize, int TileM, int TileN>
struct SharedStorage {
    array_aligned<int8_t, cosize_v<smemALayout>> smemA_8bit;
    array_aligned<int8_t, cosize_v<smemBLayout>> smemB_8bit;
    array_aligned<__nv_bfloat16, cosize_v<smemCLayout>> smemC;
    array_aligned<__nv_bfloat16, TileM> smemScaleA;
    array_aligned<__nv_bfloat16, TileN> smemScaleB;
    array_aligned<int8_t, TempSize / 2> smemT_4bit;
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

inline __device__ void decode_16x_int4_to_int8(uint64_t &packed_int4s, uint128_t *int8s) {
    int8_t *packed_int4s_ptr = reinterpret_cast<int8_t*>(&packed_int4s);
    int8_t *int8s_ptr = reinterpret_cast<int8_t*>(int8s);
    for (int i = 0; i < 16; i++) {
        int8s_ptr[i] = (packed_int4s_ptr[i / 2] >> ((i % 2) * 4)) & 0x0F;
        int8s_ptr[i] = ((int8s_ptr[i] ^ 0x08) - 0x08);
    }
}

template <
    typename KTraits,
    int K, int TileM, int TileN, int TileK, int NStage, int WarpM, int WarpN, int WarpK,
    bool castA, bool castB,
    typename SharedStorage,
    typename TgA_block, typename TgB_block, typename TgC_tile,
    typename TgScaleA_block, typename TgScaleB_block,
    typename TsA, typename TsB
>
__device__ void gemm_kernel_inner(
    SharedStorage &smem, TgA_block &gA_block, TgB_block &gB_block, TgC_tile &gC_tile, TgScaleA_block &gScaleA_block, TgScaleB_block &gScaleB_block, TsA &sA, TsB &sB
) {
    using MMA = typename KTraits::MMA;
    using G2SCopyA = typename KTraits::G2SCopyA;
    using G2SCopyB = typename KTraits::G2SCopyB;
    using S2RCopyA = typename KTraits::S2RCopyA;
    using S2RCopyB = typename KTraits::S2RCopyB;
    using S2GCopyC = typename KTraits::S2GCopyC;
    using smemCLayout = typename KTraits::smemCLayout;

    __nv_bfloat16 *smemC = smem.smemC.data();
    __nv_bfloat16 *smemScaleA = smem.smemScaleA.data();
    __nv_bfloat16 *smemScaleB = smem.smemScaleB.data();
    
    Tensor sC = make_tensor(make_smem_ptr<__nv_bfloat16>(smemC), smemCLayout{}); // (WM, WN)
    Tensor sScaleA = make_tensor(make_smem_ptr(smemScaleA), make_layout(
        make_shape(Int<TileM>{}, Int<TileN>{}),
        make_stride(_1{}, _0{})
    )); // (TM, TN)
    Tensor sScaleB = make_tensor(make_smem_ptr(smemScaleB), make_layout(
        make_shape(Int<TileM>{}, Int<TileN>{}),
        make_stride(_0{}, _1{})
    )); // (TM, TN)
    using smemTALayout = Layout<
        Shape<Int<TileM>, Int<TileK>, Int<NStage>>,
        Stride<Int<TileK>, _1, Int<TileM * TileK>>
    >;
    using smemTBLayout = Layout<
        Shape<Int<TileN>, Int<TileK>, Int<NStage>>,
        Stride<Int<TileK>, _1, Int<TileN * TileK>>
    >;
    Tensor sTA = make_tensor(make_smem_ptr<int4b_t>(smem.smemT_4bit.data()), smemTALayout{}); // (TM, TK, NStage)
    Tensor sTB = make_tensor(make_smem_ptr<int4b_t>(smem.smemT_4bit.data()), smemTBLayout{}); // (TN, TK, NStage)

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
        if constexpr (castA) {
            int total_ops = TileM * TileK / 16;
            for (int i = threadIdx.x; i < total_ops; i += size(mma)) {
                int im = i * 16 / TileK;
                int ik = i * 16 % TileK;
                uint64_t packed_int4s = *reinterpret_cast<uint64_t*>(raw_pointer_cast(&sTA(im, ik, k_idx)));
                uint128_t int8s;
                decode_16x_int4_to_int8(packed_int4s, &int8s);
                *reinterpret_cast<uint128_t*>(&sA(im, ik, k_idx)) = int8s;
            }
            __syncthreads();
        }
        if constexpr (castB) {
            int total_ops = TileN * TileK / 16;
            for (int i = threadIdx.x; i < total_ops; i += size(mma)) {
                int in = i * 16 / TileK;
                int ik = i * 16 % TileK;
                uint64_t packed_int4s = *reinterpret_cast<uint64_t*>(raw_pointer_cast(&sTB(in, ik, k_idx)));
                uint128_t int8s;
                decode_16x_int4_to_int8(packed_int4s, &int8s);
                *reinterpret_cast<uint128_t*>(&sB(in, ik, k_idx)) = int8s;
            }
            __syncthreads();
        }
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
        if constexpr (castA) {
            using G2SCopyA_4bit = decltype(make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, int4b_t>{},
                make_right_layout(Shape<Int<size(mma) / (TileK / 32)>, Int<TileK / 32>>{}),
                make_right_layout(Shape<_1, _32>{}) // load 32 int4b per thread (128bit)
            ));
            G2SCopyA_4bit copyA;
            auto copyA_thread = copyA.get_slice(threadIdx.x);
            auto gA_block_g2s_thread = copyA_thread.partition_S(gA_block); // (COPY, COPY_TM, COPY_TK, niters)
            auto sA_g2s_thread = copyA_thread.partition_D(sTA); // (COPY, COPY_TM, COPY_TK, NStage)
            copy(copyA, gA_block_g2s_thread(_, _, _, load_k), sA_g2s_thread(_, _, _, shared_idx));
        } else {
            copy(copyA, gA_block_g2s_thread(_, _, _, load_k), sA_g2s_thread(_, _, _, shared_idx));
        }
        if constexpr (castB) {
            using G2SCopyB_4bit = decltype(make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, int4b_t>{},
                make_right_layout(Shape<Int<size(mma) / (TileK / 32)>, Int<TileK / 32>>{}),
                make_right_layout(Shape<_1, _32>{}) // load 32 int4b per thread (128bit)
            ));
            G2SCopyB_4bit copyB;
            auto copyB_thread = copyB.get_slice(threadIdx.x);
            auto gB_block_g2s_thread = copyB_thread.partition_S(gB_block); // (COPY, COPY_TN, COPY_TK, niters)
            auto sB_g2s_thread = copyB_thread.partition_D(sTB); // (COPY, COPY_TN, COPY_TK, NStage)
            copy(copyB, gB_block_g2s_thread(_, _, _, load_k), sB_g2s_thread(_, _, _, shared_idx));
        } else {
            copy(copyB, gB_block_g2s_thread(_, _, _, load_k), sB_g2s_thread(_, _, _, shared_idx));
        }
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
    int N, int K, int Outlier_B,
    int TileM, int TileN, int TileK, int NStage, int WarpM, int WarpN, int WarpK,
    typename KTraits_8bit, typename KTraits_4bit, typename SharedStorage
>
__global__ void gemm_kernel(
    int M, __nv_bfloat16 *C,
    uint8_t *A, uint8_t *B, uint8_t *A_outlier, uint8_t *B_outlier,
    __nv_bfloat16 *A_scale, __nv_bfloat16 *B_scale, __nv_bfloat16 *A_scale_outlier, __nv_bfloat16 *B_scale_outlier,
    int Outlier_A
) {
    int idx = blockIdx.x, idy = blockIdx.y;
    int c_idx = idx, c_idy = idy;

    bool isInt8A = false, isInt8B = false;

    if (idx >= M / TileM) { // Load Int8 A
        idx -= M / TileM;
        isInt8A = true;
    }
    if (idy >= N / TileN) { // Load Int8 B
        idy -= N / TileN;
        isInt8B = true;
    }

    Tensor gA_8bit = make_tensor(make_gmem_ptr<int8_t>(A_outlier), make_layout(make_shape(Outlier_A, Int<K>{}), make_stride(Int<K>{}, _1{}))); // (M, K)
    Tensor gB_8bit = make_tensor(make_gmem_ptr<int8_t>(B_outlier), make_layout(make_shape(Int<Outlier_B>{}, Int<K>{}), make_stride(Int<K>{}, _1{}))); // (N, K)
    Tensor gScaleA_8bit = make_tensor(make_gmem_ptr<__nv_bfloat16>(A_scale_outlier), make_layout(make_shape(Outlier_A))); // (M)
    Tensor gScaleB_8bit = make_tensor(make_gmem_ptr<__nv_bfloat16>(B_scale_outlier), make_layout(make_shape(Int<Outlier_B>{}))); // (N)

    Tensor gA_4bit = make_tensor(make_gmem_ptr<int4b_t>(A), make_layout(make_shape(M, Int<K>{}), make_stride(Int<K>{}, _1{}))); // (M, K)
    Tensor gB_4bit = make_tensor(make_gmem_ptr<int4b_t>(B), make_layout(make_shape(Int<N>{}, Int<K>{}), make_stride(Int<K>{}, _1{}))); // (N, K)
    Tensor gScaleA_4bit = make_tensor(make_gmem_ptr<__nv_bfloat16>(A_scale), make_layout(make_shape(M))); // (M)
    Tensor gScaleB_4bit = make_tensor(make_gmem_ptr<__nv_bfloat16>(B_scale), make_layout(make_shape(Int<N>{}))); // (N)

    Tensor gC = make_tensor(make_gmem_ptr<__nv_bfloat16>(C), make_layout(make_shape(M + Outlier_A, Int<N + Outlier_B>{}), make_stride(Int<N + Outlier_B>{}, _1{}))); // (M, N)
    auto gC_block = local_tile(gC, make_tile(Int<TileM>{}, Int<TileN>{}), make_coord(c_idx, c_idy)); // (TM, TN)
    auto gC_tile = local_tile(gC_block, make_tile(Int<WarpM>{}, Int<WarpN>{}), make_coord(_, _)); // (WM, WN, iter_M, iter_N)

    extern __shared__ char shared_memory[];
    SharedStorage &smem = *reinterpret_cast<SharedStorage*>(shared_memory);

    void *smemA = reinterpret_cast<void*>(smem.smemA_8bit.data());
    void *smemB = reinterpret_cast<void*>(smem.smemB_8bit.data());
    Tensor sA_8bit = make_tensor(make_smem_ptr<int8_t>(smemA), typename KTraits_8bit::smemALayout{}); // (TM, TK, NStage)
    Tensor sB_8bit = make_tensor(make_smem_ptr<int8_t>(smemB), typename KTraits_8bit::smemBLayout{}); // (TN, TK, NStage)
    Tensor sA_4bit = make_tensor(make_smem_ptr<int4b_t>(smemA), typename KTraits_4bit::smemALayout{}); // (TM, TK, NStage)
    Tensor sB_4bit = make_tensor(make_smem_ptr<int4b_t>(smemB), typename KTraits_4bit::smemBLayout{}); // (TN, TK, NStage)

    if (!isInt8A && !isInt8B) {
        auto gA_block = local_tile(gA_4bit, make_tile(Int<TileM>{}, Int<TileK>{}), make_coord(idx, _)); // (TM, TK, niters)
        auto gB_block = local_tile(gB_4bit, make_tile(Int<TileN>{}, Int<TileK>{}), make_coord(idy, _)); // (TN, TK, niters)
        auto gScaleA_block = local_tile(gScaleA_4bit, make_tile(Int<TileM>{}), make_coord(idx)); // (TM)
        auto gScaleB_block = local_tile(gScaleB_4bit, make_tile(Int<TileN>{}), make_coord(idy)); // (TN)
        gemm_kernel_inner <
            KTraits_4bit,
            K, TileM, TileN, TileK, NStage, WarpM, WarpN, WarpK, false, false
        > (smem, gA_block, gB_block, gC_tile, gScaleA_block, gScaleB_block, sA_4bit, sB_4bit);
    } else if (isInt8A && isInt8B) {
        auto gA_block = local_tile(gA_8bit, make_tile(Int<TileM>{}, Int<TileK>{}), make_coord(idx, _)); // (TM, TK, niters)
        auto gB_block = local_tile(gB_8bit, make_tile(Int<TileN>{}, Int<TileK>{}), make_coord(idy, _)); // (TN, TK, niters)
        auto gScaleA_block = local_tile(gScaleA_8bit, make_tile(Int<TileM>{}), make_coord(idx)); // (TM)
        auto gScaleB_block = local_tile(gScaleB_8bit, make_tile(Int<TileN>{}), make_coord(idy)); // (TN)
        gemm_kernel_inner <
            KTraits_8bit,
            K, TileM, TileN, TileK, NStage, WarpM, WarpN, WarpK, false, false
        > (smem, gA_block, gB_block, gC_tile, gScaleA_block, gScaleB_block, sA_8bit, sB_8bit);
    } else if (isInt8A && !isInt8B) {
        auto gA_block = local_tile(gA_8bit, make_tile(Int<TileM>{}, Int<TileK>{}), make_coord(idx, _)); // (TM, TK, niters)
        auto gB_block = local_tile(gB_4bit, make_tile(Int<TileN>{}, Int<TileK>{}), make_coord(idy, _)); // (TN, TK, niters)
        auto gScaleA_block = local_tile(gScaleA_8bit, make_tile(Int<TileM>{}), make_coord(idx)); // (TM)
        auto gScaleB_block = local_tile(gScaleB_4bit, make_tile(Int<TileN>{}), make_coord(idy)); // (TN)
        gemm_kernel_inner <
            KTraits_8bit,
            K, TileM, TileN, TileK, NStage, WarpM, WarpN, WarpK, false, true
        > (smem, gA_block, gB_block, gC_tile, gScaleA_block, gScaleB_block, sA_8bit, sB_8bit);
    } else if (!isInt8A && isInt8B) {
        auto gA_block = local_tile(gA_4bit, make_tile(Int<TileM>{}, Int<TileK>{}), make_coord(idx, _)); // (TM, TK, niters)
        auto gB_block = local_tile(gB_8bit, make_tile(Int<TileN>{}, Int<TileK>{}), make_coord(idy, _)); // (TN, TK, niters)
        auto gScaleA_block = local_tile(gScaleA_4bit, make_tile(Int<TileM>{}), make_coord(idx)); // (TM)
        auto gScaleB_block = local_tile(gScaleB_8bit, make_tile(Int<TileN>{}), make_coord(idy)); // (TN)
        gemm_kernel_inner <
            KTraits_8bit,
            K, TileM, TileN, TileK, NStage, WarpM, WarpN, WarpK, true, false
        > (smem, gA_block, gB_block, gC_tile, gScaleA_block, gScaleB_block, sA_8bit, sB_8bit);
    }
}

template <int A, int B>
constexpr int static_max() {
    if constexpr (A > B) return A;
    else return B;
}

template <
    int N, int K, int Outlier_B,
    int TileM, int TileN, int TileK, int NStage,
    int WarpM, int WarpN, int WarpK
>
int call_gemm(
    int M, __nv_bfloat16 *C,
    uint8_t *A, uint8_t *B, uint8_t *A_outlier, uint8_t *B_outlier,
    __nv_bfloat16 *A_scale, __nv_bfloat16 *B_scale, __nv_bfloat16 *A_scale_outlier, __nv_bfloat16 *B_scale_outlier,
    int Outlier_A,
    cudaStream_t stream
) {
    using MMA_8bit = decltype(make_tiled_mma(
        SM80_16x8x32_S32S8S8S32_TN_SATURATE{},
        Layout<Shape<_2, _4, _1>>{},
        Tile<Int<WarpM>, Int<WarpN>, Int<WarpK>>{}
    ));
    using MMA_4bit = decltype(make_tiled_mma(
        SM80_16x8x64_S32S4S4S32_TN_SATURATE{},
        Layout<Shape<_2, _4, _1>>{},
        Tile<Int<WarpM>, Int<WarpN>, Int<WarpK>>{}
    ));

    constexpr int num_total_threads = size(MMA_8bit{});
    static_assert(num_total_threads == size(MMA_4bit{}), "MMA_8bit and MMA_4bit must have the same number of threads");

    using smemALayout_8bit = decltype(composition(
        get_best_swizzle<TileK, 8>(),
        Layout<
            Shape<Int<TileM>, Int<TileK>, Int<NStage>>,
            Stride<Int<TileK>, _1, Int<TileM * TileK>>
        >{}
    ));
    using smemBLayout_8bit = decltype(composition(
        get_best_swizzle<TileK, 8>(),
        Layout<
            Shape<Int<TileN>, Int<TileK>, Int<NStage>>,
            Stride<Int<TileK>, _1, Int<TileN * TileK>>
        >{}
    ));
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
    constexpr int smemT_size = static_max<TileM, TileN>() * TileK * NStage;
    using sharedStorage = SharedStorage<smemALayout_8bit, smemBLayout_8bit, smemCLayout, smemT_size, TileM, TileN>;

    using G2SCopyA_8bit = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, int8_t>{},
        make_right_layout(Shape<Int<num_total_threads / (TileK / 16)>, Int<TileK / 16>>{}),
        make_right_layout(Shape<_1, _16>{}) // load 16 int8_t per thread (128bit)
    ));
    using G2SCopyB_8bit = G2SCopyA_8bit;
    using G2SCopyA_4bit = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, int4b_t>{},
        make_right_layout(Shape<Int<num_total_threads / (TileK / 32)>, Int<TileK / 32>>{}),
        make_right_layout(Shape<_1, _32>{}) // load 32 int4b per thread (128bit)
    ));
    using G2SCopyB_4bit = G2SCopyA_4bit;

    using S2RCopyA_8bit = decltype(make_tiled_copy_A(
        Copy_Atom<SM75_U32x4_LDSM_N, int8_t>{},
        MMA_8bit{}
    ));
    using S2RCopyB_8bit = decltype(make_tiled_copy_B(
        Copy_Atom<SM75_U32x4_LDSM_N, int8_t>{},
        MMA_8bit{}
    ));
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

    using KTraits_8bit = KernelTraits<
        MMA_8bit, G2SCopyA_8bit, G2SCopyB_8bit, S2RCopyA_8bit, S2RCopyB_8bit, S2GCopyC,
        smemALayout_8bit, smemBLayout_8bit, smemCLayout
    >;
    using KTraits_4bit = KernelTraits<
        MMA_4bit, G2SCopyA_4bit, G2SCopyB_4bit, S2RCopyA_4bit, S2RCopyB_4bit, S2GCopyC,
        smemALayout_4bit, smemBLayout_4bit, smemCLayout
    >;

    dim3 block(num_total_threads);
    dim3 grid((M + Outlier_A) / TileM, (N + Outlier_B) / TileN);

    assert(M % TileM == 0);
    assert(N % TileN == 0);
    assert(K % TileK == 0);
    assert(K / TileK >= NStage - 1);
    assert(Outlier_A % TileM == 0);
    assert(Outlier_B % TileN == 0);

    auto smem_size = sizeof(sharedStorage);
    
    if (cudaFuncSetAttribute(
        gemm_kernel <
            N, K, Outlier_B,
            TileM, TileN, TileK, NStage, WarpM, WarpN, WarpK,
            KTraits_8bit, KTraits_4bit, sharedStorage
        >,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ) != cudaSuccess) return 1;

    gemm_kernel <
        N, K, Outlier_B,
        TileM, TileN, TileK, NStage, WarpM, WarpN, WarpK,
        KTraits_8bit, KTraits_4bit, sharedStorage
    > <<< grid, block, smem_size, stream >>> (
        M,
        C,
        A,
        B,
        A_outlier,
        B_outlier,
        A_scale,
        B_scale,
        A_scale_outlier,
        B_scale_outlier,
        Outlier_A
    );

    return 0;
}