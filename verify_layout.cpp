#include "cute/tensor.hpp"

using namespace cute;

int main() {
    int M = 4096;
    constexpr int N = 512;
    constexpr int K = 7168;
    constexpr int TM = 128;
    constexpr int TN = 256;
    constexpr int TK = 128;
    constexpr int WM = 64;
    constexpr int WN = 64;
    constexpr int WK = 128;
    constexpr int NStage = 3;

    using LayoutgA = decltype(make_layout(make_shape(M, Int<K>{}), make_stride(Int<K>{}, _1{})));
    Tensor gA = make_tensor(make_gmem_ptr<int4b_t>(nullptr), LayoutgA{}); // (M, K)
    Tensor gA_block = local_tile(gA, make_tile(Int<TM>{}, Int<TK>{}), make_coord(0, _)); // (TM, TK, global_iter_K)
    
    using LayoutsA = decltype(make_layout(make_shape(Int<TM>{}, Int<TK>{}, Int<NStage>{}), make_stride(Int<TK>{}, _1{}, Int<TM * TK>{})));
    Tensor sA = make_tensor(make_smem_ptr<int4b_t>(nullptr), LayoutsA{}); // (TM, TK, NStage)
    Tensor sA_tile = local_tile(sA, make_tile(Int<WM>{}, Int<WK>{}), make_coord(_, _)); // (WM, WK, iter_M, iter_K, NStage)

    using LayoutgC = decltype(make_layout(make_shape(M, Int<N>{}), make_stride(Int<N>{}, _1{})));
    Tensor gC = make_tensor(make_gmem_ptr<int32_t>(nullptr), LayoutgC{}); // (M, N)
    Tensor gC_block = local_tile(gC, make_tile(Int<TM>{}, Int<TN>{}), make_coord(0, 0)); // (TM, TN)
    Tensor gC_tile = local_tile(gC_block, make_tile(Int<WM>{}, Int<WN>{}), make_coord(_, _)); // (WM, WN, iter_M, iter_N)
    
    using LayoutsC = decltype(make_layout(make_shape(Int<WM>{}, Int<WN>{}), make_stride(Int<WN>{}, _1{})));
    Tensor sC = make_tensor(make_smem_ptr<int32_t>(nullptr), LayoutsC{}); // (WM, WN)

    using MMA = decltype(make_tiled_mma(
        SM80_16x8x64_S32S4S4S32_TN{},
        Layout<Shape<_2, _4, _1>>{},
        Tile<Int<WM>, Int<WN>, Int<WK>>{}
    ));

    auto mma_thread = MMA{}.get_slice(0);
    Tensor local_A = mma_thread.partition_fragment_A(sA_tile); // (MMA, MMA_WM, MMA_WK, iter_M, iter_K, NStage)
    Tensor local_C = partition_fragment_C(mma_thread, Shape<Int<WM>, Int<WN>, Int<TM / WM>, Int<TN / WN>>{}); // (MMA, MMA_WM, MMA_WN, iter_M, iter_N)

    using S2RCopyA = decltype(make_tiled_copy_A(
        Copy_Atom<SM75_U32x4_LDSM_N, int4b_t>{},
        MMA{}
    ));
    
    auto copyA_s2r_thread = S2RCopyA{}.get_slice(0);
    Tensor sA_tile_s2r_thread = copyA_s2r_thread.partition_S(sA_tile); // (COPY, COPY_M, COPY_K, iter_M, iter_K, NStage)
    Tensor local_A_s2r_view = copyA_s2r_thread.retile_D(local_A); // (COPY, COPY_M, COPY_K, iter_M, iter_K)

    auto sC_thread = mma_thread.partition_C(sC); // (MMA, MMA_WM, MMA_WN)


    printf("gA :"); print(gA); printf("\n");
    printf("gA_block :"); print(gA_block); printf("\n");
    printf(" ↓ \n");
    printf("sA :"); print(sA); printf("\n");
    printf("sA_tile :"); print(sA_tile); printf("\n");
    printf("sA_tile_s2r_thread :"); print(sA_tile_s2r_thread); printf("\n");
    printf(" ↓ \n");
    printf("local_A_s2r_view :"); print(local_A_s2r_view); printf("\n");
    printf("local_A :"); print(local_A); printf("\n");
    printf("local_C :"); print(local_C); printf("\n");
    printf(" ↓ \n");
    printf("sC_thread :"); print(sC_thread); printf("\n");
    printf("sC :"); print(sC); printf("\n");
    printf(" ↓ \n");
    printf("gC_tile :"); print(gC_tile); printf("\n");
    printf("gC_block :"); print(gC_block); printf("\n");
    printf("gC :"); print(gC); printf("\n");
    
    return 0;
}