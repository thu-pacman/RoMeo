#include "cutlass/gemm/device/gemm.h"

int call_gemm(
    int M, int N, int K,
    uint8_t *A, uint8_t *B, int32_t *C,
    cudaStream_t stream
) {
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::int4b_t, cutlass::layout::RowMajor,
        cutlass::int4b_t, cutlass::layout::ColumnMajor,
        int32_t, cutlass::layout::RowMajor,
        int32_t,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80
    >;

    Gemm gemm_op;

    typename Gemm::Arguments arguments{
        {M, N, K}, // problem size
        {reinterpret_cast<cutlass::int4b_t*>(A), K},    // A
        {reinterpret_cast<cutlass::int4b_t*>(B), K},    // B
        {C, N},    // C
        {C, N},    // D
        {1.0f, 0.0f} // alpha, beta
    };

    auto status = gemm_op(arguments, nullptr, stream);

    if (status != cutlass::Status::kSuccess) {
        return -1;
    }
    return 0;
}