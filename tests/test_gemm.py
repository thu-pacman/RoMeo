import torch
import qfactory

def test_w16a16(M, N, K):
    activation = torch.randn(M, K, dtype=torch.float16, device='cuda')
    weight = torch.randn(N, K, dtype=torch.float16, device='cuda')
    output = torch.empty(M, N, dtype=torch.float16, device='cuda')
    naive_output = torch.empty_like(output)

    qfactory.gemm_fp16_fp16_nt(activation, weight, output)
    qfactory.gemm_fp16_fp16_nt_naive(activation, weight, naive_output)

    print(naive_output)
    print(output)
    assert torch.allclose(output, naive_output)

def test_w4a16(M, N, K):
    activation = torch.randn(M, K, dtype=torch.float16, device='cuda')
    weight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device='cuda')
    output = torch.empty(M, N, dtype=torch.float16, device='cuda')
    naive_output = torch.empty_like(output)

    qfactory.gemm_int4_fp16_nt(activation, weight, output)
    qfactory.gemm_int4_fp16_nt_naive(activation, weight, naive_output)

    print(naive_output)
    print(output)
    assert torch.allclose(output, naive_output)

def test_w4a4(M, N, K):
    activation = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device='cuda')
    weight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device='cuda')
    output = torch.empty(M, N, dtype=torch.int32, device='cuda')
    naive_output = torch.empty_like(output)
    cutlass_output = torch.empty_like(output)

    qfactory.gemm_int4_int4_nt(activation, weight, output)
    qfactory.gemm_int4_int4_nt_naive(activation, weight, naive_output)
    qfactory.gemm_int4_int4_nt_cutlass(activation, weight, cutlass_output)
    print(cutlass_output)
    print(naive_output)
    print(output)
    torch.testing.assert_close(cutlass_output, naive_output)
    torch.testing.assert_close(output, naive_output)

if __name__ == '__main__':
    torch.manual_seed(42)

    test_w16a16(1024, 512, 7168)

    test_w4a16(1024, 512, 7168)
    
    test_w4a4(1024, 512, 7168)

    print('PASSED')