import math
import torch
import qfactory

def test_w4a4_perchannel(M, N, K):
    activation = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device='cuda')
    activation_scale = torch.randn(M, dtype=torch.float32, device='cuda')
    weight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device='cuda')
    weight_scale = torch.randn(N, dtype=torch.float32, device='cuda')
    output = torch.empty(M, N, dtype=torch.float16, device='cuda')
    naive_output = torch.empty_like(output)

    qfactory.gemm_int4_int4_nt_perchannel(activation, activation_scale, weight, weight_scale, output)
    qfactory.gemm_int4_int4_nt_perchannel_naive(activation, activation_scale, weight, weight_scale, naive_output)
    print(naive_output)
    print(output)
    torch.testing.assert_close(output, naive_output)

def test_w4a4_pergroup(M, N, K, group_k):
    activation = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device='cuda')
    activation_scale = torch.randn(M, K // group_k, dtype=torch.float16, device='cuda')
    weight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device='cuda')
    weight_scale = torch.randn(N, K // group_k, dtype=torch.float16, device='cuda')
    output = torch.empty(M, N, dtype=torch.float16, device='cuda')
    naive_output = torch.empty_like(output)

    qfactory.gemm_int4_int4_nt_pergroup(activation, activation_scale, weight, weight_scale, output, group_k)
    qfactory.gemm_int4_int4_nt_pergroup_naive(activation, activation_scale, weight, weight_scale, naive_output, group_k)
    print(naive_output)
    print(output)
    torch.testing.assert_close(output, naive_output)

def test_w8a8_perchannel(M, N, K):
    activation = torch.randint(-128, 128, (M, K), dtype=torch.int8, device='cuda')
    activation_scale = torch.randn(M, dtype=torch.float16, device='cuda') / math.sqrt(M)
    weight = torch.randint(-128, 128, (N, K), dtype=torch.int8, device='cuda')
    weight_scale = torch.randn(N, dtype=torch.float16, device='cuda') / math.sqrt(N)
    output = torch.empty(M, N, dtype=torch.float16, device='cuda')
    naive_output = torch.empty_like(output)

    qfactory.gemm_int8_int8_nt_perchannel(activation, activation_scale, weight, weight_scale, output)
    qfactory.gemm_int8_int8_nt_perchannel_naive(activation, activation_scale, weight, weight_scale, naive_output)
    print(naive_output)
    print(output)
    torch.testing.assert_close(output, naive_output)

if __name__ == '__main__':
    torch.manual_seed(42)

    test_w4a4_perchannel(1024, 512, 7168)
    test_w4a4_pergroup(1024, 512, 7168, 256)
    test_w8a8_perchannel(1024, 512, 7168)

    print('PASSED')
