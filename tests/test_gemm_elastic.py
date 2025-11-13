import torch
import qfactory

def test_w4a4_elastic(M, N, K, outlier, div):
    num_groups = 1 + div
    activation = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device='cuda')
    activation_scale = torch.randn(M, num_groups, dtype=torch.float16, device='cuda')
    weight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device='cuda')
    weight_scale = torch.randn(N, num_groups, dtype=torch.float16, device='cuda')
    output = torch.empty(M, N, dtype=torch.float16, device='cuda')
    naive_output = torch.empty_like(output)

    qfactory.gemm_int4_int4_nt_elastic(activation, activation_scale, weight, weight_scale, output, outlier, div)
    qfactory.gemm_int4_int4_nt_elastic_naive(activation, activation_scale, weight, weight_scale, naive_output, outlier, div)
    print(naive_output)
    print(output)
    torch.testing.assert_close(output, naive_output)

if __name__ == '__main__':
    torch.manual_seed(42)

    test_w4a4_elastic(1024, 512, 2304, 256, 4)

    print('PASSED')
