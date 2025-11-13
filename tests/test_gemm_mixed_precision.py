import math
import torch
import qfactory

def test_mixed_precision(M, N, K, outlier):
    activation = torch.randint(0, 256, (M + outlier * 2, K // 2), dtype=torch.uint8, device='cuda')
    activation_scale = torch.randn(M + outlier, dtype=torch.bfloat16, device='cuda') / math.sqrt(M)
    weight = torch.randint(0, 256, (N + outlier * 2, K // 2), dtype=torch.uint8, device='cuda')
    weight_scale = torch.randn(N + outlier, dtype=torch.bfloat16, device='cuda') / math.sqrt(N)
    output = torch.empty(M + outlier, N + outlier, dtype=torch.bfloat16, device='cuda')
    naive_output = torch.empty_like(output)
    multistream_output = torch.empty_like(output)

    qfactory.gemm_int4_int4_nt_mixed_precision(
        activation[:M], activation_scale[:M],
        activation[M:].view(outlier, K), activation_scale[M:],
        weight[:N], weight_scale[:N],
        weight[N:].view(outlier, K), weight_scale[N:],
        output, outlier, outlier
    )
    qfactory.gemm_int4_int4_nt_mixed_precision_multistream(activation, activation_scale, weight, weight_scale, multistream_output, outlier)
    qfactory.gemm_int4_int4_nt_mixed_precision_naive(activation, activation_scale, weight, weight_scale, naive_output, outlier)
    print(naive_output)
    print(multistream_output)
    print(output)

    from qfactory.profile import profile_latency
    print("Naive latency:", profile_latency(lambda: qfactory.gemm_int4_int4_nt_mixed_precision_naive(activation, activation_scale, weight, weight_scale, naive_output, outlier)))
    print("Multistream latency:", profile_latency(lambda: qfactory.gemm_int4_int4_nt_mixed_precision_multistream(activation, activation_scale, weight, weight_scale, multistream_output, outlier)))
    print("Mixed precision latency:", profile_latency(lambda: qfactory.gemm_int4_int4_nt_mixed_precision(
        activation[:M], activation_scale[:M],
        activation[M:].view(outlier, K), activation_scale[M:],
        weight[:N], weight_scale[:N],
        weight[N:].view(outlier, K), weight_scale[N:],
        output, outlier, outlier
    )))

    torch.testing.assert_close(output, naive_output)
    torch.testing.assert_close(multistream_output, naive_output)

def test_mixed_precision_separate(M, N, K, outlier):
    activation = torch.randint(0, 256, (M + outlier * 2, K // 2), dtype=torch.uint8, device='cuda')
    activation_scale = torch.randn(M + outlier, dtype=torch.bfloat16, device='cuda') / math.sqrt(M)
    weight = torch.randint(0, 256, (N + outlier * 2, K // 2), dtype=torch.uint8, device='cuda')
    weight_scale = torch.randn(N + outlier, dtype=torch.bfloat16, device='cuda') / math.sqrt(N)
    output = torch.empty(M + outlier, N + outlier, dtype=torch.bfloat16, device='cuda')
    naive_output = torch.empty_like(output)

    qfactory.gemm_int4_int4_nt_mixed_precision_separate(
        activation[:M], activation_scale[:M],
        activation[M:].view(outlier, K), activation_scale[M:],
        weight[:N], weight_scale[:N],
        weight[N:].view(outlier, K), weight_scale[N:],
        output, outlier, outlier, [torch.cuda.current_stream() for _ in range(4)]
    )
    qfactory.gemm_int4_int4_nt_mixed_precision_naive(activation, activation_scale, weight, weight_scale, naive_output, outlier)
    
    print(naive_output)
    print(output)

    from qfactory.profile import profile_latency
    print("Separate latency:", profile_latency(lambda: qfactory.gemm_int4_int4_nt_mixed_precision_separate(
        activation[:M], activation_scale[:M],
        activation[M:].view(outlier, K), activation_scale[M:],
        weight[:N], weight_scale[:N],
        weight[N:].view(outlier, K), weight_scale[N:],
        output, outlier, outlier, [torch.cuda.current_stream() for _ in range(4)]
    )))

    torch.testing.assert_close(output, naive_output)

if __name__ == '__main__':
    torch.manual_seed(42)

    test_mixed_precision(8192, 8192, 8192, 1024)
    test_mixed_precision_separate(8192, 8192, 8192, 1024)

    print('PASSED')
