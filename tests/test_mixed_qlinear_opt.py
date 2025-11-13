import math
import torch
import qfactory

def test_mixed_precision(M, N, K, outlier):
    activation = torch.randn(M, K, dtype=torch.float16, device='cuda') / math.sqrt(M)
    weight = torch.randn(N, K, dtype=torch.float16, device='cuda') / math.sqrt(N)

    mixed_qlinear = qfactory.MixedQLinear(weight, outlier)
    opt_mixed_qlinear = qfactory.OptMixedQLinear(weight, outlier)
    opt_mixed_qlinear_multistream = qfactory.OptMixedQLinear(weight, outlier, multistream=True)

    mixed_output = mixed_qlinear(activation)
    mixed_output_opt = opt_mixed_qlinear(activation)
    mixed_output_opt_multistream = opt_mixed_qlinear_multistream(activation)

    print(mixed_output)
    print(mixed_output_opt)
    print(mixed_output_opt_multistream)

    from qfactory.profile import profile_latency
    print("Mixed QLinear Latency:", profile_latency(lambda: mixed_qlinear(activation)))
    print("Opt Mixed QLinear Latency:", profile_latency(lambda: opt_mixed_qlinear(activation)))
    print("Opt Mixed QLinear Multistream Latency:", profile_latency(lambda: opt_mixed_qlinear_multistream(activation)))

    torch.testing.assert_close(mixed_output, mixed_output_opt)
    torch.testing.assert_close(mixed_output, mixed_output_opt_multistream)

if __name__ == '__main__':
    torch.manual_seed(42)

    test_mixed_precision(8192, 8192, 8192, 256)

    print('PASSED')
