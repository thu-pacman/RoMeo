import math
import torch
import qfactory

def test_mixed_precision(M, N, K, outlier):
    activation = torch.randn(M, K, dtype=torch.float16, device='cuda') / math.sqrt(M)
    weight = torch.randn(N, K, dtype=torch.float16, device='cuda') / math.sqrt(N)

    q8linear = qfactory.QLinear(weight, 8)
    q4linear = qfactory.QLinear(weight, 4)
    mixed_qlinear = qfactory.MixedQLinear(weight, outlier)

    output8 = q8linear(activation)
    output4 = q4linear(activation)
    mixed_output = mixed_qlinear(activation)
    std_output = torch.matmul(activation, weight.t())

    print(output8)
    print(output4)
    print(mixed_output)
    print(std_output)

    torch.testing.assert_close(output4, mixed_output)

if __name__ == '__main__':
    torch.manual_seed(42)

    test_mixed_precision(4096, 4096, 4096, 256)

    print('PASSED')
