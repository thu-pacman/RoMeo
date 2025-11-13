import torch
import argparse
import logging
from functools import partial

from qfactory.profile import profile_latency

def benchmark(
    m: int,
    n: int,
    k: int,
    gemm,
) -> float:
    x = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8)
    sx = torch.randn(m, dtype=torch.float32)
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    sw = torch.randn(n, dtype=torch.float32)
    out = torch.empty(m, n, dtype=torch.float16)
    if 'pergroup' in gemm.__name__ or 'elastic' in gemm.__name__:
        if 'pergroup' in gemm.__name__:
            num_groups = k // int(gemm.__name__.split("_")[-1])
        else:
            num_groups = int(gemm.__name__.split("_")[-2]) + 1
        sx = sx.view(-1, 1).repeat_interleave(num_groups, dim=1).to(torch.float16)
        sw = sw.view(-1, 1).repeat_interleave(num_groups, dim=1).to(torch.float16)

    def run():
        return gemm(x, sx, w, sw, out)

    return profile_latency(run)

def main(args: argparse.Namespace, gemm):
    if args.batch_size is None:
        batch_sizes = [
            256, 512, 1024, 2048, 4096, 8192, 16384
        ]
    else:
        batch_sizes = [args.batch_size]

    outputs = [benchmark(bs, args.n, args.k, gemm) for bs in batch_sizes]

    for batch_size, kernel_time in zip(batch_sizes, outputs):
        flops = 2 * args.n * args.k * batch_size
        tflops = flops / kernel_time / 1e6
        print(f"Batch size: {batch_size} Kernel time: {kernel_time:.2f} us, TFLOPS: {tflops:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--outlier", type=int, default=256)
    parser.add_argument("--divs", type=int, default=2)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    gemms = []

    from qfactory import gemm_int4_int4_nt_cutlass, gemm_int4_int4_nt_perchannel, gemm_int4_int4_nt_pergroup, gemm_int4_int4_nt_elastic

    def baseline_cutlass(
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output: torch.Tensor
    ):
        out = torch.empty_like(output, dtype=torch.int32)
        gemm_int4_int4_nt_cutlass(activation, weight, out)

    gemms.append(baseline_cutlass)
    gemms.append(gemm_int4_int4_nt_perchannel)

    func1 = partial(gemm_int4_int4_nt_pergroup, group_k=args.outlier)
    func1.__name__ = f"gemm_int4_int4_nt_pergroup_{args.outlier}"
    gemms.append(func1)

    func2 = partial(gemm_int4_int4_nt_elastic, outlier=args.outlier, div=args.divs)
    func2.__name__ = f"gemm_int4_int4_nt_elastic_{args.divs}_{args.outlier}"
    gemms.append(func2)

    for gemm in gemms:
        print(f"Benchmarking {gemm.__name__}")
        main(args, gemm)
