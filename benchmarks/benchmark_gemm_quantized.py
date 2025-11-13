import torch
import argparse
import logging
from functools import partial

from qfactory.profile import profile_latency

def benchmark_w8a8(
    m: int,
    n: int,
    k: int,
    gemm,
) -> float:
    x = torch.randint(-128, 128, (m, k), dtype=torch.int8, device='cuda')
    sx = torch.randn(m, dtype=torch.float16, device='cuda')
    w = torch.randint(-128, 128, (n, k), dtype=torch.int8, device='cuda')
    sw = torch.randn(n, dtype=torch.float16, device='cuda')
    out = torch.empty(m, n, dtype=torch.float16, device='cuda')

    def run():
        return gemm(x, sx, w, sw, out)

    return profile_latency(run)

def benchmark(
    m: int,
    n: int,
    k: int,
    gemm,
) -> float:
    x = torch.randint(0, 256, (m, k) if 'omniserve' in gemm.__name__ else (m, k // 2), dtype=torch.uint8, device='cuda')
    sx = torch.randn(m, dtype=torch.float16 if 'quarot' in gemm.__name__ or 'omniserve' in gemm.__name__ else torch.float32, device='cuda')
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device='cuda')
    sw = torch.randn(n, dtype=torch.float16 if 'quarot' in gemm.__name__ or 'omniserve' in gemm.__name__ else torch.float32, device='cuda')
    out = torch.empty(m, n, dtype=torch.float16, device='cuda')
    if 'omniserve' in gemm.__name__:
        x = x.to(torch.int8)
        w = w.to(torch.int8)
    if 'pergroup' in gemm.__name__:
        group_size = int(gemm.__name__.split("_")[-1])
        sx = sx.view(-1, 1).repeat_interleave(k // group_size, dim=1).to(torch.float16)
        sw = sw.view(-1, 1).repeat_interleave(k // group_size, dim=1).to(torch.float16)

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

    outputs = [benchmark_w8a8(bs, args.n, args.k, gemm) if 'int8_int8' in gemm.__name__ else benchmark(bs, args.n, args.k, gemm) for bs in batch_sizes]

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
    parser.add_argument("--group-k", type=str, default="256")
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    gemms = []

    from qfactory import gemm_int8_int8_nt_perchannel
    gemms.append(gemm_int8_int8_nt_perchannel)

    from qfactory import gemm_int4_int4_nt_perchannel, gemm_int4_int4_nt_cutlass
    gemms.append(gemm_int4_int4_nt_perchannel)

    from qfactory import gemm_int4_int4_nt_pergroup
    for k in args.group_k.split(","):
        func = partial(gemm_int4_int4_nt_pergroup, group_k=int(k))
        func.__name__ = f"gemm_int4_int4_nt_pergroup_{k}"
        gemms.append(func)

    def baseline_torch(
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output: torch.Tensor
    ):
        out = torch.empty_like(output, dtype=torch.int32)
        gemm_int4_int4_nt_cutlass(activation, weight, out)
        output.copy_((out * activation_scale.view(-1, 1) * weight_scale.view(1, -1)).to(torch.float16))
        output.copy_(out.to(torch.float16))

    gemms.append(baseline_torch)

    try:
        import quarot

        def baseline_quarot(
            activation: torch.Tensor,
            activation_scale: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
            output: torch.Tensor
        ):
            x = quarot.matmul(activation, weight)
            output.copy_(quarot.sym_dequant(x, activation_scale, weight_scale))

        gemms.append(baseline_quarot)
    except:
        logging.warning("quarot not installed, skipping quarot baseline")
    
    try:
        import omniserve_backend.qgemm_w4a8_per_chn

        def baseline_omniserve(
            activation: torch.Tensor,
            activation_scale: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
            output: torch.Tensor
        ):
            omniserve_backend.qgemm_w4a8_per_chn.gemm_forward_cuda(
                activation,
                weight,
                weight_scale,
                activation_scale,
                weight_scale, # weight_zero
                activation_scale, # x_sum
                output
            )
        
        gemms.append(baseline_omniserve)
    except:
        logging.warning("omniserve_backend not installed, skipping omniserve baseline")

    for gemm in gemms:
        print(f"Benchmarking {gemm.__name__}")
        main(args, gemm)
