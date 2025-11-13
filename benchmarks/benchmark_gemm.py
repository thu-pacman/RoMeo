import torch
import argparse

from qfactory.profile import profile_latency

def benchmark(
    m: int,
    n: int,
    k: int,
    abits: int,
    wbits: int,
    gemm,
) -> float:
    if abits == 16:
        x = torch.randn(m, k, dtype=torch.float16)
    elif abits == 4:
        x = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8)
    else:
        raise ValueError(f"Unsupported abits: {abits}")
    if wbits == 16:
        w = torch.randn(n, k, dtype=torch.float16)
    elif wbits == 4:
        w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    else:
        raise ValueError(f"Unsupported wbits: {wbits}")
    if abits == 4 and wbits == 4:
        out = torch.empty(m, n, dtype=torch.int32)
    else:
        out = torch.empty(m, n, dtype=torch.float16)

    def run():
        return gemm(x, w, out)

    return profile_latency(run)

def main(args: argparse.Namespace, gemm):
    if args.batch_size is None:
        batch_sizes = [
            256, 512, 1024, 2048, 4096, 8192, 16384
        ]
    else:
        batch_sizes = [args.batch_size]

    outputs = [benchmark(bs, args.n, args.k, args.abits, args.wbits, gemm) for bs in batch_sizes]

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
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--wbits", type=int, default=16)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    gemms = []

    if args.abits == 16 and args.wbits == 16:
        from qfactory import gemm_fp16_fp16_nt
        gemms.append(gemm_fp16_fp16_nt)
    elif args.abits == 16 and args.wbits == 4:
        from qfactory import gemm_int4_fp16_nt
        gemms.append(gemm_int4_fp16_nt)
    elif args.abits == 4 and args.wbits == 4:
        from qfactory import gemm_int4_int4_nt, gemm_int4_int4_nt_cutlass
        gemms.append(gemm_int4_int4_nt)
        gemms.append(gemm_int4_int4_nt_cutlass)
    else:
        raise ValueError(f"Unsupported wbits: {args.wbits}")

    for gemm in gemms:
        print(f"Benchmarking {gemm.__name__}")
        main(args, gemm)
