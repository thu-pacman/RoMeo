import math
import json
import torch
import argparse

from triton.testing import do_bench_cudagraph

STREAMS = [torch.cuda.Stream() for _ in range(4)]

def profile_latency(f):
    return do_bench_cudagraph(f) * 1000

def benchmark_kernel(
    m: int,
    n: int,
    k: int,
    o_x: int,
    o_w: int,
    gemm,
) -> float:
    scale_dtype = torch.float16 if 'quarot' in gemm.__name__ else torch.bfloat16

    if 'int8' in gemm.__name__:
        x = torch.randint(-128, 127, (m, k), dtype=torch.int8, device='cuda')
        sx = torch.randn(m, dtype=torch.float16, device='cuda')
        w = torch.randint(-128, 127, (n, k), dtype=torch.int8, device='cuda')
        sw = torch.randn(n, dtype=torch.float16, device='cuda')
    else:
        x = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device='cuda')
        sx = torch.randn(m, dtype=scale_dtype, device='cuda')
        w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device='cuda')
        sw = torch.randn(n, dtype=scale_dtype, device='cuda')

    if 'half' in gemm.__name__:
        x = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
        w = torch.randn(n, k, dtype=torch.bfloat16, device='cuda')

    if 'mixed_precision' in gemm.__name__:
        x_o = torch.randint(0, 256, (o_x, k), dtype=torch.uint8, device='cuda')
        sx_o = torch.randn(o_x, dtype=scale_dtype, device='cuda')
        w_o = torch.randint(0, 256, (o_w, k), dtype=torch.uint8, device='cuda')
        sw_o = torch.randn(o_w, dtype=scale_dtype, device='cuda')
        out = torch.empty(m + o_x, n + o_w, dtype=torch.bfloat16, device='cuda')
        
        return profile_latency(lambda : gemm(
            x, sx, x_o, sx_o,
            w, sw, w_o, sw_o,
            out, o_x, o_w,
            [torch.cuda.current_stream() for _ in range(4)]
        ))
    
    out = torch.empty(m, n, dtype=scale_dtype, device='cuda')

    return profile_latency(lambda : gemm(x, sx, w, sw, out))

def bench(m, n, k, percentage, kernel):
    def get_outlier(x):
        return math.ceil(x * percentage / 256) * 256

    kernel_time = benchmark_kernel(m, n, k, get_outlier(m), get_outlier(n), kernel)

    flops = 2 * n * k * m
    tflops = flops / kernel_time / 1e6
    print(f"M: {m} N: {n} K: {k} | Kernel time: {kernel_time:.2f} us TFLOPS: {tflops:.2f}")
    return {
        'M': m,
        'N': n,
        'K': k,
        'kernel': kernel.__name__,
        'us': kernel_time,
    }

def init_kernels():
    kernels = []

    def baseline_half(
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output: torch.Tensor
    ):
        return activation @ weight.T
    
    kernels.append(baseline_half)
    
    from qfactory import gemm_int8_int8_nt_perchannel
    
    def baseline_int8_perchannel(
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output: torch.Tensor
    ):
        out = torch.empty_like(output, dtype=torch.float16)
        gemm_int8_int8_nt_perchannel(activation, activation_scale, weight, weight_scale, out)
        return out
    
    kernels.append(baseline_int8_perchannel)

    from qfactory import gemm_int4_int4_nt_cutlass, gemm_int4_int4_nt_mixed_precision_separate, gemm_int4_int4_nt_mixed_precision

    def baseline_torch(
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output: torch.Tensor
    ):
        out = torch.empty_like(output, dtype=torch.int32)
        gemm_int4_int4_nt_cutlass(activation, weight, out)
        return out.to(torch.bfloat16) * activation_scale.view(-1, 1) * weight_scale.view(1, -1)

    kernels.append(baseline_torch)

    import quarot

    def baseline_quarot(
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output: torch.Tensor
    ):
        x = quarot.matmul(activation, weight)
        return quarot.sym_dequant(x, activation_scale, weight_scale)

    kernels.append(baseline_quarot)

    kernels.append(gemm_int4_int4_nt_mixed_precision_separate)
    
    def baseline_mixed_precision_multistream(*args):
        global STREAMS
        e0 = torch.cuda.current_stream().record_event()
        for s in STREAMS:
            s.wait_event(e0)
        events = gemm_int4_int4_nt_mixed_precision_separate(*args[:-1], STREAMS)
        for e in events:
            torch.cuda.current_stream().wait_event(e)
    
    kernels.append(baseline_mixed_precision_multistream)

    def baseline_mixed_precision_unifiedkernel(*args):
        return gemm_int4_int4_nt_mixed_precision(*args[:-1])
    
    kernels.append(baseline_mixed_precision_unifiedkernel)

    return kernels

ALL_K_N = [
    # Qwen3-8B
    (4096, 6144), # qkv
    (4096, 4096), # o
    (4096, 24576), # ug
    (12288, 4096), # d
    # Qwen3-14B
    (5120, 7168), # qkv
    (5120, 5120), # o
    (5120, 34816), # ug
    (17408, 5120), # d
    # Qwen3-32B
    (5120, 10240), # qkv
    (8192, 5120), # o
    (5120, 51200), # ug
    (25600, 5120), # d
    # Llama-3.1-70B
    (8192, 10240), # qkv
    (8192, 8192), # o
    (8192, 57344), # ug
    (28672, 8192), # d
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--percentage", type=float, default=0.05)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    kernels = init_kernels()

    RESULTS = []

    for k, n in ALL_K_N:
        print(f"Benchmarking for M: {args.m}, N: {n}, K: {k}")
        for kernel in kernels:
            print(f"Benchmarking {kernel.__name__}")
            result = bench(args.m, n, k, args.percentage, kernel)
            RESULTS.append(result)

    json.dump(RESULTS, open(f'bench_kernels_results.json', 'w'), indent=4)

if __name__ == "__main__":
    main()
