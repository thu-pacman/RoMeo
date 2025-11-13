import torch
import triton
import argparse

from qfactory.linears import OptMixedQLinear
from qfactory.profile import profile_latency

def profile_latency_graph(f):
    return 1e3 * triton.testing.do_bench_cudagraph(f)

def bench_qfactory(M, N, K, O):
    w = torch.randn(N, K, dtype=torch.float16, device="cuda")
    qlinear = OptMixedQLinear(w, O)
    qlinear_multistream = OptMixedQLinear(w, O, multistream=True)
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    print(f"QFACTORY M {M} N {N} K {K} O {O}", end=" | ")
    latency = qlinear.benchmark(x, profile_latency_graph)
    print(f"Total latency: {latency:.2f} us | TFLOPS: {2 * M * N * K / latency / 1e6:.2f}", end=" | ")
    latency_multistream = profile_latency_graph(lambda: qlinear_multistream(x))
    print(f"Multistream latency: {latency_multistream:.2f} us | TFLOPS: {2 * M * N * K / latency_multistream / 1e6:.2f}")

def bench_quarot(M, N, K):
    import quarot
    quant = quarot.nn.Quantizer()
    linear = torch.nn.Linear(K, N, bias=False, dtype=torch.float16, device="cuda")
    qlinear = quarot.nn.Linear4bit.from_float(linear).to(device="cuda")
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    time_act_quant = profile_latency(lambda: quant(x))
    x = quant(x)
    x, scales_x = x.quantized_x, x.scales_x
    time_gemm = profile_latency(lambda: quarot.matmul(x, qlinear.weight))
    x = quarot.matmul(x, qlinear.weight)
    time_post = profile_latency(lambda: quarot.sym_dequant(x, scales_x, qlinear.weight_scales))
    latency = time_act_quant + time_gemm + time_post
    print(f"QUAROT M {M} N {N} K {K}", end=" | ")
    print(f"Act-quant: {time_act_quant:.2f} us | Gemm: {time_gemm:.2f} us | Post: {time_post:.2f} us", end=" | ")
    print(f"Total latency: {latency:.2f} us | TFLOPS: {2 * M * N * K / latency / 1e6:.2f}")

def main(args: argparse.Namespace):
    if args.batch_size is None:
        batch_sizes = [
            256, 512, 1024, 2048, 4096, 8192, 16384
        ]
    else:
        batch_sizes = [args.batch_size]

    for batch_size in batch_sizes:
        bench_qfactory(batch_size, args.n, args.k, args.o)
        bench_quarot(batch_size, args.n, args.k)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--o", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    main(args)
