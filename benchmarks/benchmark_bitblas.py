import torch
import bitblas
import argparse

from qfactory.profile import profile_latency

def benchmark(
    m: int,
    n: int,
    k: int,
) -> float:
    matmul_config = bitblas.MatmulConfig(
        M=m,  # M dimension
        N=n,  # N dimension
        K=k,  # K dimension
        A_dtype="int4",  # activation A dtype
        W_dtype="int4",  # weight W dtype
        accum_dtype="int32",  # accumulation dtype
        out_dtype="float16",  # output dtype
        layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
        with_bias=False,  # bias
        # configs for weight only quantization
        group_size=None,  # setting for grouped quantization
        with_scaling=False,  # setting for scaling factor
        with_zeros=False,  # setting for zeros
        zeros_mode=None,  # setting for how to calculating zeros
    )
    matmul = bitblas.Matmul(config=matmul_config)

    parameters = tuple(map(lambda x : torch.from_numpy(x.numpy()).to('cuda'), matmul.get_profile_tensors()))

    def run():
        return matmul(parameters[0], parameters[1], output=parameters[2])

    return profile_latency(run)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    if args.batch_size is None:
        batch_sizes = [
            256, 512, 1024, 2048, 4096, 8192, 16384
        ]
    else:
        batch_sizes = [args.batch_size]

    outputs = [benchmark(bs, args.n, args.k) for bs in batch_sizes]

    for batch_size, kernel_time in zip(batch_sizes, outputs):
        flops = 2 * args.n * args.k * batch_size
        tflops = flops / kernel_time / 1e6
        print(f"Batch size: {batch_size} Kernel time: {kernel_time:.2f} us, TFLOPS: {tflops:.2f}")
