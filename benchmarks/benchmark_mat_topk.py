import torch
import argparse

from qfactory.profile import profile_latency

def main(mat, kernel):
    kernel_time = profile_latency(lambda: kernel(mat))
    print(f"{kernel.__name__} Kernel time: {kernel_time:.2f} us")

import triton
import triton.language as tl

@triton.jit
def row_max_kernel(X_ptr, Y_ptr, stride_xn, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    col_idxs = offsets

    X_row = X_ptr + row_idx * stride_xn
    acc = tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        mask = col_idxs < n_cols
        x = tl.load(X_row + col_idxs, mask=mask, other=-float("inf"))
        acc = tl.maximum(acc, x.to(tl.float32))
        col_idxs += BLOCK_SIZE

    max_val = tl.max(acc)
    tl.store(Y_ptr + row_idx, max_val)

def triton_row_max(mat):
    n_rows, n_cols = mat.shape
    output = torch.empty(n_rows, dtype=torch.float32, device=mat.device)
    BLOCK_SIZE = 1024
    grid = lambda meta: (n_rows,)
    row_max_kernel[grid](mat, output, mat.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, default=4096)
    parser.add_argument("-n", type=int, default=4096)
    parser.add_argument("-o", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    mat = torch.randn(args.m + args.o, args.n, dtype=torch.float16)

    def mat_topk(mat):
        row_max = mat.max(dim=1).values
        _, topk_indices = row_max.topk(k=args.o)
        return topk_indices
    
    def triton_topk(mat):
        row_maxes = triton_row_max(mat)
        topk_vals, topk_indices = torch.topk(row_maxes, args.o)
        return topk_indices

    main(mat, mat_topk)
    main(mat, triton_topk)
