import torch
import triton
import triton.language as tl

@triton.jit
def row_abs_max_kernel(X_ptr, Y_ptr, stride_xn, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    col_idxs = offsets

    X_row = X_ptr + row_idx * stride_xn
    acc = tl.full((BLOCK_SIZE,), 0, dtype=tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        mask = col_idxs < n_cols
        x = tl.load(X_row + col_idxs, mask=mask, other=0.0)
        abs_x = tl.abs(x)
        acc = tl.maximum(acc, abs_x)
        col_idxs += BLOCK_SIZE

    max_val = tl.max(acc)
    tl.store(Y_ptr + row_idx, max_val)

def triton_row_max(mat):
    n_rows, n_cols = mat.shape
    output = torch.empty(n_rows, dtype=torch.bfloat16, device=mat.device)
    BLOCK_SIZE = 1024
    grid = lambda meta: (n_rows,)
    row_abs_max_kernel[grid](mat, output, mat.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output

def mat_topk(
    mat: torch.Tensor,
    k: int,
):
    row_maxes = triton_row_max(mat)
    topk_vals, topk_indices = torch.topk(row_maxes, k)
    return row_maxes, topk_indices

@triton.jit
def quantize_pack_kernel(
    inp_ptr,              # [M, N] bfloat16
    token_max_ptr,        # [M] bfloat16
    output_ptr,           # [M, N//2] uint8
    M: tl.constexpr, 
    N: tl.constexpr,
    stride_m: tl.constexpr,
    out_stride_m: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row index
    pid_n = tl.program_id(1)  # block column index (each block handles 2 columns)

    out_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_n0 = out_offsets * 2
    offs_n1 = out_offsets * 2 + 1

    mask_0 = offs_n0 < N
    mask_1 = offs_n1 < N

    # Load inp and token_max
    inp0 = tl.load(inp_ptr + pid_m * stride_m + offs_n0, mask=mask_0)
    inp1 = tl.load(inp_ptr + pid_m * stride_m + offs_n1, mask=mask_1)
    token_max = tl.load(token_max_ptr + pid_m)

    scale = token_max / 7.0

    q0 = tl.math.floor((inp0 / scale) + 0.5).to(tl.int32)
    q1 = tl.math.floor((inp1 / scale) + 0.5).to(tl.int32)

    # map to [0,15] for int4
    q0 = q0 + 16 * (q0 < 0)
    q1 = q1 + 16 * (q1 < 0)

    # pack into uint8
    packed = q0 | (q1 << 4)

    tl.store(output_ptr + pid_m * out_stride_m + out_offsets, packed.to(tl.uint8), mask=mask_0)

def quantize_pack(inp: torch.Tensor, token_max: torch.Tensor):
    assert inp.dtype == torch.bfloat16 and token_max.dtype == torch.bfloat16
    M, N = inp.shape
    assert N % 2 == 0

    output = torch.empty((M, N // 2), dtype=torch.uint8, device=inp.device)

    grid = lambda meta: (M, triton.cdiv(N // 2 , meta['BLOCK_SIZE_N']))
    quantize_pack_kernel[grid](
        inp, token_max, output,
        M=M, N=N,
        stride_m=inp.stride(0),
        out_stride_m=output.stride(0),
        BLOCK_SIZE_N=512
    )
    return output
