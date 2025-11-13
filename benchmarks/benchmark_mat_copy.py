import torch
import argparse

from qfactory.profile import profile_latency

def main(mat, src, dst, kernel):
    total_bytes = (src.shape[0] * mat.shape[1] + dst.shape[0] * mat.shape[1]) * mat.element_size()
    kernel_time = profile_latency(lambda: kernel(mat, src, dst))

    gbs = total_bytes / kernel_time / 1e3
    print(f"{kernel.__name__} Kernel time: {kernel_time:.2f} us, GB/s: {gbs:.2f}")

import triton
import triton.language as tl

@triton.jit
def _index_copy_2d_rows_kernel(
    mat_ptr,          # *T
    src_ptr,          # *int32
    dst_ptr,          # *int32
    stride_m, stride_n,
    num_rows, num_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # 行块 id
    pid_n = tl.program_id(1)  # 列块 id

    # 当前行块对应的索引
    row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)      # [BLOCK_M]
    mask_m = row_idx < num_rows

    src_row = tl.load(src_ptr + row_idx, mask=mask_m)      # [BLOCK_M]
    dst_row = tl.load(dst_ptr + row_idx, mask=mask_m)      # [BLOCK_M]

    # 列方向偏移量
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)       # [BLOCK_N]
    mask_n = offs_n < num_cols

    # 把 BLOCK_M 行一次性复制（用广播+掩码，没有显式 for）
    # 先把 src_row, dst_row 拉成 [BLOCK_M, 1] 方便广播
    src_row = src_row[:, None]  # [BLOCK_M, 1]
    dst_row = dst_row[:, None]  # [BLOCK_M, 1]

    src_ptrs = mat_ptr + src_row * stride_m + offs_n[None, :]  # [BLOCK_M, BLOCK_N]
    dst_ptrs = mat_ptr + dst_row * stride_m + offs_n[None, :]  # [BLOCK_M, BLOCK_N]

    vals = tl.load(src_ptrs, mask=mask_m[:, None] & mask_n[None, :])
    tl.store(dst_ptrs, vals, mask=mask_m[:, None] & mask_n[None, :])

def index_copy_2d_rows(mat: torch.Tensor, src: torch.Tensor, dst: torch.Tensor):
    assert mat.ndim == 2
    assert src.shape == dst.shape
    assert src.dtype == torch.int64
    num_rows = src.numel()
    num_cols = mat.size(1)

    # 简单分块
    BLOCK_M = 4          # 每个 CTA 处理 4 行
    BLOCK_N = 128        # 每行按 128 列分块
    grid = (triton.cdiv(num_rows, BLOCK_M),
            triton.cdiv(num_cols, BLOCK_N))

    _index_copy_2d_rows_kernel[grid](
        mat, src, dst,
        mat.stride(0), mat.stride(1),
        num_rows, num_cols,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return mat

@triton.jit
def _batch_copy_kernel(
    mat_ptr, 
    src_ptr, 
    dst_ptr,
    stride_m, 
    stride_n,
    L: tl.constexpr,  # 复制操作的数量
    N: tl.constexpr,  # 列数
    BLOCK_SIZE: tl.constexpr
):
    # 每个实例处理一个复制操作
    op_idx = tl.program_id(0)
    if op_idx >= L:
        return
    
    # 加载当前操作的源行和目标行索引
    src_row = tl.load(src_ptr + op_idx)
    dst_row = tl.load(dst_ptr + op_idx)
    
    # 计算行起始指针
    src_start = src_row * stride_m
    dst_start = dst_row * stride_m
    
    # 并行复制整行数据
    col_offsets = tl.arange(0, BLOCK_SIZE)
    for col_base in range(0, N, BLOCK_SIZE):
        col_mask = col_offsets < N - col_base
        
        # 读取源数据
        src_ptrs = src_start + (col_base + col_offsets) * stride_n
        data = tl.load(mat_ptr + src_ptrs, mask=col_mask)
        
        # 写入目标位置
        dst_ptrs = dst_start + (col_base + col_offsets) * stride_n
        tl.store(mat_ptr + dst_ptrs, data, mask=col_mask)

def triton_batch_copy(mat: torch.Tensor, src: torch.Tensor, dst: torch.Tensor):
    """
    使用 Triton 高效批量复制矩阵行
    :param mat: 二维矩阵 (M x N)
    :param src: 源行索引 (L,)
    :param dst: 目标行索引 (L,)
    """
    assert mat.is_cuda, "Matrix must be on GPU"
    assert src.is_cuda and dst.is_cuda, "Indices must be on GPU"
    assert src.dim() == 1 and dst.dim() == 1, "Indices must be 1D"
    assert len(src) == len(dst), "src and dst must have same length"
    
    L = len(src)
    M, N = mat.shape
    
    # 自动选择块大小 (32-512 之间)
    BLOCK_SIZE = triton.next_power_of_2(min(N, 512))
    if BLOCK_SIZE < 32:
        BLOCK_SIZE = 32
    
    # 配置并启动核函数
    grid = (triton.cdiv(L, 1),)  # 每个复制操作一个实例
    _batch_copy_kernel[grid](
        mat_ptr=mat,
        src_ptr=src,
        dst_ptr=dst,
        stride_m=mat.stride(0),
        stride_n=mat.stride(1),
        L=L,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )

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
    src = torch.randint(0, args.m, (args.o,), dtype=torch.int64)
    dst = torch.arange(args.m, args.m + args.o, dtype=torch.int64)

    def mat_copy(mat, src, dst):
        mat[dst] = mat[src]
    
    def index_copy(mat, src, dst):
        return mat.index_copy_(0, dst, mat[src])

    main(mat, src, dst, mat_copy)
    main(mat, dst, src, mat_copy)
    main(mat, src, dst, index_copy)
    main(mat, dst, src, index_copy)
    main(mat, src, dst, index_copy_2d_rows)
    main(mat, dst, src, index_copy_2d_rows)
    main(mat, src, dst, triton_batch_copy)
    main(mat, dst, src, triton_batch_copy)
