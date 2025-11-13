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
def _col_index_copy_kernel(
    mat_ptr,          # T*
    src_ptr,          # int32*
    dst_ptr,          # int32*
    stride_m, stride_n,
    num_rows, num_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)   # 行块
    pid_n = tl.program_id(1)   # 列对

    # 当前列对
    col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    mask_n = col_idx < num_cols
    src_col = tl.load(src_ptr + col_idx, mask=mask_n)   # [BLOCK_N]
    dst_col = tl.load(dst_ptr + col_idx, mask=mask_n)   # [BLOCK_N]

    # 行方向
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)    # [BLOCK_M]
    mask_m = offs_m < num_rows                          # [BLOCK_M]

    # 广播到 [BLOCK_M, BLOCK_N]
    src_col = src_col[None, :]      # [1, BLOCK_N]
    dst_col = dst_col[None, :]      # [1, BLOCK_N]
    offs_m  = offs_m[:, None]       # [BLOCK_M, 1]

    # 计算二维指针
    src_ptrs = mat_ptr + offs_m * stride_m + src_col
    dst_ptrs = mat_ptr + offs_m * stride_m + dst_col

    # 二维掩码
    mask = mask_m[:, None] & mask_n[None, :]   # [BLOCK_M, BLOCK_N]

    # load/store
    vals = tl.load(src_ptrs, mask=mask)
    tl.store(dst_ptrs, vals, mask=mask)

def col_index_copy(mat: torch.Tensor, src: torch.Tensor, dst: torch.Tensor):
    assert mat.ndim == 2 and src.shape == dst.shape and src.dtype == torch.int64
    K = src.numel()
    M, _ = mat.shape

    BLOCK_M = 64
    BLOCK_N = 1
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))

    _col_index_copy_kernel[grid](
        mat, src, dst,
        mat.stride(0), mat.stride(1),
        M, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return mat

# 自动调优配置
AUTO_TUNE_CONFIGS = [
    triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
]

@triton.autotune(
    configs=AUTO_TUNE_CONFIGS,
    key=['M', 'L']
)
@triton.jit
def _column_copy_kernel(
    mat_ptr,
    src_ptr,
    dst_ptr,
    stride_row,
    stride_col,
    M,  # 行数
    L,  # 复制操作数量
    BLOCK_SIZE: tl.constexpr  # 自动调优参数
):
    # 每个实例处理一个复制操作
    op_idx = tl.program_id(0)
    if op_idx >= L:
        return
    
    # 加载当前操作的源列和目标列索引
    src_col = tl.load(src_ptr + op_idx)
    dst_col = tl.load(dst_ptr + op_idx)
    
    # 计算列起始偏移量
    src_start = src_col * stride_col
    dst_start = dst_col * stride_col
    
    # 并行处理列数据
    row_offsets = tl.arange(0, BLOCK_SIZE)
    for row_base in range(0, M, BLOCK_SIZE):
        row_mask = row_offsets < M - row_base
        
        # 读取源列数据
        src_ptrs = mat_ptr + (row_base + row_offsets) * stride_row + src_start
        data = tl.load(src_ptrs, mask=row_mask)
        
        # 写入目标列
        dst_ptrs = mat_ptr + (row_base + row_offsets) * stride_row + dst_start
        tl.store(dst_ptrs, data, mask=row_mask)

def auto_tuned_column_copy(mat: torch.Tensor, src: torch.Tensor, dst: torch.Tensor):
    """
    修复后的自动调优列复制实现
    :param mat: 二维矩阵 (M x N)
    :param src: 源列索引 (L,)
    :param dst: 目标列索引 (L,)
    """
    assert mat.is_cuda, "Matrix must be on GPU"
    assert src.is_cuda and dst.is_cuda, "Indices must be on GPU"
    assert mat.dim() == 2, "Matrix must be 2D"
    assert src.dim() == 1 and dst.dim() == 1, "Indices must be 1D"
    assert len(src) == len(dst), "src and dst must have same length"
    
    M, N = mat.shape
    L = len(src)
    
    # 配置并启动核函数
    grid = (triton.cdiv(L, 1),)
    
    _column_copy_kernel[grid](
        mat,
        src,
        dst,
        mat.stride(0),
        mat.stride(1),
        M,
        L
    )

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "num_warps": 2}),
        triton.Config({"BLOCK_M": 128, "num_warps": 4}),
        triton.Config({"BLOCK_M": 256, "num_warps": 8}),
    ],
    key=["M"],
)
@triton.jit
def column_index_copy_kernel(
    mat_ptr, src_ptr, dst_ptr,
    M, K,
    stride_m, stride_n,
    BLOCK_M: tl.constexpr,
):

    row_idx = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = row_idx < M

    # 循环每对 src/dst 列索引
    for i in range(K):
        src_col = tl.load(src_ptr + i)
        dst_col = tl.load(dst_ptr + i)

        # 按列索引访问元素
        src_offset = row_idx * stride_m + src_col * stride_n
        dst_offset = row_idx * stride_m + dst_col * stride_n

        val = tl.load(mat_ptr + src_offset, mask=mask)
        tl.store(mat_ptr + dst_offset, val, mask=mask)

def triton_index_copy(mat: torch.Tensor, src: torch.Tensor, dst: torch.Tensor):
    M, N = mat.shape
    K = src.shape[0]

    assert mat.is_cuda and src.is_cuda and dst.is_cuda
    mat = mat.contiguous()
    
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    column_index_copy_kernel[grid](
        mat, src, dst,
        M, K,
        mat.stride(0), mat.stride(1),
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

    mat = torch.randn(args.m + args.o, args.n + args.o, dtype=torch.float16)
    src = torch.randint(0, args.n, (args.o,), dtype=torch.int64)
    dst = torch.arange(args.n, args.n + args.o, dtype=torch.int64)

    def mat_copy(mat, src, dst):
        mat[:, dst] = mat[:, src]
    
    def index_copy(mat, src, dst):
        return mat.index_copy_(1, dst, mat[:, src])

    main(mat, src, dst, mat_copy)
    main(mat, dst, src, mat_copy)
    main(mat, src, dst, index_copy)
    main(mat, dst, src, index_copy)
    main(mat, src, dst, col_index_copy)
    main(mat, dst, src, col_index_copy)
    main(mat, src, dst, auto_tuned_column_copy)
    main(mat, dst, src, auto_tuned_column_copy)
    main(mat, src, dst, triton_index_copy)
    main(mat, dst, src, triton_index_copy)
