import torch

def safe_intmm(a, b):
    min_dim = 32
    m, n, k = a.shape[0], b.shape[1], a.shape[1]
    if k % 8 != 0:
        pad_k = 8 - (k % 8)
        a = torch.nn.functional.pad(a, (0, pad_k, 0, 0))
        b = torch.nn.functional.pad(b, (0, 0, 0, pad_k))
    if a.shape[0] < min_dim:
        pad_a_rows = min_dim - a.shape[0]
        a = torch.nn.functional.pad(a, (0, 0, 0, pad_a_rows))
    if b.shape[1] < min_dim:
        pad_b_cols = min_dim - b.shape[1]
        b = torch.nn.functional.pad(b, (0, pad_b_cols, 0, 0))
    if b.shape[1] % 8 != 0:
        pad_b_cols = 8 - (b.shape[1] % 8)
        b = torch.nn.functional.pad(b, (0, pad_b_cols, 0, 0))
    return torch._int_mm(a, b)[:m, :n]

def matmul_w8a8(a, scale_a, b, scale_b, group_size_a=None, group_size_b=None):
    group_size_a = group_size_a if group_size_a is not None else a.shape[1]
    group_size_b = group_size_b if group_size_b is not None else b.shape[0]
    assert len(a.shape) == 2 and len(b.shape) == 2
    assert a.shape[1] == b.shape[0]
    assert a.shape[1] % group_size_a == 0
    assert b.shape[0] % group_size_b == 0
    group_size = min(group_size_a, group_size_b)
    assert group_size_a % group_size == 0 and group_size_b % group_size == 0
    num_groups = a.shape[1] // group_size
    assert scale_a.shape == (a.shape[0], a.shape[1] // group_size_a)
    assert scale_b.shape == (b.shape[0] // group_size_b, b.shape[1])
    # assert a.max() <= 127 and a.min() >= -128
    # assert b.max() <= 127 and b.min() >= -128
    out = torch.zeros(a.shape[0], b.shape[1], device=a.device, dtype=scale_a.dtype)
    for i in range(num_groups):
        group_id_a = i * group_size // group_size_a
        group_id_b = i * group_size // group_size_b
        start = i * group_size
        end = (i + 1) * group_size
        sa = scale_a[:, group_id_a:group_id_a + 1]
        sb = scale_b[group_id_b:group_id_b + 1, :]
        out += safe_intmm(a[:, start:end].to(torch.int8), b[start:end, :].to(torch.int8)).to(scale_a.dtype) * sa * sb
    return out

def matmul_w4a4(a, scale_a, b, scale_b, group_size_a, group_size_b):
    assert len(a.shape) == 2 and len(b.shape) == 2
    assert a.shape[1] == b.shape[0]
    assert a.shape[1] % group_size_a == 0
    assert b.shape[0] % group_size_b == 0
    group_size = min(group_size_a, group_size_b)
    assert group_size_a % group_size == 0 and group_size_b % group_size == 0
    num_groups = a.shape[1] // group_size
    assert scale_a.shape == (a.shape[0], a.shape[1] // group_size_a)
    assert scale_b.shape == (b.shape[0] // group_size_b, b.shape[1])
    # assert a.max() <= 7 and a.min() >= -8
    # assert b.max() <= 7 and b.min() >= -8
    out = torch.zeros(a.shape[0], b.shape[1], device=a.device, dtype=scale_a.dtype)
    for i in range(num_groups):
        group_id_a = i * group_size // group_size_a
        group_id_b = i * group_size // group_size_b
        start = i * group_size
        end = (i + 1) * group_size
        sa = scale_a[:, group_id_a:group_id_a + 1]
        sb = scale_b[group_id_b:group_id_b + 1, :]
        out += safe_intmm(a[:, start:end].to(torch.int8), b[start:end, :].to(torch.int8)).to(scale_a.dtype) * sa * sb
    return out

def matmul_w4a16(a, b, scale_b, group_size_b):
    assert len(a.shape) == 2 and len(b.shape) == 2
    assert a.shape[1] == b.shape[0]
    assert b.shape[0] % group_size_b == 0
    num_groups = a.shape[1] // group_size_b
    assert scale_b.shape == (b.shape[0] // group_size_b, b.shape[1])
    assert b.max() <= 7 and b.min() >= -8
    out = torch.zeros(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
    for i in range(num_groups):
        start = i * group_size_b
        end = (i + 1) * group_size_b
        sb = scale_b[i:i + 1, :]
        out += (a[:, start:end] @ b[start:end, :].to(a.dtype)) * sb
    return out
