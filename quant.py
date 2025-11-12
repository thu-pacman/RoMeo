import math
import torch

@torch.no_grad()
def pertoken_threshold(inp, threshold, policy):
    assert len(inp.shape) == 2
    pertoken_max = torch.max(torch.abs(inp), dim=1, keepdim=True).values
    if policy == 'value':
        high_idx = (pertoken_max >= threshold).squeeze(1).nonzero().squeeze(1)
    elif policy == 'amount':
        number_outliers = int(threshold)
        _, high_idx = torch.topk(pertoken_max.squeeze(1), min(number_outliers, pertoken_max.shape[0]), largest=True, sorted=False)
    elif policy == 'percentage':
        number_outliers = math.ceil(pertoken_max.shape[0] * threshold / 256) * 256
        _, high_idx = torch.topk(pertoken_max.squeeze(1), min(number_outliers, pertoken_max.shape[0]), largest=True, sorted=False)
    elif policy == 'percentage-noround':
        number_outliers = round(pertoken_max.shape[0] * threshold)
        _, high_idx = torch.topk(pertoken_max.squeeze(1), min(number_outliers, pertoken_max.shape[0]), largest=True, sorted=False)
    else:
        raise NotImplementedError(f"Unknown threshold policy: {policy}")
    return high_idx

@torch.no_grad()
def perchannel_threshold(inp, threshold, policy):
    assert len(inp.shape) == 2
    perchannel_max = torch.max(torch.abs(inp), dim=0, keepdim=True).values
    if policy == 'value':
        high_idx = (perchannel_max >= threshold).squeeze(0).nonzero().squeeze(1)
    elif policy == 'amount':
        number_outliers = int(threshold)
        _, high_idx = torch.topk(perchannel_max.squeeze(0), min(number_outliers, perchannel_max.shape[1]), largest=True, sorted=False)
    elif policy == 'percentage':
        number_outliers = math.ceil(perchannel_max.shape[1] * threshold / 256) * 256
        _, high_idx = torch.topk(perchannel_max.squeeze(0), min(number_outliers, perchannel_max.shape[1]), largest=True, sorted=False)
    elif policy == 'percentage-noround':
        number_outliers = round(perchannel_max.shape[1] * threshold)
        _, high_idx = torch.topk(perchannel_max.squeeze(0), min(number_outliers, perchannel_max.shape[1]), largest=True, sorted=False)
    else:
        raise NotImplementedError(f"Unknown threshold policy: {policy}")
    return high_idx

@torch.no_grad()
def get_complementary(total_size, indices):
    mask = torch.ones(total_size, dtype=torch.bool, device=indices.device)
    mask[indices] = False
    idx = torch.argsort(mask.to(torch.int8), stable=True, descending=True)
    return idx[:total_size - indices.shape[0]]

@torch.no_grad()
def pertoken_quantize(inp, bits, group_size):
    assert len(inp.shape) == 2
    assert inp.shape[1] % group_size == 0
    int_range = 2 ** (bits - 1) - 1
    num_groups = inp.shape[1] // group_size
    pergroup_max = torch.empty((inp.shape[0], num_groups), device=inp.device, dtype=inp.dtype)

    for i in range(num_groups):
        start = i * group_size
        end = (i + 1) * group_size
        pergroup_max[:, i] = torch.max(torch.abs(inp[:, start:end]), dim=1).values

    pergroup_scale = pergroup_max / int_range
    quantized_inp = torch.clip(torch.round(inp / pergroup_scale.repeat_interleave(group_size, dim=1)).to(torch.int32), min=-int_range - 1, max=int_range)
    return quantized_inp.to(torch.int8), pergroup_scale

@torch.no_grad()
def perchannel_quantize(inp, bits):
    assert len(inp.shape) == 2
    int_range = 2 ** (bits - 1) - 1
    perchannel_max = torch.max(torch.abs(inp), dim=0, keepdim=True).values # (1, channel)
    perchannel_scale = perchannel_max / int_range
    quantized_inp = torch.round(inp / perchannel_scale).to(torch.int32)
    return quantized_inp.to(torch.int8), perchannel_scale

@torch.no_grad()
def pertoken_quantize_search(inp, bits, group_size, threshold=None, max_shrink=0.8, grid_size=100, norm=2.4):
    assert len(inp.shape) == 2
    assert inp.shape[1] % group_size == 0
    int_range = 2 ** (bits - 1) - 1
    num_groups = inp.shape[1] // group_size

    best_err = torch.full((inp.shape[0], ), float('inf'), dtype=inp.dtype, device=inp.device)
    best_scale = torch.empty((inp.shape[0], num_groups), dtype=inp.dtype, device=inp.device)

    pertoken_max = torch.max(torch.abs(inp), dim=1, keepdim=True).values # (token, 1)
    if threshold is not None:
        low_idx = pertoken_max < threshold

    pergroup_max = torch.empty((inp.shape[0], num_groups), device=inp.device, dtype=inp.dtype)
    for i in range(num_groups):
        start = i * group_size
        end = (i + 1) * group_size
        pergroup_max[:, i] = torch.max(torch.abs(inp[:, start:end]), dim=1).values

    for i in range(int(max_shrink * grid_size)):
        clip_percentile = 1 - i / grid_size
        clipped_pergroup_max = pergroup_max * clip_percentile
        clipped_pergroup_scale = clipped_pergroup_max / int_range
        quantized_inp = torch.round(inp / clipped_pergroup_scale.repeat_interleave(group_size, dim=1)).to(torch.int32).clamp(-int_range, int_range)
        dequantized_inp = quantized_inp.to(inp.dtype) * clipped_pergroup_scale.repeat_interleave(group_size, dim=1)
        per_token_err = torch.mean((inp - dequantized_inp).abs().pow(norm), dim=1)
        mask = per_token_err < best_err
        if mask.any():
            best_err[mask] = per_token_err[mask]
            best_scale[mask, :] = clipped_pergroup_scale[mask, :]
    
    if threshold is not None:
        return torch.round(inp / best_scale.repeat_interleave(group_size, dim=1)).to(torch.int32).clamp(-int_range, int_range).to(torch.int8), best_scale, low_idx
    else:
        return torch.round(inp / best_scale.repeat_interleave(group_size, dim=1)).to(torch.int32).clamp(-int_range, int_range).to(torch.int8), best_scale
