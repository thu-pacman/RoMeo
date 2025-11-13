import torch
from typing import Tuple, Optional, Dict, List

from .utils import span_tuning_space
from ..jit import jit

includes = (
    'cuda/fused_moe/fused_moe_w8a8_sm90.h',
)

template = """
constexpr int N = {N};
constexpr int K = {K};
constexpr int E = {E};
constexpr int topk = {topk};
constexpr int has_moe_weight = {has_moe_weight};
constexpr int TileM = {TileM};
constexpr int TileN = {TileN};
constexpr int TileK = {TileK};
constexpr int NStage = {NStage};

call_gemm <
    N, K, E, topk, has_moe_weight,
    TileM, TileN, TileK, NStage
> (
    m, num_chunks_tensor, max_num_chunks,
    activation, weight, output,
    activation_scale, weight_scale,
    token_ids, expert_ids, moe_weight,
    stream
);
"""

def generate_tunable_keys(keys) -> Dict[str, List[int]]:
    return {
        'NStage': [1, 2, 3, 4, 5],
        'TileM': [keys['chunk_size']],
        'TileN': [keys['block_n']],
        'TileK': [keys['block_k']],
    }

def fused_moe_fp8_fp8_bf16_nt(
    num_chunks_tensor: torch.Tensor,
    activations: Tuple[torch.Tensor, torch.Tensor], # activation & scale
    weights: Tuple[torch.Tensor, torch.Tensor], # weight & scale
    output: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    chunk_size: int,
    topk: int,
    block_n: int,
    block_k: int,
    moe_weight : Optional[torch.Tensor] = None
):
    activation, activation_scale = activations
    weight, weight_scale = weights

    m, k = activation.shape
    e, n, k_ = weight.shape
    # num_chunks = expert_ids.shape[0]
    max_num_chunks = expert_ids.shape[0]
    
    assert k == k_
    assert k % block_k == 0
    assert n % block_n == 0

    # tensor size check
    assert activation_scale.shape == (m, k // block_k)
    assert weight_scale.shape == (e, n // block_n, k // block_k)
    assert output.shape == (m * topk, n)
    # assert token_ids.shape == (num_chunks * chunk_size,)
    # assert expert_ids.shape == (num_chunks,)
    if moe_weight is not None:
        assert moe_weight.shape == (m, topk)
    
    # tensor dtype check
    assert activation.dtype == torch.float8_e4m3fn and activation_scale.dtype == torch.float32
    assert weight.dtype == torch.float8_e4m3fn and weight_scale.dtype == torch.float32
    assert output.dtype == torch.bfloat16
    assert token_ids.dtype == torch.int32
    assert expert_ids.dtype == torch.int32
    if moe_weight is not None:
        assert moe_weight.dtype == torch.float32
    
    # contiguous check
    assert activation.is_contiguous() and activation_scale.is_contiguous()
    assert weight.is_contiguous() and weight_scale.is_contiguous()
    assert output.is_contiguous()
    assert token_ids.is_contiguous() and expert_ids.is_contiguous()
    if moe_weight is not None:
        assert moe_weight.is_contiguous()

    # compile-time keys for specifying the kernel
    keys = {
        'E': e,
        'N': n,
        'K': k,
        'chunk_size': chunk_size,
        'topk': topk,
        'block_n': block_n,
        'block_k': block_k,
        'has_moe_weight': int(moe_weight is not None)
    }
    tunable_keys = generate_tunable_keys(keys)

    config_space = span_tuning_space(tunable_keys)

    args = (
        (m, 'm'), (num_chunks_tensor, 'num_chunks_tensor'), (max_num_chunks, 'max_num_chunks'),
        (activation, 'activation'), (weight, 'weight'), (output, 'output'),
        (activation_scale, 'activation_scale'), (weight_scale, 'weight_scale'),
        (token_ids, 'token_ids'), (expert_ids, 'expert_ids'), (moe_weight, 'moe_weight'),
        (torch.cuda.current_stream(), 'stream')
    )

    global includes, template

    runtime = jit.compile_and_tune(
        name='fused_moe_fp8_fp8_bf16_nt',
        includes=includes,
        template=template,
        keys=keys,
        space=config_space,
        args=args,
    )

    return runtime.run(*(arg for arg, _ in args))

def fused_moe_fp8_fp8_bf16_nt_naive(
    num_chunks_tensor: torch.Tensor,
    activations: Tuple[torch.Tensor, torch.Tensor], # activation & scale
    weights: Tuple[torch.Tensor, torch.Tensor], # weight & scale
    output: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    chunk_size: int,
    topk: int,
    block_n: int,
    block_k: int,
    moe_weight : Optional[torch.Tensor] = None
):
    num_chunks = num_chunks_tensor.item()
    activation = activations[0].to(torch.float)
    activation_scale = activations[1]
    weight = weights[0].to(torch.float)
    weight_scale = weights[1]
    for c in range(num_chunks):
        act_ids = token_ids[c * chunk_size : (c + 1) * chunk_size]
        act_ids = act_ids[act_ids < topk * activation.shape[0]]
        if act_ids.numel() == 0:
            continue
        eid = expert_ids[c]
        
        a = activation[act_ids // topk] * activation_scale[act_ids // topk].repeat_interleave(block_k).reshape(act_ids.shape[0], -1)
        b = weight[eid] * weight_scale[eid].repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)

        c = a @ b.T
        if moe_weight is not None:
            output[act_ids] = (c * moe_weight[act_ids // topk, act_ids % topk].reshape(-1, 1)).to(torch.bfloat16)
        else:
            output[act_ids] = c.to(torch.bfloat16)