import torch
from typing import Dict, List

from .utils import span_tuning_space
from ..jit import jit

def generate_tunable_keys() -> Dict[str, List[int]]:
    return {
        'NStage': [1, 2, 3],
        'TileM': [128, 256],
        'TileN': [128, 256],
        'TileK': [128, 256],
        'WarpM': [64],
        'WarpN': [64],
        'WarpK': [128],
    }

def generate_tunable_keys_pergroup() -> Dict[str, List[int]]:
    return {
        'NStage': [1, 2, 3],
        'TileM': [128, 256],
        'TileN': [128, 256],
        'TileK': [256], # TileK 128 causes issue in per_group kernel
        'WarpM': [64],
        'WarpN': [64],
        'WarpK': [128],
    }

def gemm_int4_int4_nt_perchannel(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor
):
    m, k = activation.shape
    n, k_ = weight.shape
    
    assert k * 2 == k_ * 2

    # tensor size check
    assert activation_scale.shape == (m,)
    assert weight_scale.shape == (n,)
    assert output.shape == (m, n)
    
    # tensor dtype check
    assert activation.dtype == torch.uint8
    assert activation_scale.dtype == torch.float32
    assert weight.dtype == torch.uint8
    assert weight_scale.dtype == torch.float32
    assert output.dtype == torch.float16
    
    # contiguous check
    assert activation.is_contiguous()
    assert activation_scale.is_contiguous()
    assert weight.is_contiguous()
    assert weight_scale.is_contiguous()
    assert output.is_contiguous()

    # key for kernel selection
    perf_keys = {
        'M': m,
    }
    # compile-time keys for specifying the kernel
    keys = {
        'N': n,
        'K': k * 2,
    }
    tunable_keys = generate_tunable_keys()

    config_space = span_tuning_space(tunable_keys)

    args = (
        (m, 'm'),
        (activation, 'activation'), (weight, 'weight'), (output, 'output'),
        (activation_scale, 'activation_scale'), (weight_scale, 'weight_scale'),
        (torch.cuda.current_stream(), 'stream')
    )

    includes = (
        'cuda/gemm/gemm_w4a4_perchannel_sm80.h',
    )

    template = """
    constexpr int N = {N};
    constexpr int K = {K};
    constexpr int TileM = {TileM};
    constexpr int TileN = {TileN};
    constexpr int TileK = {TileK};
    constexpr int NStage = {NStage};
    constexpr int WarpM = {WarpM};
    constexpr int WarpN = {WarpN};
    constexpr int WarpK = {WarpK};

    __return_code = call_gemm <
        N, K,
        TileM, TileN, TileK, NStage,
        WarpM, WarpN, WarpK
    > (
        m,
        activation, weight, output,
        activation_scale, weight_scale,
        stream
    );
    """

    runtime = jit.compile_and_tune(
        name='gemm_int4_int4_nt_perchannel',
        includes=includes,
        template=template,
        perf_keys=perf_keys,
        keys=keys,
        space=config_space,
        args=args,
    )

    return runtime.run(*(arg for arg, _ in args))

def gemm_int4_int4_nt_perchannel_naive(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor
):
    def convert(tensor): # uint4 -> int4
        negative_mask = (tensor & 0x8) > 0
        tensor[negative_mask] = tensor[negative_mask] - 16
        return tensor
    
    j_indices = torch.arange(weight.shape[1] * 2, device=weight.device)
    deq_act = convert(((activation[:, j_indices // 2] >> (4 * (j_indices % 2))) & 0xf).to(torch.int32)).to(torch.float16)
    deq_weight = convert(((weight[:, j_indices // 2] >> (4 * (j_indices % 2))) & 0xf).to(torch.int32)).to(torch.float16)
    output.copy_(((deq_act @ deq_weight.t()) * activation_scale.view(-1, 1) * weight_scale.view(1, -1)).to(torch.float16))

def gemm_int4_int4_nt_pergroup(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    group_k: int,
):
    m, k = activation.shape
    n, k_ = weight.shape
    
    assert k * 2 == k_ * 2

    # tensor size check
    assert k * 2 % group_k == 0
    assert activation_scale.shape == (m, k * 2 // group_k)
    assert weight_scale.shape == (n, k * 2 // group_k)
    assert output.shape == (m, n)
    
    # tensor dtype check
    assert activation.dtype == torch.uint8
    assert activation_scale.dtype == torch.float16
    assert weight.dtype == torch.uint8
    assert weight_scale.dtype == torch.float16
    assert output.dtype == torch.float16
    
    # contiguous check
    assert activation.is_contiguous()
    assert activation_scale.is_contiguous()
    assert weight.is_contiguous()
    assert weight_scale.is_contiguous()
    assert output.is_contiguous()

    # key for kernel selection
    perf_keys = {
        'M': m,
    }
    # compile-time keys for specifying the kernel
    keys = {
        'N': n,
        'K': k * 2,
        'GROUP_K': group_k,
    }
    tunable_keys = generate_tunable_keys_pergroup()

    config_space = span_tuning_space(tunable_keys)

    args = (
        (m, 'm'),
        (activation, 'activation'), (weight, 'weight'), (output, 'output'),
        (activation_scale, 'activation_scale'), (weight_scale, 'weight_scale'),
        (torch.cuda.current_stream(), 'stream')
    )

    includes = (
        'cuda/gemm/gemm_w4a4_pergroup_sm80.h',
    )

    template = """
    constexpr int N = {N};
    constexpr int K = {K};
    constexpr int GROUP_K = {GROUP_K};
    constexpr int TileM = {TileM};
    constexpr int TileN = {TileN};
    constexpr int TileK = {TileK};
    constexpr int NStage = {NStage};
    constexpr int WarpM = {WarpM};
    constexpr int WarpN = {WarpN};
    constexpr int WarpK = {WarpK};

    __return_code = call_gemm <
        N, K, GROUP_K,
        TileM, TileN, TileK, NStage,
        WarpM, WarpN, WarpK
    > (
        m,
        activation, weight, output,
        activation_scale, weight_scale,
        stream
    );
    """

    runtime = jit.compile_and_tune(
        name='gemm_int4_int4_nt_pergroup',
        includes=includes,
        template=template,
        perf_keys=perf_keys,
        keys=keys,
        space=config_space,
        args=args,
    )

    return runtime.run(*(arg for arg, _ in args))

def gemm_int4_int4_nt_pergroup_naive(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    group_k: int,
):
    def convert(tensor): # uint4 -> int4
        negative_mask = (tensor & 0x8) > 0
        tensor[negative_mask] = tensor[negative_mask] - 16
        return tensor
    
    j_indices = torch.arange(weight.shape[1] * 2, device=weight.device)
    deq_act = convert(((activation[:, j_indices // 2] >> (4 * (j_indices % 2))) & 0xf).to(torch.int32)).to(torch.float16)
    deq_weight = convert(((weight[:, j_indices // 2] >> (4 * (j_indices % 2))) & 0xf).to(torch.int32)).to(torch.float16)
    output.copy_((
        (deq_act * activation_scale.repeat_interleave(group_k, dim=1))
        @
        (deq_weight * weight_scale.repeat_interleave(group_k, dim=1)).t()
    ).to(torch.float16))
