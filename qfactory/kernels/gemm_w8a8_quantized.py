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

def gemm_int8_int8_nt_perchannel(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor
):
    m, k = activation.shape
    n, k_ = weight.shape
    
    assert k == k_

    # tensor size check
    assert activation_scale.shape == (m,)
    assert weight_scale.shape == (n,)
    assert output.shape == (m, n)
    
    # tensor dtype check
    assert activation.dtype == torch.int8
    assert activation_scale.dtype == torch.float16
    assert weight.dtype == torch.int8
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
        'K': k,
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
        'cuda/gemm/gemm_w8a8_perchannel_sm80.h',
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
        name='gemm_int8_int8_nt_perchannel',
        includes=includes,
        template=template,
        perf_keys=perf_keys,
        keys=keys,
        space=config_space,
        args=args,
    )

    return runtime.run(*(arg for arg, _ in args))

def gemm_int8_int8_nt_perchannel_naive(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor
):
    if activation.dtype == torch.int8 and weight.dtype == torch.int8:
        output.copy_((activation.to(torch.float32) * activation_scale.view(-1, 1)) @ (weight.to(torch.float32) * weight_scale.view(-1, 1)).t())
    elif activation.dtype == torch.uint8 and weight.dtype == torch.uint8:
        def convert8(tensor): # uint8 -> int8
            negative_mask = (tensor & 0x80) > 0
            tensor[negative_mask] = tensor[negative_mask] - 256
            return tensor
        output.copy_((convert8(activation.to(torch.int32)).to(torch.float32) * activation_scale.view(-1, 1)) @ (convert8(weight.to(torch.int32)).to(torch.float32) * weight_scale.view(-1, 1)).t())
    else:
        raise ValueError(f"Unsupported data types for activation ({activation.dtype}) and weight ({weight.dtype}). Expected int8 or uint8.")
