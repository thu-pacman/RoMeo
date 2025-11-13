import torch
from typing import Dict, List

from .utils import span_tuning_space
from ..jit import jit

includes = (
    'cuda/gemm/gemm_w4a4_sm80.h',
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
    stream
);
"""

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

def gemm_int4_int4_nt(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor
):
    m, k = activation.shape
    n, k_ = weight.shape
    
    assert k * 2 == k_ * 2

    # tensor size check
    assert output.shape == (m, n)
    
    # tensor dtype check
    assert activation.dtype == torch.uint8
    assert weight.dtype == torch.uint8
    assert output.dtype == torch.int32
    
    # contiguous check
    assert activation.is_contiguous()
    assert weight.is_contiguous()
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
        (torch.cuda.current_stream(), 'stream')
    )

    global includes, template

    runtime = jit.compile_and_tune(
        name='gemm_int4_int4_nt',
        includes=includes,
        template=template,
        perf_keys=perf_keys,
        keys=keys,
        space=config_space,
        args=args,
    )

    return runtime.run(*(arg for arg, _ in args))

def gemm_int4_int4_nt_naive(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor
):
    def convert(tensor): # uint4 -> int4
        negative_mask = (tensor & 0x8) > 0
        tensor[negative_mask] = tensor[negative_mask] - 16
        return tensor
    
    j_indices = torch.arange(weight.shape[1] * 2, device=weight.device)
    deq_act = convert(((activation[:, j_indices // 2] >> (4 * (j_indices % 2))) & 0xf).to(torch.int32))
    deq_weight = convert(((weight[:, j_indices // 2] >> (4 * (j_indices % 2))) & 0xf).to(torch.int32))
    output.copy_(deq_act.float() @ deq_weight.t().float()).int()

def gemm_int4_int4_nt_cutlass(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor
):
    m, k = activation.shape
    n, k_ = weight.shape
    
    assert k * 2 == k_ * 2

    # tensor size check
    assert output.shape == (m, n)
    
    # tensor dtype check
    assert activation.dtype == torch.uint8
    assert weight.dtype == torch.uint8
    assert output.dtype == torch.int32
    
    # contiguous check
    assert activation.is_contiguous()
    assert weight.is_contiguous()
    assert output.is_contiguous()

    args = (
        (m, 'm'), (n, 'n'), (k * 2, 'k'),
        (activation, 'activation'), (weight, 'weight'), (output, 'output'),
        (torch.cuda.current_stream(), 'stream')
    )

    includes = (
        'cuda/gemm/gemm_w4a4_cutlass.h',
    )

    template = """
__return_code = call_gemm (
    m, n, k,
    activation, weight, output,
    stream
);
    """

    runtime = jit.compile_and_tune(
        name='gemm_int4_int4_nt_cutlass',
        includes=includes,
        template=template,
        perf_keys={},
        keys={},
        space=({}, ),
        args=args,
    )

    return runtime.run(*(arg for arg, _ in args)
)