import torch
from typing import Dict, List

from .utils import span_tuning_space
from ..jit import jit

includes = (
    'cuda/gemm/gemm_w4a16_sm90.h',
)

template = """
constexpr int N = {N};
constexpr int K = {K};
constexpr int TileM = {TileM};
constexpr int TileN = {TileN};
constexpr int TileK = {TileK};
constexpr int NStage = {NStage};

__return_code = call_gemm <
    N, K,
    TileM, TileN, TileK, NStage
> (
    m,
    activation, weight, output,
    stream
);
"""

def generate_tunable_keys() -> Dict[str, List[int]]:
    return {
        'NStage': [1],
        'TileM': [64],
        'TileN': [128],
        'TileK': [64],
    }

def gemm_int4_fp16_nt(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor
):
    m, k = activation.shape
    n, k_ = weight.shape
    
    assert k == k_ * 2

    # tensor size check
    assert output.shape == (m, n)
    
    # tensor dtype check
    assert activation.dtype == torch.float16
    assert weight.dtype == torch.uint8
    assert output.dtype == torch.float16
    
    # contiguous check
    assert activation.is_contiguous()
    assert weight.is_contiguous()
    assert output.is_contiguous()

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
        (torch.cuda.current_stream(), 'stream')
    )

    global includes, template

    runtime = jit.compile_and_tune(
        name='gemm_int4_fp16_nt',
        includes=includes,
        template=template,
        keys=keys,
        space=config_space,
        args=args,
    )

    return runtime.run(*(arg for arg, _ in args))

def gemm_int4_fp16_nt_naive(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor
):
    j_indices = torch.arange(weight.shape[1] * 2, device=weight.device)
    deq_weight = ((weight[:, j_indices // 2] >> (4 * (j_indices % 2))) & 0xf).to(torch.float16)
    deq_weight -= 8.0
    output.copy_(activation @ deq_weight.t())