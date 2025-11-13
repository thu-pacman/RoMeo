import torch
from typing import Dict, List

from .utils import span_tuning_space
from ..jit import jit

def generate_tunable_keys(gs1, gs2) -> Dict[str, List[int]]:
    ret = {
        'NStage': [1, 2, 3],
        'TileM': [128, 256],
        'TileN': [128, 256],
        'TileK': [],
        'WarpM': [64],
        'WarpN': [64],
        'WarpK': [128],
    }
    if gs1 % 256 == 0 and gs2 % 256 == 0:
        ret['TileK'] = [128, 256]
    elif gs1 % 128 == 0 and gs2 % 128 == 0:
        ret['TileK'] = [128]
    else:
        raise ValueError(
            f"Invalid group size: {gs1}, {gs2}. "
            "Group size must be divisible by 128 or 256."
        )
    return ret

def gemm_int4_int4_nt_elastic(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    outlier: int,
    div: int,
):
    m, k2 = activation.shape
    n, k2_ = weight.shape

    k = k2 * 2
    assert k == k2_ * 2

    assert (k - outlier) % div == 0
    num_groups = 1 + div

    # tensor size check
    assert activation_scale.shape == (m, num_groups)
    assert weight_scale.shape == (n, num_groups)
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
        'K': k,
        'OUTLIER': outlier,
        'DIV': div,
    }
    tunable_keys = generate_tunable_keys(outlier, (k - outlier) // div)

    config_space = span_tuning_space(tunable_keys)

    args = (
        (m, 'm'),
        (activation, 'activation'), (weight, 'weight'), (output, 'output'),
        (activation_scale, 'activation_scale'), (weight_scale, 'weight_scale'),
        (torch.cuda.current_stream(), 'stream')
    )

    includes = (
        'cuda/gemm/gemm_w4a4_elastic_sm80.h',
    )

    template = """
    constexpr int N = {N};
    constexpr int K = {K};
    constexpr int Outlier = {OUTLIER};
    constexpr int Div = {DIV};
    constexpr int TileM = {TileM};
    constexpr int TileN = {TileN};
    constexpr int TileK = {TileK};
    constexpr int NStage = {NStage};
    constexpr int WarpM = {WarpM};
    constexpr int WarpN = {WarpN};
    constexpr int WarpK = {WarpK};

    __return_code = call_gemm <
        N, K, Outlier, Div,
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
        name='gemm_int4_int4_nt_elastic',
        includes=includes,
        template=template,
        perf_keys=perf_keys,
        keys=keys,
        space=config_space,
        args=args,
    )

    return runtime.run(*(arg for arg, _ in args))

def gemm_int4_int4_nt_elastic_naive(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    outlier: int,
    div: int,
):
    def convert(tensor): # uint4 -> int4
        negative_mask = (tensor & 0x8) > 0
        tensor[negative_mask] = tensor[negative_mask] - 16
        return tensor
    
    output.zero_()
    num_groups = 1 + div
    group_size = (activation.shape[1] * 2 - outlier) // div
    for i in range(num_groups):
        if i == 0: # outlier group
            j_indices = torch.arange(outlier, device=weight.device)
        else:
            j_indices = outlier + (i - 1) * group_size + torch.arange(group_size, device=weight.device)
        deq_act = convert(((activation[:, j_indices // 2] >> (4 * (j_indices % 2))) & 0xf).to(torch.int32)).to(torch.float16)
        deq_weight = convert(((weight[:, j_indices // 2] >> (4 * (j_indices % 2))) & 0xf).to(torch.int32)).to(torch.float16)
        output += ((deq_act * activation_scale[:, i:i+1]) @ (deq_weight * weight_scale[:, i:i+1]).t()).to(torch.float16)
