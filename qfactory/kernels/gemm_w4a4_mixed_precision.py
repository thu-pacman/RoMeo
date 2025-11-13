import os
import torch
from typing import Dict, List

from .utils import span_tuning_space
from ..jit import jit

NO_PIPELINE = os.environ.get('QFACTORY_NO_PIPELINE', 0)

def generate_tunable_keys() -> Dict[str, List[int]]:
    return {
        'NStage': [1] if NO_PIPELINE else [1, 2, 3],
        'TileM': [128, 256],
        'TileN': [128, 256],
        'TileK': [128, 256],
        'WarpM': [64],
        'WarpN': [64],
        'WarpK': [128],
    }

def gemm_int4_int4_nt_mixed_precision(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    act_outlier: torch.Tensor,
    act_scale_outlier: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_outlier: torch.Tensor,
    weight_scale_outlier: torch.Tensor,
    output: torch.Tensor,
    a_outlier: int,
    w_outlier: int,
):
    m, k2 = act.shape
    n, k2_ = weight.shape

    k = k2 * 2

    assert k == k2_ * 2

    # tensor size check
    assert act_scale.shape == (m,)
    assert weight_scale.shape == (n,)
    assert act_outlier.shape == (a_outlier, k)
    assert act_scale_outlier.shape == (a_outlier,)
    assert weight_outlier.shape == (w_outlier, k)
    assert weight_scale_outlier.shape == (w_outlier,)
    assert output.shape == (m + a_outlier, n + w_outlier)
    
    # tensor dtype check
    assert act.dtype == torch.uint8
    assert act_scale.dtype == torch.bfloat16
    assert act_outlier.dtype == torch.uint8
    assert act_scale_outlier.dtype == torch.bfloat16
    assert weight.dtype == torch.uint8
    assert weight_scale.dtype == torch.bfloat16
    assert weight_outlier.dtype == torch.uint8
    assert weight_scale_outlier.dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16
    
    # contiguous check
    assert act.is_contiguous()
    assert act_scale.is_contiguous()
    assert act_outlier.is_contiguous()
    assert act_scale_outlier.is_contiguous()
    assert weight.is_contiguous()
    assert weight_scale.is_contiguous()
    assert weight_outlier.is_contiguous()
    assert weight_scale_outlier.is_contiguous()
    assert output.is_contiguous()

    # key for kernel selection
    perf_keys = {
        'M': m,
    }
    # compile-time keys for specifying the kernel
    keys = {
        'N': n,
        'K': k,
        'W_OUTLIER': w_outlier,
    }
    tunable_keys = generate_tunable_keys()

    config_space = span_tuning_space(tunable_keys)

    args = (
        (m, 'm'), (output, 'output'),
        (act, 'act'), (weight, 'weight'),
        (act_outlier, 'act_outlier'), (weight_outlier, 'weight_outlier'),
        (act_scale, 'act_scale'), (weight_scale, 'weight_scale'),
        (act_scale_outlier, 'act_scale_outlier'), (weight_scale_outlier, 'weight_scale_outlier'),
        (a_outlier, 'a_outlier'),
        (torch.cuda.current_stream(), 'stream')
    )

    includes = (
        'cuda/gemm/gemm_w4a4_mixed_precision_sm80.h',
    )

    template = """
    constexpr int N = {N};
    constexpr int K = {K};
    constexpr int Outlier_W = {W_OUTLIER};
    constexpr int TileM = {TileM};
    constexpr int TileN = {TileN};
    constexpr int TileK = {TileK};
    constexpr int NStage = {NStage};
    constexpr int WarpM = {WarpM};
    constexpr int WarpN = {WarpN};
    constexpr int WarpK = {WarpK};

    __return_code = call_gemm <
        N, K, Outlier_W,
        TileM, TileN, TileK, NStage,
        WarpM, WarpN, WarpK
    > (
        m, output,
        act, weight, act_outlier, weight_outlier,
        act_scale, weight_scale, act_scale_outlier, weight_scale_outlier,
        a_outlier,
        stream
    );
    """

    runtime = jit.compile_and_tune(
        name='gemm_int4_int4_nt_mixed_precision',
        includes=includes,
        template=template,
        perf_keys=perf_keys,
        keys=keys,
        space=config_space,
        args=args,
    )

    return runtime.run(*(arg for arg, _ in args))

def gemm_int4_int4_nt_mixed_precision_naive(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    outlier: int,
):
    def convert4(tensor): # uint4 -> int4
        negative_mask = (tensor & 0x8) > 0
        tensor[negative_mask] = tensor[negative_mask] - 16
        return tensor
    
    def convert8(tensor): # uint8 -> int8
        negative_mask = (tensor & 0x80) > 0
        tensor[negative_mask] = tensor[negative_mask] - 256
        return tensor
    
    j_indices = torch.arange(weight.shape[1] * 2, device=weight.device)
    deq_act = torch.empty((activation.shape[0] - outlier, weight.shape[1] * 2), dtype=torch.float32, device=activation.device)
    deq_act[:-outlier] = convert4(((activation[:-2 * outlier, j_indices // 2] >> (4 * (j_indices % 2))) & 0xf).to(torch.int32)).to(torch.float32)
    deq_act[-outlier:] = convert8(activation[-2 * outlier:].view(outlier, weight.shape[1] * 2).to(torch.int32)).to(torch.float32)
    deq_weight = torch.empty((weight.shape[0] - outlier, weight.shape[1] * 2), dtype=torch.float32, device=weight.device)
    deq_weight[:-outlier] = convert4(((weight[:-2 * outlier, j_indices // 2] >> (4 * (j_indices % 2))) & 0xf).to(torch.int32)).to(torch.float32)
    deq_weight[-outlier:] = convert8(weight[-2 * outlier:].view(outlier, weight.shape[1] * 2).to(torch.int32)).to(torch.float32)
    output.copy_(((deq_act * activation_scale.view(-1, 1)) @ (deq_weight * weight_scale.view(-1, 1)).t()).to(torch.bfloat16))

def gemm_mixed_nt_perchannel(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    name: str
):
    m, n = activation.shape[0], weight.shape[0]
    if name == 'a4w4':
        assert activation.shape[1] == weight.shape[1]
        k = activation.shape[1] * 2
    elif name == 'a8w8':
        assert activation.shape[1] == weight.shape[1]
        k = activation.shape[1]
    elif name == 'a4w8':
        assert activation.shape[1] * 2 == weight.shape[1]
        k = weight.shape[1]
    elif name == 'a8w4':
        assert activation.shape[1] == weight.shape[1] * 2
        k = activation.shape[1]
    else:
        raise ValueError(f"Unsupported mixed precision name: {name}")
    
    # tensor size check
    assert activation_scale.shape == (m,)
    assert weight_scale.shape == (n,)
    assert output.shape == (m, n)
    
    # tensor dtype check
    assert activation.dtype == torch.uint8
    assert activation_scale.dtype == torch.bfloat16
    assert weight.dtype == torch.uint8
    assert weight_scale.dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16
    
    # contiguous check
    assert activation.is_contiguous()
    assert activation_scale.is_contiguous()
    assert weight.is_contiguous()
    assert weight_scale.is_contiguous()
    # assert output.is_contiguous()
    output_stride = output.stride()
    assert len(output_stride) == 2 and output_stride[1] == 1

    # key for kernel selection
    perf_keys = {
        'M': m,
    }
    # compile-time keys for specifying the kernel
    keys = {
        'N': n,
        'K': k,
        'LDC': output_stride[0],
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
        f'cuda/gemm/mixed_precision/gemm_{name}_perchannel_sm80.h',
    )

    template = """
    constexpr int N = {N};
    constexpr int K = {K};
    constexpr int LDC = {LDC};
    constexpr int TileM = {TileM};
    constexpr int TileN = {TileN};
    constexpr int TileK = {TileK};
    constexpr int NStage = {NStage};
    constexpr int WarpM = {WarpM};
    constexpr int WarpN = {WarpN};
    constexpr int WarpK = {WarpK};

    __return_code = call_gemm <
        N, K, LDC,
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
        name=f'gemm_mixed_{name}_nt_perchannel',
        includes=includes,
        template=template,
        perf_keys=perf_keys,
        keys=keys,
        space=config_space,
        args=args,
    )

    return runtime.run(*(arg for arg, _ in args))

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
s3 = torch.cuda.Stream()

def gemm_int4_int4_nt_mixed_precision_multistream(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    outlier: int,
):
    m, k2 = activation.shape
    n, k2_ = weight.shape

    k = k2 * 2
    m -= outlier
    n -= outlier
    
    assert k == k2_ * 2

    # tensor size check
    assert activation_scale.shape == (m,)
    assert weight_scale.shape == (n,)
    assert output.shape == (m, n)
    
    # tensor dtype check
    assert activation.dtype == torch.uint8
    assert activation_scale.dtype == torch.bfloat16
    assert weight.dtype == torch.uint8
    assert weight_scale.dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16
    
    # contiguous check
    assert activation.is_contiguous()
    assert activation_scale.is_contiguous()
    assert weight.is_contiguous()
    assert weight_scale.is_contiguous()
    assert output.is_contiguous()

    global s1, s2, s3
    
    e0 = torch.cuda.current_stream().record_event()
    s1.wait_event(e0)
    s2.wait_event(e0)
    s3.wait_event(e0)

    gemm_mixed_nt_perchannel(
        activation[:m - outlier], activation_scale[:m - outlier],
        weight[:n - outlier], weight_scale[:n - outlier],
        output[:m - outlier, :n - outlier],
        name='a4w4'
    )

    with torch.cuda.stream(s1):
        gemm_mixed_nt_perchannel(
            activation[m - outlier:].view(outlier, -1), activation_scale[m - outlier:],
            weight[n - outlier:].view(outlier, -1), weight_scale[n - outlier:],
            output[m - outlier:, n - outlier:],
            name='a8w8'
        )
    e1 = s1.record_event()

    with torch.cuda.stream(s2):
        gemm_mixed_nt_perchannel(
            activation[:m - outlier], activation_scale[:m - outlier],
            weight[n - outlier:].view(outlier, -1), weight_scale[n - outlier:],
            output[:m - outlier, n - outlier:],
            name='a4w8'
        )
    e2 = s2.record_event()

    with torch.cuda.stream(s3):
        gemm_mixed_nt_perchannel(
            activation[m - outlier:].view(outlier, -1), activation_scale[m - outlier:],
            weight[:n - outlier], weight_scale[:n - outlier],
            output[m - outlier:, :n - outlier],
            name='a8w4'
        )
    e3 = s3.record_event()

    torch.cuda.current_stream().wait_event(e1)
    torch.cuda.current_stream().wait_event(e2)
    torch.cuda.current_stream().wait_event(e3)

    e4 = torch.cuda.current_stream().record_event()
    s1.wait_event(e4)
    s2.wait_event(e4)
    s3.wait_event(e4)

def gemm_int4_int4_nt_mixed_precision_separate(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    act_outlier: torch.Tensor,
    act_scale_outlier: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_outlier: torch.Tensor,
    weight_scale_outlier: torch.Tensor,
    output: torch.Tensor,
    a_outlier: int,
    w_outlier: int,
    streams: List[torch.cuda.Stream]
):
    m, k2 = act.shape
    n, k2_ = weight.shape

    k = k2 * 2
    
    assert k == k2_ * 2

    # tensor size check
    assert act_scale.shape == (m,)
    assert weight_scale.shape == (n,)
    assert act_outlier.shape == (a_outlier, k)
    assert act_scale_outlier.shape == (a_outlier,)
    assert weight_outlier.shape == (w_outlier, k)
    assert weight_scale_outlier.shape == (w_outlier,)
    assert output.shape == (m + a_outlier, n + w_outlier)
    
    # tensor dtype check
    assert act.dtype == torch.uint8
    assert act_scale.dtype == torch.bfloat16
    assert act_outlier.dtype == torch.uint8
    assert act_scale_outlier.dtype == torch.bfloat16
    assert weight.dtype == torch.uint8
    assert weight_scale.dtype == torch.bfloat16
    assert weight_outlier.dtype == torch.uint8
    assert weight_scale_outlier.dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16
    
    # contiguous check
    assert act.is_contiguous()
    assert act_scale.is_contiguous()
    assert act_outlier.is_contiguous()
    assert act_scale_outlier.is_contiguous()
    assert weight.is_contiguous()
    assert weight_scale.is_contiguous()
    assert weight_outlier.is_contiguous()
    assert weight_scale_outlier.is_contiguous()
    assert output.is_contiguous()

    assert len(streams) == 4

    with torch.cuda.stream(streams[0]):
        gemm_mixed_nt_perchannel(
            act, act_scale,
            weight, weight_scale,
            output[:-a_outlier, :-w_outlier],
            name='a4w4'
        )

    with torch.cuda.stream(streams[1]):
        gemm_mixed_nt_perchannel(
            act_outlier, act_scale_outlier,
            weight_outlier, weight_scale_outlier,
            output[-a_outlier:, -w_outlier:],
            name='a8w8'
        )

    with torch.cuda.stream(streams[2]):
        gemm_mixed_nt_perchannel(
            act, act_scale,
            weight_outlier, weight_scale_outlier,
            output[:-a_outlier, -w_outlier:],
            name='a4w8'
        )

    with torch.cuda.stream(streams[3]):
        gemm_mixed_nt_perchannel(
            act_outlier, act_scale_outlier,
            weight, weight_scale,
            output[-a_outlier:, :-w_outlier],
            name='a8w4'
        )

    return [s.record_event() for s in streams]
