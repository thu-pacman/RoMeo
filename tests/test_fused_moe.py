import torch
import qfactory
import triton.language as tl
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_kernel
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

def run_vllm(activation, activation_scale, weight, weight_scale, output, token_ids, expert_ids, moe_weight, block_size):
    M = activation.shape[0]
    topk = output.shape[0] // M
    N = output.shape[1]
    _output = torch.empty_like(output).reshape(M, topk, N)
    invoke_fused_moe_kernel(
        activation,
        weight,
        _output,
        activation_scale,
        weight_scale,
        None,
        moe_weight,
        torch.arange(0, M * topk, dtype=torch.int32, device='cuda').reshape(M, topk),
        token_ids,
        expert_ids,
        torch.tensor(num_chunks * chunk_size, dtype=torch.int32, device='cuda'),
        moe_weight is not None,
        topk,
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        tl.bfloat16,
        True,
        False,
        False,
        block_size
    )
    return _output.reshape(M * topk, N)

if __name__ == '__main__':
    torch.manual_seed(42)

    num_chunks = 64
    M = 128
    E = 256
    N = 512
    K = 7168
    topk = 8
    blockN = 128
    blockK = 128
    chunk_size = 64
    
    assert M * topk <= num_chunks * chunk_size

    activation = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    weight = torch.randn(E, N, K, dtype=torch.float16, device='cuda').to(torch.float8_e4m3fn)
    weight_scale = torch.randn(E, N // blockN, K // blockK, dtype=torch.float32, device='cuda')
    output = torch.empty(M * topk, N, dtype=torch.bfloat16, device='cuda')
    token_ids = torch.randperm(num_chunks * chunk_size, dtype=torch.int32, device='cuda')
    token_ids = token_ids.clamp_max(M * topk)
    expert_ids = torch.randint(0, E, (num_chunks,), dtype=torch.int32, device='cuda')
    moe_weight = torch.randn(M, topk, dtype=torch.float32, device='cuda')

    # online scaling
    A, A_scale = per_token_group_quant_fp8(activation, blockK)

    qfactory.fused_moe_fp8_fp8_bf16_nt(
        torch.tensor(num_chunks, dtype=torch.int, device='cuda'),
        (A, A_scale),
        (weight, weight_scale),
        output,
        token_ids,
        expert_ids,
        chunk_size,
        topk,
        blockN,
        blockK,
        moe_weight
    )

    naive_output = torch.empty_like(output)
    qfactory.fused_moe_fp8_fp8_bf16_nt_naive(
        torch.tensor(num_chunks, dtype=torch.int, device='cuda'),
        (A, A_scale),
        (weight, weight_scale),
        naive_output,
        token_ids,
        expert_ids,
        chunk_size,
        topk,
        blockN,
        blockK,
        moe_weight
    )

    std_output = run_vllm(activation, None, weight, weight_scale, output, token_ids, expert_ids, moe_weight, [blockN, blockK])

    # torch.cuda.synchronize()
    print('Comparing outputs...')
    print(output)
    print(naive_output)
    print(std_output)

    diff = calc_diff(naive_output, std_output)
    print(f"diff between naive and std: {diff}")
    assert diff < 1e-2

    diff = calc_diff(naive_output, output)
    print(f"diff between naive and qfactory: {diff}")
    assert diff < 1e-2
    print('PASSED')