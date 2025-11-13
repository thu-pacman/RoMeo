import torch
import argparse
from transformers import AutoConfig

from vllm.platforms import current_platform
from vllm.utils import FlexibleArgumentParser

def benchmark(
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    num_iters: int = 100,
    print_result: bool = False
) -> float:
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16) / 10
    w1 = torch.randn(num_experts,
                    shard_intermediate_size,
                    hidden_size,
                    dtype=torch.float16).to(torch.float8_e4m3fn)
    w2 = torch.randn(num_experts,
                    hidden_size,
                    shard_intermediate_size // 2,
                    dtype=torch.float16).to(torch.float8_e4m3fn)
    gating_output = torch.randn(num_iters,
                                num_tokens,
                                num_experts,
                                dtype=torch.float32)
    block_shape = [128, 128]
    w1_scale = torch.randn((w1.shape[0], w1.shape[1] // 128, w1.shape[2] // 128),
                            dtype=torch.float32)
    w2_scale = torch.randn((w2.shape[0], w2.shape[1] // 128, w2.shape[2] // 128),
                            dtype=torch.float32)

    input_gating = torch.empty(num_tokens, num_experts, dtype=torch.float32)

    def prepare(i: int):
        input_gating.copy_(gating_output[i])

    def run():
        return fused_moe(
            hidden_states=x,
            w1=w1,
            w2=w2,
            gating_output=input_gating,
            topk=topk,
            renormalize=True,
            inplace=True,
            use_fp8_w8a8=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            block_shape=block_shape
        )

    ret = run()
    torch.cuda.synchronize()

    if print_result:
        print(ret)
        return

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    # Warmup
    for _ in range(5):
        graph.replay()

    for i in range(num_iters):
        prepare(i)
        # torch.cuda.synchronize()
        start_event[i].record()
        graph.replay()
        end_event[i].record()
        # end_event.synchronize()
    torch.cuda.synchronize()

    latencies = [start_event[i].elapsed_time(end_event[i]) for i in range(num_iters)]
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    graph.reset()
    return avg

def main(args: argparse.Namespace):
    print(args)

    config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code)
    if config.architectures[0] == "DeepseekV3ForCausalLM":
        E = config.n_routed_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    else:
        raise NotImplementedError(f"Model {args.model} not supported")

    hidden_size = config.hidden_size

    if args.batch_size is None:
        batch_sizes = [
            # 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096
            64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096
        ]
    else:
        batch_sizes = [args.batch_size]

    # warmup & tuning & validate round
    benchmark(256, E, shard_intermediate_size, hidden_size, topk, print_result=True)
    # return

    outputs = [benchmark(bs, E, shard_intermediate_size, hidden_size, topk) for bs in batch_sizes]

    for batch_size, kernel_time in zip(batch_sizes, outputs):
        print(f"Batch size: {batch_size} Kernel time: {kernel_time:.2f} us")

if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    current_platform.seed_everything(args.seed)

    from vllm_fused_moe_inject import fused_moe

    main(args)

    from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe

    main(args)