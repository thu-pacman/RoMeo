import os
import sys
import torch
import argparse
import transformers
from tqdm import trange
from functools import partial

sys.path.insert(0, os.getcwd())
from eval import parse_args as eval_parse_args

ALL_EVENTS = [] # Store all CUDA events to make CUDA Graph happy

def bench_model(model, tokenizer, batch, seqlen):
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch, seqlen), dtype=torch.long, device='cuda')
    with torch.inference_mode():
        for _ in trange(3, desc="Warmup."):
            model(input_ids).logits.mean().cpu()
        for _ in trange(5, desc="Benchmarking."):
            model(input_ids).logits.mean().cpu()

def inject(layer, baseline, multistream, verbose):
    attn_start_events = []
    attn_end_events = []

    all_start_events = {
        key: {
            'q': [], 'k': [], 'v': [], 'o': [],
            'u': [], 'g': [], 'd': [],
        } for key in ['gemm', 'quant', 'post_mul', 'hadamard']
    }
    all_end_events = {
        key: {
            'q': [], 'k': [], 'v': [], 'o': [],
            'u': [], 'g': [], 'd': [],
        } for key in ['gemm', 'quant', 'post_mul', 'hadamard']
    }

    # Inject Attn
    def wrap_attn(module):
        import math
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        # from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            return q_embed, k_embed
        
        def Qwen3Attention_forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values = None,
            cache_position = None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            if baseline == "bitweaver":
                quant_start_event = torch.cuda.Event(enable_timing=True)
                quant_end_event = torch.cuda.Event(enable_timing=True)

                torch.cuda.set_device(hidden_states.device)
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                a_outliers = math.ceil(hidden_states.shape[0] * module.q_proj.linear_impl.p_a_outlier / 256) * 256
                output_q = torch.empty(hidden_states.shape[0] + a_outliers, module.q_proj.linear_impl.w.shape[0] - module.q_proj.linear_impl.w_outliers, dtype=torch.bfloat16, device=hidden_states.device)
                output_kv = torch.empty(hidden_states.shape[0] + a_outliers, module.k_proj.linear_impl.w.shape[0] - module.k_proj.linear_impl.w_outliers, dtype=torch.bfloat16, device=hidden_states.device)
                
                quant_start_event.record()
                hidden_states = module.q_proj.linear_impl.quant_method(hidden_states, a_outliers)
                quant_end_event.record()
                hidden_states_q = (output_q, a_outliers, *hidden_states)
                hidden_states_kv = (output_kv, a_outliers, *hidden_states)

                if multistream:
                    all_start_events['gemm']['q'].append(quant_start_event)
                else:
                    all_start_events['quant']['q'].append(quant_start_event)
                    all_end_events['quant']['q'].append(quant_end_event)
                global ALL_EVENTS
                ALL_EVENTS.append((quant_start_event, quant_end_event))
                # hidden_states is now a tuple, mixedqlinear will skip quant if passed in a tuple
            else:
                hidden_states_q = hidden_states_kv = hidden_states

            q_proj_out = self.q_proj(hidden_states_q)
            k_proj_out = self.k_proj(hidden_states_kv)
            v_proj_out = self.v_proj(hidden_states_kv)
            query_states = self.q_norm(q_proj_out.view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(k_proj_out.view(hidden_shape)).transpose(1, 2)
            value_states = v_proj_out.view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

            assert self.config._attn_implementation != "eager"
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,  # diff with Llama
                **kwargs,
            )
            end_event.record()

            attn_start_events.append(start_event)
            attn_end_events.append(end_event)

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights
        return partial(Qwen3Attention_forward, module)

    layer.self_attn.forward = wrap_attn(layer.self_attn)

    def attn_clear():
        attn_start_events.clear()
        attn_end_events.clear()
    
    def attn_stat():
        if verbose:
            print(f"attn: {len(attn_start_events)} events recorded")
        assert(len(attn_start_events) == len(attn_end_events))
        total_time = 0
        for start, end in zip(attn_start_events, attn_end_events):
            total_time += start.elapsed_time(end)
        return total_time / len(attn_start_events)
    
    # Inject MLP
    def wrap_mlp(module):
        # from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
        def Qwen3MLP_forward(self, x):
            assert len(x.shape) == 3
            nbatch = x.shape[0]

            if baseline == "bitweaver":
                quant_start_event = torch.cuda.Event(enable_timing=True)
                quant_end_event = torch.cuda.Event(enable_timing=True)

                torch.cuda.set_device(x.device)
                x = x.view(-1, x.shape[-1])
                a_outliers = math.ceil(x.shape[0] * module.up_proj.linear_impl.p_a_outlier / 256) * 256
                output = torch.empty(x.shape[0] + a_outliers, module.up_proj.linear_impl.w.shape[0] - module.up_proj.linear_impl.w_outliers, dtype=torch.bfloat16, device=x.device)

                quant_start_event.record()
                x = module.up_proj.linear_impl.quant_method(x, a_outliers)
                quant_end_event.record()
                x = (output, a_outliers, *x)

                if multistream:
                    all_start_events['gemm']['g'].append(quant_start_event)
                else:
                    all_start_events['quant']['u'].append(quant_start_event)
                    all_end_events['quant']['u'].append(quant_end_event)
                global ALL_EVENTS
                ALL_EVENTS.append((quant_start_event, quant_end_event))
                # x is now a tuple, mixedqlinear will skip quant if passed in a tuple

            gate_proj_out = self.gate_proj(x)
            up_proj_out = self.up_proj(x)
            down_proj = self.down_proj(self.act_fn(gate_proj_out) * up_proj_out)
            down_proj = down_proj.view(nbatch, -1, self.hidden_size)
            return down_proj
        return partial(Qwen3MLP_forward, module)

    layer.mlp.forward = wrap_mlp(layer.mlp)
    
    # inject Linear
    if baseline == 'bf16':
        def wrap(kernel, name):
            def wrapped(*args, **kwargs):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                with torch.cuda.nvtx.range(f"Gemm-{name}"):
                    start_event.record()
                    output = kernel(*args, **kwargs)
                    end_event.record()
                all_start_events['gemm'][name].append(start_event)
                all_end_events['gemm'][name].append(end_event)
                return output
            return wrapped
        
        layer.self_attn.q_proj.forward = wrap(layer.self_attn.q_proj.forward, 'q')
        layer.self_attn.k_proj.forward = wrap(layer.self_attn.k_proj.forward, 'k')
        layer.self_attn.v_proj.forward = wrap(layer.self_attn.v_proj.forward, 'v')
        layer.self_attn.o_proj.forward = wrap(layer.self_attn.o_proj.forward, 'o')
        layer.mlp.up_proj.forward = wrap(layer.mlp.up_proj.forward, 'u')
        layer.mlp.gate_proj.forward = wrap(layer.mlp.gate_proj.forward, 'g')
        layer.mlp.down_proj.forward = wrap(layer.mlp.down_proj.forward, 'd')
    else:
        import math
        from qfactory import gemm_int4_int4_nt_mixed_precision_separate

        def wrap_mixedlinear(linear, name):
            def BitWeaver_forward(self, inp):
                start_events = {
                    key: torch.cuda.Event(enable_timing=True) for key in ['gemm', 'quant', 'post_mul']
                }
                end_events = {
                    key: torch.cuda.Event(enable_timing=True) for key in ['gemm', 'quant', 'post_mul']
                }

                if isinstance(inp, tuple):
                    output, a_outliers, act, act_scale, act_outlier, act_scale_outlier, inp_outlier_idx = inp
                else:
                    torch.cuda.set_device(inp.device)
                    a_outliers = math.ceil(inp.shape[0] * self.p_a_outlier / 256) * 256
                    output = torch.empty(inp.shape[0] + a_outliers, self.w.shape[0] - self.w_outliers, dtype=torch.bfloat16, device=inp.device)

                    start_events['quant'].record()
                    act, act_scale, act_outlier, act_scale_outlier, inp_outlier_idx = self.act_quantize(inp, a_outliers)
                    end_events['quant'].record()
                
                start_events['gemm'].record()
                self.kernel(
                    act, act_scale,
                    act_outlier, act_scale_outlier,
                    self.w[:-2 * self.w_outliers], self.w_scale[:-self.w_outliers],
                    self.w[-2 * self.w_outliers:].view(self.w_outliers, -1), self.w_scale[-self.w_outliers:],
                    output, a_outliers, self.w_outliers,
                    [torch.cuda.current_stream() for _ in range(4)]
                )
                end_events['gemm'].record()
                
                start_events['post_mul'].record()
                output[inp_outlier_idx] = output[-a_outliers:].clone()
                output[:, self.w_idx] = output[:, -self.w_outliers:]
                ret = output[:-a_outliers, :-self.w_outliers]
                end_events['post_mul'].record()

                for key in ['gemm', 'quant', 'post_mul']:
                    if key == 'quant' and isinstance(inp, tuple):
                        continue
                    all_start_events[key][name].append(start_events[key])
                    all_end_events[key][name].append(end_events[key])
                
                return ret
        
            def BitWeaver_forward_multistream(self, inp):
                gemm_start_event = torch.cuda.Event(enable_timing=True)
                gemm_end_event = torch.cuda.Event(enable_timing=True)
                post_mul_start_event = torch.cuda.Event(enable_timing=True)
                post_mul_end_event = torch.cuda.Event(enable_timing=True)

                if isinstance(inp, tuple):
                    output, a_outliers, act, act_scale, act_outlier, act_scale_outlier, inp_outlier_idx, e_finish_normal, e_finish_outliers = inp
                else:
                    torch.cuda.set_device(inp.device)
                    a_outliers = math.ceil(inp.shape[0] * self.p_a_outlier / 256) * 256
                    output = torch.empty(inp.shape[0] + a_outliers, self.w.shape[0] - self.w_outliers, dtype=torch.bfloat16, device=inp.device)

                    gemm_start_event.record()
                    act, act_scale, act_outlier, act_scale_outlier, inp_outlier_idx, e_finish_normal, e_finish_outliers = self.act_quantize_multistream(inp, a_outliers)
                
                self.streams[2].wait_event(e_finish_normal)
                self.streams[3].wait_event(e_finish_outliers)
                self.streams[4].wait_event(e_finish_normal)
                self.streams[5].wait_event(e_finish_outliers)

                es_finish = gemm_int4_int4_nt_mixed_precision_separate(
                    act, act_scale,
                    act_outlier, act_scale_outlier,
                    self.w[:-2 * self.w_outliers], self.w_scale[:-self.w_outliers],
                    self.w[-2 * self.w_outliers:].view(self.w_outliers, -1), self.w_scale[-self.w_outliers:],
                    output, a_outliers, self.w_outliers,
                    self.streams[2:]
                )

                for e in es_finish:
                    torch.cuda.current_stream().wait_event(e)
                gemm_end_event.record()

                post_mul_start_event.record()
                output[inp_outlier_idx] = output[-a_outliers:].clone()
                output[:, self.w_idx] = output[:, -self.w_outliers:]
                ret = output[:-a_outliers, :-self.w_outliers]
                post_mul_end_event.record()

                if name == 'o' or name == 'd':
                    all_start_events['gemm'][name].append(gemm_start_event)
                    all_end_events['gemm'][name].append(gemm_end_event)
                    all_start_events['post_mul'][name].append(post_mul_start_event)
                    all_end_events['post_mul'][name].append(post_mul_end_event)
                else:
                    if name == 'v':
                        all_end_events['gemm']['q'].append(post_mul_end_event)
                    if name == 'u':
                        all_end_events['gemm']['g'].append(post_mul_end_event)
                global ALL_EVENTS
                ALL_EVENTS.append((gemm_start_event, gemm_end_event, post_mul_start_event, post_mul_end_event))

                return ret
            
            return partial(BitWeaver_forward_multistream, linear) if multistream else partial(BitWeaver_forward, linear)
        
        layer.self_attn.q_proj.linear_impl.forward = wrap_mixedlinear(layer.self_attn.q_proj.linear_impl, 'q')
        layer.self_attn.k_proj.linear_impl.forward = wrap_mixedlinear(layer.self_attn.k_proj.linear_impl, 'k')
        layer.self_attn.v_proj.linear_impl.forward = wrap_mixedlinear(layer.self_attn.v_proj.linear_impl, 'v')
        layer.self_attn.o_proj.linear_impl.forward = wrap_mixedlinear(layer.self_attn.o_proj.linear_impl, 'o')
        layer.mlp.up_proj.linear_impl.forward = wrap_mixedlinear(layer.mlp.up_proj.linear_impl, 'u')
        layer.mlp.gate_proj.linear_impl.forward = wrap_mixedlinear(layer.mlp.gate_proj.linear_impl, 'g')
        layer.mlp.down_proj.linear_impl.forward = wrap_mixedlinear(layer.mlp.down_proj.linear_impl, 'd')

        def wrap_linear(linear, name):
            def BitWeaver_forward(self, inp):
                if isinstance(inp, tuple):
                    assert self.rotate_type == 'up'
                    return self.linear_method(inp)
                scaled_input = inp / self.pretrain_scale
                if self.rotate_type == 'up':
                    return self.linear_method(scaled_input / self.norm_scale.unsqueeze(0))
                if self.rotate_type == 'down':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    start_event.record()
                    rotated = self.rotate.apply(scaled_input)
                    end_event.record()

                    all_start_events['hadamard'][name].append(start_event)
                    all_end_events['hadamard'][name].append(end_event)

                    return self.linear_method(rotated)
            return partial(BitWeaver_forward, linear)
        
        layer.self_attn.q_proj.forward = wrap_linear(layer.self_attn.q_proj, 'q')
        layer.self_attn.k_proj.forward = wrap_linear(layer.self_attn.k_proj, 'k')
        layer.self_attn.v_proj.forward = wrap_linear(layer.self_attn.v_proj, 'v')
        layer.self_attn.o_proj.forward = wrap_linear(layer.self_attn.o_proj, 'o')
        layer.mlp.up_proj.forward = wrap_linear(layer.mlp.up_proj, 'u')
        layer.mlp.gate_proj.forward = wrap_linear(layer.mlp.gate_proj, 'g')
        layer.mlp.down_proj.forward = wrap_linear(layer.mlp.down_proj, 'd')

    def get_callbacks(name):
        def clear():
            for key in all_start_events[name]:
                all_start_events[name][key].clear()
                all_end_events[name][key].clear()
        
        def stat():
            total_time = 0
            for key in all_start_events[name]:
                if verbose:
                    print(f"{name}-{key}: {len(all_start_events[name][key])} events recorded")
                assert(len(all_start_events[name][key]) == len(all_end_events[name][key]))
                for start, end in zip(all_start_events[name][key], all_end_events[name][key]):
                    total_time += start.elapsed_time(end)
            try:
                return total_time / len(all_start_events[name]['d'])
            except ZeroDivisionError:
                return 0
        
        return clear, stat

    return {'attn': (attn_clear, attn_stat)} | {key: get_callbacks(key) for key in ['gemm', 'quant', 'post_mul', 'hadamard']}

def bench_layer(layer, rotary_emb, batch, seqlen, hidden, dtype, baseline, multistream, verbose):
    input_tensor = torch.randn(batch, seqlen, hidden, device='cuda', dtype=dtype)
    position_ids = torch.arange(0, input_tensor.shape[1], device=input_tensor.device).unsqueeze(0)
    position_embeddings = rotary_emb(input_tensor, position_ids)
    with torch.inference_mode():
        repeat = 5
        warmup = 30
        runs = 10

        for _ in trange(warmup, desc="Warmup..."):
            layer(
                input_tensor,
                position_embeddings=position_embeddings
            )[0].mean().cpu()

        if baseline == 'quarot':
            CALL_BACKS = {}
        else:
            _original_event_constructor = torch.cuda.Event
            def func_external_event(*args, **kwargs):
                return _original_event_constructor(*args, **kwargs, external=True)
            torch.cuda.Event = func_external_event

            CALL_BACKS = inject(layer, baseline, multistream, verbose)
            print("Hooks injected.")

        for clear_func, _ in CALL_BACKS.values():
            clear_func()

        print("Start capturing cuda graph.")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            layer(
                input_tensor,
                position_embeddings=position_embeddings
            )
        print("Cuda graph captured, start benchmarking.")

        total_times = []
        collected_data = {key: [] for key in CALL_BACKS}

        for _ in trange(repeat, desc="Benchmarking..."):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for _ in range(warmup):
                graph.replay()

            start_event.record()
            for _ in range(runs):
                graph.replay()
            end_event.record()
        
            torch.cuda.synchronize()
            
            total_times.append(start_event.elapsed_time(end_event) / runs)

            for key in CALL_BACKS:
                collected_data[key].append(CALL_BACKS[key][1]()) # only records last iter

        print(f"Layer {sum(total_times) / repeat:.2f} ms", end='')
        for k, v in collected_data.items():
            print(f" | {k} {sum(v) / repeat:.2f} ms", end='')
        print("")


def main():
    # only need the following arguments to run this benchmark
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--nbatch', type=int, required=True)
    parser.add_argument('--seqlen', type=int, required=True)
    parser.add_argument("--baseline", type=str, required=True, choices=['bf16', 'bitweaver', 'quarot'])
    parser.add_argument('--multistream', action="store_true")
    parser.add_argument('--unifiedkernel', action="store_true")
    parser.add_argument("--bench-model", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    print(args)

    eval_args = eval_parse_args(['--model', args.model] + (['--multistream'] if args.multistream else []) + (['--unifiedkernel'] if args.unifiedkernel else []))

    if args.baseline == 'bf16':
        eval_args.a_bits = 16
        eval_args.w_bits = 16
    elif args.baseline == 'bitweaver':
        eval_args.a_bits = 4
        eval_args.w_bits = 4
        eval_args.rotate = 'hadamard'
        eval_args.mixed_precision = 'bitweaver'
        eval_args.threshold_policy = 'percentage'
        eval_args.activation_threshold = 0.05
        eval_args.weight_threshold = 0.05
        eval_args.qfactory_kernel = True
        eval_args.rotate_opt = True
    
    torch.manual_seed(eval_args.seed)
    transformers.set_seed(eval_args.seed)

    if args.baseline == 'quarot':
        assert args.bench_model == False
        config = transformers.AutoConfig.from_pretrained(args.model, torch_dtype=torch.float16)
        
        sys.path.insert(0, os.path.join(os.getcwd(), 'benchmark'))
        import modeling_qwen3_quarot

        import quarot
        import hadamard_utils
        quarot.functional.hadamard.get_hadK = hadamard_utils.get_hadK

        torch.set_default_dtype(torch.float16)
        with transformers.modeling_utils.no_init_weights():
            model = modeling_qwen3_quarot.QuarotQwen3ForCausalLM(config)
        model.eval()
        
        model_layer = model.model.layers[0].to('cuda')
        rotary_emb = model.model.rotary_emb.to('cuda')
                
        bench_layer(model_layer, rotary_emb, args.nbatch, args.seqlen, model.config.hidden_size, model.dtype, args.baseline, args.multistream, args.verbose)
        return

    tokenizer = transformers.AutoTokenizer.from_pretrained(eval_args.model)

    if args.bench_model:
        from eval import get_model
        model = get_model(eval_args, tokenizer)
        model = torch.compile(model)
        bench_model(model, tokenizer, args.nbatch, args.seqlen)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(eval_args.model, torch_dtype=eval_args.torch_dtype)
        model.eval()

        eval_args.cuda_count = 1
        model_layer = model.model.layers[0].to('cuda')
        rotary_emb = model.model.rotary_emb.to('cuda')

        from qlinear import RQLinear, MixedRQLinear

        for submodule_name in ['self_attn','mlp']:
            submodule = model_layer.__getattr__(submodule_name)
            for name, module in submodule.named_children():
                if not isinstance(module, torch.nn.Linear):
                    continue
                norm_scale = model_layer.input_layernorm.weight.data if 'attn' in submodule_name else model_layer.post_attention_layernorm.weight.data
                if eval_args.mixed_precision != 'none':
                    submodule.__setattr__(name, MixedRQLinear(eval_args, module, name, 0, norm_scale))
                else:
                    submodule.__setattr__(name, RQLinear(eval_args, module, name, 0, norm_scale))
        
        bench_layer(model_layer, rotary_emb, args.nbatch, args.seqlen, model.config.hidden_size, model.dtype, args.baseline, args.multistream, args.verbose)


if __name__ == "__main__":
    main()
