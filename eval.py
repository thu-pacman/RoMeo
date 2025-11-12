import os
import torch
import datasets
import transformers
import argparse
import lm_eval

from tqdm import tqdm, trange
from transformers.modeling_outputs import CausalLMOutputWithPast
from capture import inject_capture_layer, dump_captured_layer_input, dump_model_weights
from qlinear import RQLinear, MixedRQLinear
from smooth_q import smooth_quantize
from lm_eval.models.huggingface import HFLM
from utils import print_peak_memory
from rotate import Rotation

class DistRotatedModel(torch.nn.Module):
    def __init__(self, model, cuda_count: int, rotate: str, rotate_opt: bool):
        super().__init__()
        print(f'Using {cuda_count} CUDA devices.')
        self.cuda_count = cuda_count
        self.rotate_opt = rotate_opt
        self.distributed_device = tuple('cuda:' + str(i) for i in range(cuda_count))
        self.rotate_initial = Rotation.get_rotation(rotate, model.config.hidden_size, model.dtype, self.distributed_device[0])
        self.rotate_final = Rotation.get_rotation(rotate, model.config.hidden_size, model.dtype, self.distributed_device[-1])
        
        # for lm_eval
        self.model = torch.nn.Module()
        self.config = model.config
        self.tie_weights = model.tie_weights
        self.device = torch.device(self.distributed_device[0])

        self.model.embed_tokens = model.model.embed_tokens.to(self.distributed_device[0])
        self.model.rotary_emb = tuple(model.model.rotary_emb.to(self.distributed_device[i]) for i in range(cuda_count))
        self.model.layers = tuple(torch.nn.ModuleList(model.model.layers[len(model.model.layers) * i // cuda_count : len(model.model.layers) * (i + 1) // cuda_count].to(self.distributed_device[i])) for i in range(cuda_count))
        self.lm_head = model.lm_head.to(self.distributed_device[-1])
        self.model.norm = model.model.norm.to(self.distributed_device[-1])
    
    def forward(
        self,
        input_ids: torch.LongTensor,
    ) :
        inputs_embeds = self.model.embed_tokens(input_ids.to(self.distributed_device[0]))
        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds

        if self.rotate_opt:
            hidden_states = self.rotate_initial.apply(hidden_states)
        
        for i in range(self.cuda_count):
            hidden_states = hidden_states.to(self.distributed_device[i])
            position_embeddings = self.model.rotary_emb[i](hidden_states, position_ids.to(self.distributed_device[i]))
            for decoder_layer in self.model.layers[i]:
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                )
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
        
        if self.rotate_opt:
            hidden_states = self.rotate_final.apply_trans(hidden_states)

        hidden_states = self.model.norm(hidden_states)
        
        return CausalLMOutputWithPast(
            loss=None,
            logits=self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).to(self.distributed_device[0]),
        )

def get_model(args, tokenizer):
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=args.torch_dtype)
    model.eval()
    
    if args.cuda_count is None:
        args.cuda_count = torch.cuda.device_count()
    
    model = DistRotatedModel(model, args.cuda_count, args.rotate, args.rotate_opt)
    
    if args.smooth_quant:
        smooth_quantize(model, tokenizer, args.cuda_count, args.smooth_quant_alpha, args.smooth_quant_dataset)

    layer_id = 0
    with torch.inference_mode():
        for i in range(args.cuda_count):
            for decoder_layer in tqdm(model.model.layers[i], desc="Converting model"):
                submodules = [
                    'self_attn',
                    'mlp'
                ]
                
                for submodule_name in submodules:
                    submodule = decoder_layer.__getattr__(submodule_name)
                    for name, module in submodule.named_children():
                        if not isinstance(module, torch.nn.Linear):
                            continue
                        norm_scale = decoder_layer.input_layernorm.weight.data if 'attn' in submodule_name else decoder_layer.post_attention_layernorm.weight.data
                        if args.mixed_precision != 'none':
                            submodule.__setattr__(name, MixedRQLinear(args, module, name, layer_id, norm_scale))
                        else:
                            submodule.__setattr__(name, RQLinear(args, module, name, layer_id, norm_scale))
                layer_id += 1
    return model

def eval_zero_shot(model, tokenizer, dataset_name, batch_size):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = lm_eval.simple_evaluate(hflm, tasks=dataset_name, batch_size=f'auto:{batch_size}')['results']
    metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
    metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    print(metric_vals)

def eval_perplexity(model, tokenizer, eval_perplexity_dataset, capture_layer_ids, capture_layer_input, capture_layer_input_dir, capture_layer_weight_dir, batch_size, max_iters=-1):
    max_length = 2048
    if eval_perplexity_dataset == 'wikitext2':
        testdata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        input_ids = testenc.input_ids.flatten()
        nsamples = input_ids.numel() // max_length
        input_ids = input_ids[:nsamples * max_length].view(nsamples, max_length).to('cuda')
    else:
        raise ValueError(f"Unknown dataset {eval_perplexity_dataset}")
    
    if capture_layer_ids:
        capture_layer_ids = list(map(int, args.capture_layer_ids.split(',')))
        if capture_layer_input:
            model, captured_weights = inject_capture_layer(model, capture_layer_ids)
            batch = input_ids[:5]
            with torch.no_grad():
                logits = model(batch).logits.cpu()
            dump_captured_layer_input(captured_weights, capture_layer_input_dir)
            torch.save(logits, f"{capture_layer_input_dir}/logits.pt")
        if capture_layer_weight_dir:
            dump_model_weights(model, capture_layer_ids, capture_layer_weight_dir)
        return
    
    nlls = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    n_iters = (nsamples + batch_size - 1) // batch_size
    if max_iters != -1:
        n_iters = min(n_iters, max_iters)

    pbar = trange(n_iters, desc="PPL:")
    for i in pbar:
        batch = input_ids[i * batch_size : min(nsamples, (i + 1) * batch_size)]
        with torch.inference_mode():
            logits = model(batch).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        loss = loss_fn(shift_logits.permute(0, 2, 1), shift_labels)

        nlls.append(loss.float())
        current_ppl = torch.exp(loss.float().mean())
        pbar.set_description(f"PPL: {current_ppl.item()}")

    ppl = torch.exp(torch.cat(nlls).mean())
    print(f"Perplexity: {ppl.item()}")

def main(args):
    torch.manual_seed(args.seed)
    transformers.set_seed(args.seed)

    if args.profile_memory:
        torch.cuda.memory._record_memory_history()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = get_model(args, tokenizer)

    if args.profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_stack=True
        ) as prof:
            eval_perplexity(model, tokenizer, args.eval_perplexity_dataset, args.capture_layer_ids, args.capture_layer_input, args.capture_layer_input_dir, args.capture_layer_weight_dir, args.batch_size, max_iters=3)
        prof.export_chrome_trace(args.profile_trace)

        if args.profile_memory:
            torch.cuda.memory._dump_snapshot(args.profile_memory_snapshot)
            torch.cuda.memory._record_memory_history(enabled=None)
        return
    
    if args.profile_memory and not args.profile:
        eval_perplexity(model, tokenizer, args.eval_perplexity_dataset, args.capture_layer_ids, args.capture_layer_input, args.capture_layer_input_dir, args.capture_layer_weight_dir, args.batch_size, max_iters=3)
        torch.cuda.memory._dump_snapshot(args.profile_memory_snapshot)
        torch.cuda.memory._record_memory_history(enabled=None)
        return

    if args.eval_perplexity:
        eval_perplexity(model, tokenizer, args.eval_perplexity_dataset, args.capture_layer_ids, args.capture_layer_input, args.capture_layer_input_dir, args.capture_layer_weight_dir, args.batch_size)
    if args.eval_zero_shot:
        eval_zero_shot(model, tokenizer, args.eval_zero_shot_dataset.split(","), args.batch_size)
    if args.mixed_precision != 'none':
        MixedRQLinear.dump_counter()
        
def parse_args(sys_args = None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--torch-dtype', type=str, default='auto', choices=['auto', 'float16', 'bfloat16'])
    parser.add_argument('--cuda-count', type=int, default=None)
    
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--eval-perplexity', action='store_true', help='Calculate perplexity on the evaluation dataset')
    parser.add_argument('--eval-perplexity-dataset', type=str, choices=['wikitext2'], default='wikitext2')
    parser.add_argument('--eval-zero-shot', action='store_true', help='Run zero-shot evaluation on lm_eval tasks')
    parser.add_argument('--eval-zero-shot-dataset', type=str, default='piqa,winogrande,lambada_openai,arc_easy,arc_challenge')
    
    parser.add_argument('--a-bits', type=int, default=16)
    parser.add_argument('--w-bits', type=int, default=16)
    parser.add_argument('--w-clip', action='store_true', help='Enable searching for weight scale clipping')
    parser.add_argument('--rotate', type=str, default='none', choices=['none', 'hadamard', 'random'])
    parser.add_argument('--a-group', type=int, default=None)
    parser.add_argument('--w-group', type=int, default=None)
    
    parser.add_argument('--mixed-precision', type=str, default='none', choices=['none', 'bitweaver', 'mixq'])
    parser.add_argument('--threshold-policy', type=str, choices=['value', 'amount', 'percentage', 'percentage-noround'], default='value')
    parser.add_argument('--activation-threshold', type=float, default=0)
    parser.add_argument('--weight-threshold', type=float, default=0)

    parser.add_argument('--smooth-quant', action='store_true', default=False)
    parser.add_argument('--smooth-quant-alpha', type=float, default=0.5)
    parser.add_argument('--smooth-quant-dataset', type=str, default='./smooth_quant_dataset.json')

    parser.add_argument('--capture-layer-ids', type=str, default='', help='Comma separated list of layer ids to capture')
    parser.add_argument('--capture-layer-input', action='store_true')
    parser.add_argument('--capture-layer-input-dir', type=str, default='.dump/input')
    parser.add_argument('--capture-layer-weight', action='store_true')
    parser.add_argument('--capture-layer-weight-dir', type=str, default='.dump/weight')

    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    parser.add_argument('--profile-trace', type=str, default='trace.json', help='Path to the profiling trace file')
    parser.add_argument('--profile-memory', action='store_true', help='Enable memory profiling')
    parser.add_argument('--profile-memory-snapshot', type=str, default='memory.pickle', help='Path to the memory snapshot file')

    parser.add_argument('--qfactory-kernel', action='store_true', help='Use qfactory kernel for mixed precision')
    parser.add_argument('--rotate-opt', action='store_true', help='Use rotation optimization')
    parser.add_argument('--multistream', action='store_true', help='Use multistream')
    parser.add_argument('--unifiedkernel', action='store_true', help='Use unified kernel')

    return parser.parse_args(sys_args)

if __name__ == '__main__':
    args = parse_args()
    print(args)

    main(args)
    print_peak_memory()
