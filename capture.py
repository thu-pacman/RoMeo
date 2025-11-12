import os
import torch
from tqdm import tqdm

from qlinear import RLinear

def inject_capture_layer(model, layer_ids):
    print(model)
    captured_inputs = {}
    assert len(model.model.layers) == 1, "Only single GPU supported"
    for i, decoder_layer in enumerate(model.model.layers[0]):
        if i not in layer_ids:
            continue
        for name, module in decoder_layer.named_modules():
            if not isinstance(module, RLinear):
                continue
            if "k_proj" in name or "v_proj" in name:
                continue # q_proj == k_proj == v_proj
            if "gate_proj" in name:
                continue # gate_proj == up_proj

            def create_hook(layer_id, module_name): # closure for freezing name
                def hook(module, inputs, outputs):
                    captured_inputs[f"layer{layer_id}.{module_name}"] = module.dump_mm_input(inputs[0])
                return hook
            
            module.register_forward_hook(create_hook(i, name))
    
    def create_hook(module_name): # closure for freezing name
        def hook(module, inputs, outputs):
            captured_inputs[f"{module_name}"] = inputs[0]
        return hook
    
    model.lm_head.register_forward_hook(create_hook("lm_head"))

    return model, captured_inputs

def dump_captured_layer_input(captured_inputs, dump_dir):
    os.makedirs(dump_dir, exist_ok=True)
    for name, inp in tqdm(captured_inputs.items(), desc="Saving tensors"):
        torch.save(inp, f"{dump_dir}/{name}.pt")
    print(f"Saved captured {len(captured_inputs)} inputs to {dump_dir}")

def dump_model_weights(model, layer_ids, dump_dir):
    print(model)
    os.makedirs(dump_dir, exist_ok=True)
    captured_weights = {}
    for i, decoder_layer in enumerate(model.model.layers):
        if i not in layer_ids:
            continue
        for name, module in decoder_layer.named_modules():
            if not isinstance(module, RLinear):
                continue
            captured_weights[f"layer{i}.{name}"] = module.dump_mm_weight()
    for name, weight in tqdm(captured_weights.items(), desc="Saving tensors"):
        torch.save(weight, f"{dump_dir}/{name}.pt")
    print(f"Saved captured {len(captured_weights)} weights to {dump_dir}")
