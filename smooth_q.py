import torch
from tqdm import trange
from datasets import load_dataset

pretrain_matrices = []

class PretrainQuantizedLinear(torch.nn.Linear):
    def __init__(self, original_layer, smooth_quant_alpha):
        super().__init__(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            bias=original_layer.bias is not None,
            device=original_layer.weight.device,
            dtype=original_layer.weight.dtype,
        )
        self.weight = original_layer.weight
        self.bias = original_layer.bias

        self.alpha = smooth_quant_alpha
        self.max_val = torch.full((self.in_features,), 1e-5, dtype=torch.float32, device=original_layer.weight.device)
        self.max_weight = torch.full((self.in_features,), 1e-5, dtype=torch.float32, device=original_layer.weight.device)
        self.finished = False

        del original_layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.finished:
            self.max_val = torch.max(self.max_val, torch.max(x.abs(), dim=1)[0])
            self.max_weight = torch.max(self.max_weight, torch.max(self.weight.abs(), dim=0)[0])
        return torch.nn.functional.linear(x, self.weight, self.bias)
    
    def finish(self):
        self.pretrain_scale = torch.clip(self.max_val ** self.alpha / self.max_weight ** (1 - self.alpha), min=1e-5).to(self.weight.dtype)
        self.finished = True
        del self.max_val
        del self.max_weight

def pretrain_quantize_module(model, cuda_count, smooth_quant_alpha):
    for i in range(cuda_count):
        for decoder_layer in model.model.layers[i]:
            submodules = [
                'self_attn',
                'mlp'
            ]
            for submodule_name in submodules:
                submodule = decoder_layer.__getattr__(submodule_name)
                for name, module in submodule.named_children():
                    if not isinstance(module, torch.nn.Linear):
                        continue
                    submodule.__setattr__(name, PretrainQuantizedLinear(module, smooth_quant_alpha))
                    pretrain_matrices.append(submodule.__getattr__(name))

def smooth_quantize(model, tokenizer, cuda_count, smooth_quant_alpha, smooth_quant_dataset, num_samples=512, seq_len=512):
    if not 0 <= smooth_quant_alpha <= 1:
        raise ValueError("smooth_quant_alpha must be between 0 and 1.")
    
    print("Smooth quantization started.")
    pretrain_quantize_module(model, cuda_count, smooth_quant_alpha)

    dataset = load_dataset("json", data_files=smooth_quant_dataset, split="train")
    dataset = dataset.shuffle(seed=42)
    if dataset[0]['meta'] == 'default smooth quant dataset':
        print("Using builtin default smooth quant dataset.")
    
    with torch.inference_mode():
        for i in trange(min(num_samples, len(dataset)), desc="SQ Calibration"):
            pretrain_input_ids = tokenizer(dataset[i]["text"], return_tensors='pt', max_length=seq_len, truncation=True)['input_ids']
            model(input_ids=pretrain_input_ids)
    
    for i in pretrain_matrices:
        i.finish()
    
    print("Smooth quantization finished.")
