import torch
import quarot
import logging
from typing import Optional, Tuple, Callable

from transformers import Cache
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3ForCausalLM, apply_rotary_pos_emb, Qwen3MLP, eager_attention_forward

logger = logging.getLogger(__name__)

ALL_LAYERNORM_LAYERS.append(quarot.nn.RMSNorm)

class QuarotFP16Qwen3Attention(Qwen3Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.quantizer = torch.nn.Identity()
        self.o_proj_hadamard = torch.nn.Identity()
        
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.hidden_size = config.hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        hidden_states = self.quantizer(hidden_states)
        
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        assert past_key_value is None # Disable KV Cache

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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

        attn_output = self.o_proj_hadamard(attn_output.transpose(-1, -2)).transpose(-1, -2)
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights

class QuarotQwen3Attention(QuarotFP16Qwen3Attention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = quarot.nn.Quantizer()
        self.q_proj = quarot.nn.Linear4bit.from_float(self.q_proj)
        self.k_proj = quarot.nn.Linear4bit.from_float(self.k_proj)
        self.v_proj = quarot.nn.Linear4bit.from_float(self.v_proj)
        self.o_proj_hadamard = quarot.nn.OnlineHadamard(self.num_heads)
        self.o_proj = torch.nn.Sequential(
            quarot.nn.Quantizer(),
            quarot.nn.Linear4bit.from_float(self.o_proj)
        )

class QuarotQwen3MLP(Qwen3MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = quarot.nn.Quantizer()
        self.up_proj = quarot.nn.Linear4bit.from_float(self.up_proj)
        self.gate_proj = quarot.nn.Linear4bit.from_float(self.gate_proj)
        self.down_proj = torch.nn.Sequential(
            quarot.nn.OnlineHadamard(self.intermediate_size),
            quarot.nn.Quantizer(),
            quarot.nn.Linear4bit.from_float(self.down_proj)
        )

    def forward(self, x):
        x = self.quantizer(x)
        return super().forward(x)


class QuarotFP16Qwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = QuarotFP16Qwen3Attention(config=config, layer_idx=layer_idx)

    def forward(self, input_ids, *args, **kwargs):
        out = super().forward(input_ids, *args, **kwargs)
        return out

class QuarotQwen3ForCausalLM(QuarotFP16Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.norm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = QuarotQwen3Attention(config=config, layer_idx=layer_idx)
            layer.input_layernorm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.post_attention_layernorm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.mlp = QuarotQwen3MLP(config=config)
