from typing import Optional
from transformers.models.llama.modeling_llama import (
    LlamaConfig, 
    LlamaModel,
    LlamaForCausalLM,
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    LlamaMLP,
    LlamaDecoderLayer,
)
from mybitnet.bitnet import BitLinear
from torch import nn

class BitLlamaConfig(LlamaConfig):
    model_type = "bit_llama"

    def __init__(self, bits=8, **kwargs):
        super().__init__(**kwargs)
        self.bits = bits

class BitLlamaMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.gate_proj = BitLinear(self.hidden_size, self.intermediate_size, bias=False, bits=config.bits, flg_before_linear=True)
        self.up_proj = BitLinear(self.hidden_size, self.intermediate_size, bias=False, bits=config.bits, flg_before_linear=True)
        self.down_proj = BitLinear(self.intermediate_size, self.hidden_size, bias=False, bits=config.bits, flg_before_linear=False)
        
class BitLlamaAttention(LlamaAttention):
    def __init__(self, config: BitLlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config)
        self.q_proj = BitLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False, bits=config.bits, flg_before_linear=True)
        self.k_proj = BitLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, bits=config.bits, flg_before_linear=True)
        self.v_proj = BitLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, bits=config.bits, flg_before_linear=True)
        self.o_proj = BitLinear(self.hidden_size, self.hidden_size, bias=False, bits=config.bits, flg_before_linear=True)

class BitLlamaFlashAttention2(LlamaFlashAttention2):
    def __init__(self, config: BitLlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.q_proj = BitLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False, bits=config.bits, flg_before_linear=True)
        self.k_proj = BitLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, bits=config.bits, flg_before_linear=True)
        self.v_proj = BitLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, bits=config.bits, flg_before_linear=True)
        self.o_proj = BitLinear(self.hidden_size, self.hidden_size, bias=False, bits=config.bits, flg_before_linear=True)

class BitLlamaSdpaAttention(LlamaSdpaAttention):
    def __init__(self, config: BitLlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.q_proj = BitLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False, bits=config.bits, flg_before_linear=True)
        self.k_proj = BitLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, bits=config.bits, flg_before_linear=True)
        self.v_proj = BitLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, bits=config.bits, flg_before_linear=True)
        self.o_proj = BitLinear(self.hidden_size, self.hidden_size, bias=False, bits=config.bits, flg_before_linear=True)

BITLLAMA_ATTENTION_CLASSES = {
    "eager": BitLlamaAttention,
    "flash_attention_2": BitLlamaFlashAttention2,
    "sdpa": BitLlamaSdpaAttention,
}

class BitLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: BitLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = BITLLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.mlp = BitLlamaMLP(config)

class BitLlamaModel(LlamaModel):
    def __init__(self, config: BitLlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [BitLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

class BitLlamaForCausalLM(LlamaForCausalLM):
    config_class = BitLlamaConfig

    def __init__(self, config: BitLlamaConfig):
        super().__init__(config)
        self.model = BitLlamaModel(config)
        self.lm_head = BitLinear(config.hidden_size, config.vocab_size, bias=False, bits=config.bits, flg_before_linear=True)
        