import os
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from flash_attn import flash_attn_func


@dataclass
class Qwen3Config:
    config_name: str = "Qwen3"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 512
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    max_position_embeddings: int = 40960
    max_window_layers: int = 28
    model_type: str = "qwen3"
    num_attention_heads: int = 16
    num_hidden_layers: int = 1
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-6
    rope_scaling: None = None
    rope_theta: int = 1000000
    sliding_window: None = None
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 50176


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y


class Attention(nn.Module):
    def __init__(
        self,
        scale: float,
        dropout_p: float = 0.0, 
    ):
        super().__init__()
        self.scale = scale
        self.dropout_p = dropout_p

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0, 
            softmax_scale=self.scale,
            causal=True 
        )
        return o
    

class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        total_out_dim = self.q_size + self.kv_size * 2

        self.qkv_proj = nn.Linear(self.hidden_size, total_out_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim , self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
                    self.head_dim, 
                    rotary_dim=self.head_dim, 
                    max_position_embeddings=config.max_position_embeddings,
                    base=config.rope_theta
                )
        
        self.attn = Attention(
            scale=self.head_dim**-0.5
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)


    def forward(self, positions,
                hidden_states,
                past_key_value:tuple=None,
                use_cache:bool=False):
            
            qkv = self.qkv_proj(hidden_states)
            
            # Split Q, K, V
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            
            # Reshape: [Batch, SeqLen, NumHeads, HeadDim]
            q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim)
            k = k.view(k.shape[0], k.shape[1], self.num_kv_heads, self.head_dim)
            v = v.view(v.shape[0], v.shape[1], self.num_kv_heads, self.head_dim)
            
            q = self.q_norm(q)
            k = self.k_norm(k)
            
            q, k = self.rotary_emb(positions, q, k)
            
            current_key_value = None
            if use_cache:

                if past_key_value is not None:
                    past_k, past_v = past_key_value
                    k = torch.cat((past_k, k), dim=1)
                    v = torch.cat((past_v, v), dim=1)

                current_key_value = (k, v)

            attn_output = self.attn(q, k, v)
            
            attn_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[1], -1)
            output = self.o_proj(attn_output)
            
            return output, current_key_value

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):

        var = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(var + self.eps)
        out = norm_x * self.weight
        return norm_x * self.weight


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(
            hidden_size,
            intermediate_size * 2,
            bias=False,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config=config)
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        past_kv_cache: tuple =None,
        use_cache:bool =False

    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key_value = self.self_attn(positions, hidden_states, past_kv_cache, use_cache)

        hidden_states = residual + attn_output

        residual = hidden_states
        
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        
        return hidden_states, present_key_value


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        past_key_values: tuple= None,
        use_cache:bool= False
    ) -> torch.Tensor:
        
        hidden_states = self.embed_tokens(input_ids)
        
        next_decoder_cache = () if use_cache else None
        
        for idx, layer in enumerate(self.layers):
            past_kv_cache =  past_key_values[idx] if past_key_values is not None else None 

            assert hidden_states.dim() == 3,f"hidden_states dim has to be 3 but {hidden_states.dim()}"
            
            hidden_states, layer_kv  = layer(
                positions=positions, 
                hidden_states=hidden_states, 
                past_kv_cache=past_kv_cache, 
                use_cache=use_cache
                )
            
            if use_cache:
                next_decoder_cache += (layer_kv, )

            hidden_states = self.norm(hidden_states)

        if use_cache:
            return hidden_states, next_decoder_cache
        
        return hidden_states


class Qwen3CausalLM(nn.Module):
    
    def __init__(self,config: Qwen3Config):
        super().__init__()
        self.config = config
        self.base = Qwen3Model(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def get_input_embeddings(self):
        return self.base.embed_tokens
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        positions: torch.Tensor = None,
        past_key_values: torch.Tensor=None,
        use_cache:bool=False
    ):
        batch_size, seq_len = input_ids.shape
        hidden_states = None
        if not use_cache:
            positions = torch.arange(seq_len,  device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)

        outputs = self.base(input_ids=input_ids, positions=positions, past_key_values=past_key_values, use_cache=use_cache)

        if use_cache:
            hidden_states, next_cache = outputs
        else:
            hidden_states = outputs
            next_cache = None

        logits = self.lm_head(hidden_states)
        if use_cache:
            return logits, next_cache
            
        return logits
    
    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 20, 
        temperature: float = 1.0, 
        top_k: int = None,
        use_cache = True

    ):
        self.eval()

        past_key_values = None
        generated_ids = input_ids

        for _ in range(max_new_tokens):
            seq_len = input_ids.size(1)

            # Prefill
            if past_key_values is None:
                positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

            # Decode
            else:
                current_pos = generated_ids.shape[1] - 1
                positions = torch.tensor([[current_pos]], device=input_ids.device)
            
            model_input = input_ids if past_key_values is None else input_ids[:, -1:]

            # Forward pass
            outputs = self.base(input_ids=model_input, 
                          positions=positions, 
                          past_key_values=past_key_values,
                          use_cache=use_cache)

            hidden_states, past_key_values = outputs # Update cache

            logits = self.lm_head(hidden_states)

            logits = logits[:,-1,:]

            # Temperature
            if temperature:
                logits = logits / temperature

            # Top-K
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            input_ids = next_token
            
            if next_token.item() ==  self.config.eos_token_id:
                break

        self.train()
        return generated_ids
    