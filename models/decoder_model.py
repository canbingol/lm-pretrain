import os
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, login

try:
    from flash_attn import flash_attn_func
except Exception:
    print("flash_attn package is not installed, flash attention is not available.")
    flash_attn_func = None

@dataclass
class ModelConfig:
    
    config_name: str = "decoder_only"
    model_type: str = "decoder"

    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645

    intermediate_size: int = 3072
    hidden_act: str = "silu"

    head_dim: int = 32
    hidden_size: int = 512
    num_attention_heads: int = 16
    num_hidden_layers: int = 1
    num_key_value_heads: int = 8
    vocab_size: int = 50176
    max_position_embeddings: int = 40960    

    rms_norm_eps: float = 1e-6
    rope_scaling: None = None
    rope_theta: int = 1000000

    use_cache: bool = True
    attn_type = "eager"
    torch_dtype = "bfloat16"

def repeat_kv(x: torch.tensor, n_rep: int) -> torch.tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    
    return(
        # (batch_size, seq_len, n_kv_heads, 1, head_dim)
        x[:, :, :, None, :]

        # (batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .expand(batch_size, seq_len,n_kv_heads,n_rep,head_dim)

        #! (batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        .reshape(batch_size, seq_len,n_kv_heads * n_rep,head_dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        scale: float,
        attn_type: str,
        is_causal: bool,
        dropout_p: float = 0.0, 
    ):
        super().__init__()
        self.scale = scale
        self.dropout_p = dropout_p
        self.attn_type = attn_type
        self.is_causal = is_causal

    def _eager_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal_mask: torch.Tensor, dropout_p: float):
        
        scores = q @ k.transpose(-2, -1) * self.scale

        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weight = torch.softmax(scores, dim=-1)

        if not torch.isfinite(attn_weight).all():
            raise RuntimeError("Non-Finite values in attention weights")
        
        if (attn_weight < 0).any():
            raise RuntimeError("Negative values in attention weights")

        attn_weight = F.dropout(attn_weight, dropout_p, training=self.training,)
        out = attn_weight @ v

        if not torch.isfinite(out).all():
            raise RuntimeError("Non-finite values in attention output")

        out_max = out.abs().max().item()
        if out_max > 100:
            print(f"Warning: attention output magnitude too large: {out_max}")

        return out

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal_mask: torch.Tensor | None):
        dropout_p = self.dropout_p if self.training else 0.0

        if self.attn_type == "flash_attn":
            assert flash_attn_func is not None, "For using flash attention install flash_attn (pip install flash_attn)"
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=dropout_p, 
                softmax_scale=self.scale,
                causal=self.is_causal 
            )

            attn_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[1], -1)

        elif self.attn_type == "sdpa":
            attn_output = F.scaled_dot_product_attention(q,k,v, is_causal=self.is_causal, dropout_p=dropout_p)
            attn_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[2], -1)

        elif self.attn_type == "eager":
            attn_output = self._eager_attn(q,k,v, causal_mask, dropout_p)
            attn_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[2], -1)

        else:
            raise ValueError(
                f"Unknown attn_type: {self.attn_type}. "
                "Expected one of ['flash_attn', 'sdpa', 'eager']."
            )

        return attn_output

class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_type = config.attn_type
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        max_seq_len = config.max_position_embeddings

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        total_out_dim = self.q_size + self.kv_size * 2

        self.qkv_proj = nn.Linear(self.hidden_size, total_out_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim , self.hidden_size, bias=False)

        self.is_causal = True
        self.rotary_emb = RotaryEmbedding(
                    self.head_dim, 
                    rotary_dim=self.head_dim, 
                    max_position_embeddings=config.max_position_embeddings,
                    base=config.rope_theta
                )
        
        self.attn = Attention(
            scale=self.head_dim**-0.5,
            attn_type=self.attn_type,
            is_causal=self.is_causal,
            dropout_p=self.attention_dropout,
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if max_seq_len is not None:
            m = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
            self.register_buffer("_causal_mask", m, persistent=False)
        else:
            self._causal_mask = None

    def _get_causal_mask(self, Lq: int, Lk: int, device: torch.device) -> torch.Tensor:
        if not self.is_causal:
            return None
        
        if self._causal_mask is not None and Lq == Lk:
            return self._causal_mask[:Lq, :Lq].to(device)
        
        if Lq == 1:
            return torch.zeros((1, Lk), dtype=torch.bool, device=device)

        
        mask = torch.triu(torch.ones(Lq, Lk, device=Lq.device), diagonal=1).bool()
        return mask
    
            
    def _check_values(self, q, k, v, mask):
        for name, x in [("q", q), ("k", k), ("v", v)]:
            if not torch.isfinite(x).all():
                raise RuntimeError(f"Non-finite values detected in {name}")

            max_val = x.abs().max().item()
            if max_val > 50:
                print(f"Warning: {name} abs max too large: {max_val}")

        if mask is not None:
            row_all_masked = mask.all(dim=-1)
            if row_all_masked.any():
                raise RuntimeError("At least one query has all keys masked")


    def forward(self, positions,
                hidden_states,
                past_key_value:tuple=None,
                use_cache:bool=False):
            
            qkv = self.qkv_proj(hidden_states)
            
            # Split Q, K, V
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            lq, lk = q.shape[1], k.shape[1]


            # Reshape: [Batch, SeqLen, NumHeads, HeadDim]
            q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim)
            k = k.view(k.shape[0], k.shape[1], self.num_kv_heads, self.head_dim)
            v = v.view(v.shape[0], v.shape[1], self.num_kv_heads, self.head_dim)

            q, k = self.rotary_emb(positions, q, k)
            
            n_rep = self.num_heads // self.num_kv_heads

            k = repeat_kv(x=k, n_rep=n_rep)
            v = repeat_kv(x=v, n_rep=n_rep)

            if self.attn_type == "sdpa" or self.attn_type == "eager":
                q = q.transpose(1,2)
                k = k.transpose(1,2)
                v = v.transpose(1,2)

            q = self.q_norm(q)
            k = self.k_norm(k)
            
            current_key_value = None
            if use_cache:
                if past_key_value is not None:
                    past_k, past_v = past_key_value

                    dim = 1 if self.attn_type == "flash_attn" else 2
                    k = torch.cat((past_k, k), dim=dim)
                    v = torch.cat((past_v, v), dim=dim)

                current_key_value = (k, v)


            assert self.num_heads % self.num_kv_heads == 0


            mask = self._get_causal_mask(lq, lk, device=q.device)
            
            self._check_values(q, k, v, mask)
            attn_output = self.attn(q, k, v, mask)
            
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
        return norm_x * self.weight

class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

class MLP(nn.Module):

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


class DecoderLayer(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.self_attn = SelfAttention(config=config)
        self.mlp = MLP(
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


class DecoderModel(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
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


class DecoderCausalLM(nn.Module,
                      PyTorchModelHubMixin):
    
    def __init__(self,config: ModelConfig):
        super().__init__()
        self.config = config
        self.name = config.config_name
        self.base = DecoderModel(config=config)
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
            # No history exists yet. Generate positions for the entire prompt (e.g., 0, 1, 2...).
            if past_key_values is None:
                positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

            # Decode
            # History exists. We need to manually tell the model the position index 
            # for the SINGLE new token being processed.
            else:
                current_pos = generated_ids.shape[1] - 1 # Total length - 1 = Index of the new token
                positions = torch.tensor([[current_pos]], device=input_ids.device)

            # INPUT PREPARATION:
            # If cache exists, feed ONLY the last token (newly generated one) to save compute.
            # If no cache (first run), feed the whole prompt.
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
    