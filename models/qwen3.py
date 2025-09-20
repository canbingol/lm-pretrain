import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
import numpy as np

@dataclass
class Qwen3Config:
    # Model architecture
    config_name = "QwenConfig"
    hidden_size: int = 512
    num_attention_heads: int = 128
    num_hidden_layers: int = 20
    intermediate_size: int = 1024
    batch_size: int = 29

    # Qwen3-like parameters
    num_key_value_heads: int = 64 # For Grouped-Query Attention
    sliding_window: int = None  # Set a large default, effectively disabling it unless specified
    attention_bias: bool = False  # Qwen3 often sets this to False
    rms_norm_eps: float = 1e-06  # Epsilon for RMSNorm
    qk_norm = True

    # Data parameters
    max_position_embeddings: int = 128
    vocab_size = 32768
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Regularization
    dropout: float = 0.0

    # Technical
    drop_last = True
    shuffle = False
    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.hidden_size % self.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        assert self.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"
        self.n_kv_groups = self.num_attention_heads // self.num_key_value_heads


class SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        activated_x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.dropout(activated_x))
    
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape

    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)

    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Rotary(nn.Module):
    """
    Rotary Positional Embedding (RoPE)
    - precompute_cis: prepares the cos/sin lookup tables
    - apply_rope: applies RoPE transformation to the input tensor
    """

    def __init__(self, head_dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        cos, sin = self.precompute_cis(head_dim, max_position_embeddings, base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def precompute_cis(self, head_dim: int, max_position_embeddings: int, base: float):
        """cos/sin tablolarını hesapla"""
        half = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (S, half)
        return freqs.cos(), freqs.sin()

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """x üzerine RoPE uygula"""
        D = x.size(-1); half = D // 2; S = x.size(-2)
        if x.dim() == 4:   # (B,H,S,D)
            cos = cos[:S, :half].view(1, 1, S, half)
            sin = sin[:S, :half].view(1, 1, S, half)
        elif x.dim() == 3: # (B,S,D)
            cos = cos[:S, :half].view(1, S, half)
            sin = sin[:S, :half].view(1, S, half)
        else:
            raise ValueError("x must be (B,H,S,D) or (B,S,D)")

        x_f32 = x.to(torch.float32)
        xe, xo = x_f32[..., ::2], x_f32[..., 1::2]
        ye = xe * cos - xo * sin
        yo = xe * sin + xo * cos
        out = torch.empty_like(x_f32)
        out[..., ::2], out[..., 1::2] = ye, yo
        return out.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_rope(x, self.cos, self.sin)

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.n_kv_groups = config.n_kv_groups
        self.head_dim = config.head_dim
        self.qk_norm = config.qk_norm

        # Separate linear layers for Q, K, V
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # QK-Normalization layers
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary = Rotary(self.head_dim, config.max_position_embeddings)
        self.dropout = config.dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        # 1. Project Q, K, V separately
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape into heads
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        if self.qk_norm:
            # 3. Apply QK-Norm
            q = self.q_norm(q)
            k = self.k_norm(k)


        # 4. Apply RoPE
        # Transpose to (batch, seq_len, num_attention_heads, head_dim) -> (batch, num_attention_heads, seq_len, head_dim) for rotary
        q = self.rotary(q.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        k = self.rotary(k.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        # Transpose for attention: (batch, seq_len, num_attention_heads, head_dim) -> (batch, num_attention_heads, seq_len, head_dim)
        Q = q.transpose(1, 2)
        K = k.transpose(1, 2)
        V = v.transpose(1, 2)

        # 5. Repeat K and V heads for GQA
        K = repeat_kv(K, self.n_kv_groups)
        V = repeat_kv(V, self.n_kv_groups)

        # 6. Scaled Dot-Product Attention
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        # 7. Reshape and final projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)
    
class Qwen3Decoder(nn.Module):
    def __init__(self, config):  # Pass the entire config object
        super().__init__()
        self.self_attn = GroupedQueryAttention(config)
        self.mlp = SwiGLUFeedForward(config.hidden_size, config.intermediate_size, config.dropout)
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        attn_out = self.self_attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.mlp(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x


class Qwen3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = "qwen3"
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            Qwen3Decoder(config) for _ in range(config.num_hidden_layers)
        ])

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_dropout = nn.Dropout(config.dropout)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight


        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.embed_tokens(x) * math.sqrt(self.config.hidden_size)
        x = self.position_dropout(x)

        for block in self.layers:
            x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits