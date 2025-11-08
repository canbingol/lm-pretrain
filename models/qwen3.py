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
    embed_dim: int = 1024
    n_heads: int = 128
    n_layers: int = 12
    d_ff: int = 512 * 4
    batch_size: int = 29
    max_steps: int = 200_000

    # Qwen3-like parameters
    n_kv_heads: int = 64 # For Grouped-Query Attention
    sliding_window: int = 1024  # Set a large default, effectively disabling it unless specified
    attention_bias: bool = False  # Qwen3 often sets this to False
    rms_norm_eps: float = 1e-06  # Epsilon for RMSNorm
    qk_norm = True

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01

    # Data parameters
    max_seq_len: int = 512
    max_tokens: int = 50000000
    vocab_size = 32768
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.0
    grad_clip: float = 1.0


    data_dtype = np.uint16
    # Technical
    use_amp: bool = True
    drop_last = True
    shuffle = False
    def __post_init__(self):
        self.d_k = self.embed_dim // self.n_heads
        assert self.embed_dim % self.n_heads == 0, "embed_dim must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_kv_groups = self.n_heads // self.n_kv_heads


class SwiGLUFeedForward(nn.Module):
    def __init__(self, embed_dim: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, embed_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, d_ff, bias=False)
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

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        cos, sin = self.precompute_cis(head_dim, max_seq_len, base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def precompute_cis(self, head_dim: int, max_seq_len: int, base: float):
        """cos/sin tablolarını hesapla"""
        half = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
        t = torch.arange(max_seq_len, dtype=torch.float32)
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
        self.embed_dim = config.embed_dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = config.n_kv_groups
        self.d_k = config.d_k
        self.qk_norm = config.qk_norm

        # Separate linear layers for Q, K, V
        self.q_proj = nn.Linear(self.embed_dim, self.n_heads * self.d_k, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.n_kv_heads * self.d_k, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.n_kv_heads * self.d_k, bias=config.attention_bias)
        self.w_o = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # QK-Normalization layers
        self.q_norm = nn.RMSNorm(self.d_k, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.d_k, eps=config.rms_norm_eps)

        self.rotary = Rotary(self.d_k, config.max_seq_len)
        self.dropout = config.dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        # 1. Project Q, K, V separately
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape into heads
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.d_k)

        if self.qk_norm:
            # 3. Apply QK-Norm
            q = self.q_norm(q)
            k = self.k_norm(k)


        # 4. Apply RoPE
        # Transpose to (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k) for rotary
        q = self.rotary(q.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        k = self.rotary(k.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        # Transpose for attention: (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
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
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.w_o(attn_output)
    
class Qwen3Decoder(nn.Module):
    def __init__(self, config):  # Pass the entire config object
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = SwiGLUFeedForward(config.embed_dim, config.d_ff, config.dropout)
        self.norm1 = nn.RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x


class Qwen3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_dropout = nn.Dropout(config.dropout)

        self.transformer_blocks = nn.ModuleList([
            Qwen3Decoder(config) for _ in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.output_dropout = nn.Dropout(config.dropout)

        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.embed_dim)
        x = self.position_dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits