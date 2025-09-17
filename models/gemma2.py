import dataclasses
from typing import List, Tuple,Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclasses.dataclass
class GemmaConfig:
    config_name: str = "GemmaConfig"
    device: str = "cuda"
    # The architecture of the model.
    architecture = 1
    batch_size:int = 5
    training: bool = True
    # The number of tokens in the vocabulary.
    vocab_size: int = 32_768
    # The maximum sequence length that this model might ever be used with.
    max_position_embeddings: int = 256
    # The number of blocks in the model.
    num_hidden_layers: int = 8
    # The number of attention heads used in the attention layers of the model.
    num_attention_heads: int = 16
    # The number of key-value heads for implementing attention.
    num_key_value_heads: int = 16
    # The hidden size of the model.
    hidden_size: int = 128
    # The dimension of the MLP representations.
    intermediate_size: int = 256
    # The number of head dimensions.
    head_dim: int = 8
    # The epsilon used by the rms normalization layers.
    rms_norm_eps: float = 1e-6
    # The dtype of the weights.
    dtype: str = 'float32'
    # Whether a quantized version of the model is used.
    quant: bool = False
    # The path to the model tokenizer.
    tokenizer: Optional[str] = 'tokenizer/tokenizer.model'
    # The types of attention used in the layers of the model.
    # The size of the sliding window used for local attention.
    sliding_window_size: Optional[int] = None
    # If provided, the final logits are softcapped to this value.
    final_logit_softcapping: Optional[float] = None
    # If provided, the attention logits are softcapped to this value.
    attn_logit_softcapping: Optional[float] = None
    # If provided, the query vector is normalized using the
    # inverse square root of this value instead of head_dim.
    query_pre_attn_scalar: Optional[int] = None
    # Whether to use pre mlp normalization.
    use_pre_ffw_norm: bool = False
    # Whether to use post mlp normalization.
    use_post_ffw_norm: bool = False


def precompute_theta_freqs(seq_len, head_dim,device: str,theta: float= 10000.0):

    #! theta_i = 10000.0^(-2(x_i-1)/d) , x_i = [1,2,3, .. .... , d/2]

    # here we calculate 2(x_i-1) this part
    theta_numerator = torch.arange(0,head_dim,2).float()
    # 10000.0 ^ -2(theta / head_dim)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    m = torch.arange(seq_len, device=device)

    freqs = torch.outer(m,theta).float()

    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

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

def apply_rope(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # x: (batch, seq_len, n_heads, head_dim)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # (B, S, H, D//2)
    
    # freqs_complex: (seq_len, head_dim//2) → reshape for broadcasting
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D//2)
    freqs_complex = freqs_complex.to(x_complex.device)

    assert freqs_complex.shape[1] == x_complex.shape[1], f"Seq len mismatch: {freqs_complex.shape[1]} vs {x_complex.shape[1]}"
    assert freqs_complex.shape[-1] == x_complex.shape[-1], f"Head dim mismatch: {freqs_complex.shape[-1]} vs {x_complex.shape[-1]}"
    x_rotated = x_complex * freqs_complex  # (B, S, H, D//2)
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)  # (B, S, H, D)
    return x_out.type_as(x).to(device)


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):

        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        # Learnable parameter for scaling the normalized values
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):

        # Compute RMS norm: x / sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):

        # Normalize the input tensor to float32 for stability
        output = self._norm(x.float())
        
        # Scale the normalized tensor using the learnable weight
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        
        # Cast the output back to the input tensor's data type
        return output.type_as(x)


class GemmaMLP(nn.Module):

    def __init__(
        self,
        config:GemmaConfig,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)

        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)

        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):

        # Apply the gating projection and activation function
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")  # GELU activation for gating

        # Apply the up projection
        up = self.up_proj(x)

        # Fuse the gate and up projections element-wise
        fuse = gate * up

        # Project the fused result back to the original hidden size
        outputs = self.down_proj(fuse)

        return outputs
    
def apply_causal_mask(scores: torch.tensor, seq_len: int):

    mask = torch.triu(torch.ones(seq_len,seq_len), diagonal=1)
    mask = mask.to(scores.device)

    mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))

    return scores

class GemmaAttention(nn.Module):
    """
    Implements multi-head attention with support for rotary embeddings, sliding window masking,
    and key-value caching.

    The class handles:
    - Query, Key, Value (QKV) projection.
    - Applying rotary positional embeddings.
    - Key-value caching for efficient sequential processing.
    - Scaled dot-product attention with optional sliding window masking.
    """

    def __init__(
        self,
        config:GemmaConfig,
    ):
        """
        Initializes the GemmaAttention module.

        Args:
            hidden_size (int): Input tensor dimension.
            num_heads (int): Number of attention heads.
            num_kv_heads (int): Number of key-value attention heads.
            attn_logit_softcapping (Optional[float]): Softcapping value for attention logits.
            query_pre_attn_scalar (Optional[int]): Pre-scaling factor for queries.
            head_dim (int): Dimension of each attention head.
            attn_type (gemma_config.AttentionType): Type of attention (e.g., local sliding).
            sliding_window_size (Optional[int]): Size of the sliding window for local attention.
        """
        super().__init__()
        
        self.training = config.training
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        # Ensure num_heads is divisible by num_kv_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim

        # Compute query and key-value projection sizes
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Scaling factor for query
        if config.query_pre_attn_scalar is not None:
            self.scaling = config.query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        # Define nn.Linear projections for QKV and output
        self.qkv_proj = nn.Linear(
        config.hidden_size,
        (self.num_heads + 2 * self.num_kv_heads) * config.head_dim
    )

        self.o_proj = nn.Linear(
        self.num_heads * config.head_dim,
        config.hidden_size
    )

        self.sliding_window_size = config.sliding_window_size
        self.attn_logit_softcapping = config.attn_logit_softcapping

        self.cache_k = torch.zeros((config.batch_size, config.max_position_embeddings, self.num_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((config.batch_size, config.max_position_embeddings, self.num_kv_heads, self.head_dim))
    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = None
    ) -> torch.Tensor:
        """
        Forward pass for attention mechanism.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            freqs_cis (torch.Tensor): Precomputed rotary embeddings.
            kv_write_indices (torch.Tensor): Indices to update key-value cache.
            kv_cache (Tuple[torch.Tensor, torch.Tensor]): Cached keys and values.
            mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Attention output of shape (batch_size, seq_len, hidden_size).
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project hidden states to QKV
        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape QKV for multi-head attention
        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings to Q and K
        xq = apply_rope(xq, freqs_complex=freqs_cis,device=hidden_states.device)
        xk = apply_rope(xk, freqs_complex=freqs_cis,device=hidden_states.device)

        # Update key-value cache
        if not self.training:
            self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
            self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

            key = self.cache_k[:batch_size, :start_pos + seq_len]
            value = self.cache_v[:batch_size, :start_pos + seq_len]
        else:
            key = xk
            value = xv

        key = repeat_kv(key, self.n_rep)
        value = repeat_kv(value, self.n_rep)

        key = key.to(xq.device)
        value = value.to(xq.device)

        # Compute scaled dot-product attention
        q = xq.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        q.mul_(self.scaling)  # Scale query
        scores = torch.matmul(q, k.transpose(-1, -2))  # Attention scores

        # Apply sliding window masking if needed
        if self.sliding_window_size is not None:
            sliding_mask = torch.triu(torch.ones_like(mask), -self.sliding_window_size + 1)
            sliding_mask = sliding_mask * torch.tril(sliding_mask, self.sliding_window_size - 1)
            mask = torch.where(sliding_mask == 1, mask, -2.3819763e38)

        # Apply softcapping to scores
        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = torch.tanh(scores) * self.attn_logit_softcapping

        # Add mask and compute attention probabilities
        scores = apply_causal_mask(scores,seq_len)
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # Compute attention output
        output = torch.matmul(scores, v)  # Attention applied to values
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Project back to hidden size
        output = self.o_proj(output)
        return output
    
class Gemma2DecoderLayer(nn.Module):

    def __init__(
        self,
        config
    ):
        super().__init__()
        self.self_attn = GemmaAttention(config)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_pre_ffw_norm else None
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_post_ffw_norm else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int
    ) -> torch.Tensor:

        # Self-Attention Block
        residual = hidden_states  # Save input for residual connection
        hidden_states = self.input_layernorm(hidden_states)  # Normalize input
        hidden_states = self.self_attn(  # Apply self-attention
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            start_pos = start_pos
        )
        hidden_states = self.post_attention_layernorm(hidden_states)  # Post-attention normalization
        hidden_states = residual + hidden_states  # Add residual connection

        # MLP Block
        residual = hidden_states  # Save input for residual connection
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)  # Normalize before MLP
        hidden_states = self.mlp(hidden_states)  # Apply MLP
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)  # Normalize after MLP
        hidden_states = residual + hidden_states  # Add residual connection

        return hidden_states

class Gemma2(nn.Module):

    def __init__(self, config:GemmaConfig):
        super().__init__()
        self.name = "gemma2"
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size,config.hidden_size)
        self.head_dim = config.num_attention_heads
        self.seq_length = config.max_position_embeddings
        self.n_layers = config.num_hidden_layers
        self.layers = nn.ModuleList([
            Gemma2DecoderLayer(config) for _ in range(self.n_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.freqs_complex = precompute_theta_freqs(
            seq_len=self.config.max_position_embeddings * 2,
            head_dim=self.config.hidden_size // self.config.num_attention_heads,
            device=self.config.device
        )
        self.Linear = nn.Linear(config.hidden_size, config.vocab_size)
    def forward(
        self,
        hidden_states: torch.Tensor,
        start_pos: int=0,

    ) -> torch.Tensor:
        _,seq_len = hidden_states.shape
        hidden_states = self.embed_tokens(hidden_states)
        freqs_cis = self.freqs_complex[start_pos: start_pos +seq_len] 
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                start_pos=start_pos
            )
        hidden_states = self.norm(hidden_states)
        out = self.Linear(hidden_states)
        return out