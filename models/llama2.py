import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from dataclasses import dataclass
import torch

@dataclass
class LlamaConfig:
    # Model architecture
    config_name = "Llama2-100M"
    kv_cache: bool = True
    hidden_size: int = 512
    n_heads: int = 8
    n_layers: int = 12
    intermediate_size: int = 2048
    batch_size: int = 8
    max_steps: int = 2000

    n_kv_heads: int = 8
    sliding_window: int = 1024
    attention_bias: bool = False
    rms_norm_eps: float = 1e-6
    qk_norm: bool = False

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01
    train: bool = True

    # Data parameters
    max_position_embeddings: int = 2048
    vocab_size: int = 12000

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    drop_last: bool = True
    shuffle: bool = False

    def __post_init__(self):
        self.d_k = self.hidden_size // self.n_heads
        assert self.hidden_size % self.n_heads == 0, "embed_dim must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_kv_groups = self.n_heads // self.n_kv_heads


class RMSNorm(nn.Module):

    def __init__(self,dim, eps: float= 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.tensor):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.tensor):
        return self.gamma * self._norm(x)


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

def apply_causal_mask(scores: torch.tensor, seq_len: int):

    mask = torch.triu(torch.ones(seq_len,seq_len), diagonal=1)
    mask = mask.to(scores.device)

    mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))

    return scores

class SelfAttention(nn.Module):

    def __init__(self, args: LlamaConfig):
        super().__init__()
        self.training = args.train # for controling KV-cache
        self.n_kv_heads = args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.hidden_size // args.n_heads
        self.hidden_size = args.hidden_size

        self.wq = nn.Linear(args.hidden_size, self.n_heads_q * self.head_dim, bias=False)
        self.wk = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.hidden_size, bias=False)

        self.cache_k = torch.zeros((args.batch_size, args.max_position_embeddings, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.batch_size, args.max_position_embeddings, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.tensor, freqs_comlex: torch.tensor, start_pos: int=None):

        batch_size, seq_len, _ = x.shape # (batch_size, seq_len, dim)
        #! (batch_size, seq_len,dim) -> = (batch_size, seq_len, head_dim * n_heads)
        xq = self.wq(x)

        #! (batch_size, seq_len,dim) -> = (batch_size, seq_len, n_kv_head_dim * n_kv_heads)
        xk = self.wk(x)
        xv = self.wv(x)

        #! (batch_size, seq_len, dim) -> (batch_size, seq_len, n_head_q, head_dim)
        xq = xq.view((batch_size, seq_len, self.n_heads_q, self.head_dim))

        #! (batch_size, seq_len, dim) -> (batch_size, seq_len, n_kv_head, head_dim)
        xk = xk.view((batch_size, seq_len, self.n_kv_heads, self.head_dim))
        xv = xv.view((batch_size, seq_len, self.n_kv_heads, self.head_dim))

        # applying RoPE
        xq = apply_rope(xq, freqs_comlex,device=x.device)
        xk = apply_rope(xk, freqs_comlex,device=x.device)

        if not self.training:
            self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
            self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

            keys = self.cache_k[:batch_size, :start_pos + seq_len]
            values = self.cache_v[:batch_size, :start_pos + seq_len]

        else:
            keys = xk
            values = xv

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        keys = keys.to(xq.device)
        values = values.to(xq.device)


        # (B, seq_len, H_Q, Head_Dim) -> (B, H_Q, seq_len, Head_Dim)
        attn_output = F.scaled_dot_product_attention(
            xq, keys, values, is_causal=True, dropout_p= 0.0
        )
        # (B, H_Q, seq_len, Head_Dim) -> (B, seq_len, H_Q, Head_Dim) -> (B, seq_len, Dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.wo(attn_output)

class FFN(nn.Module):

    def __init__(self, args:LlamaConfig):
        super().__init__()

        self.w1 = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.w2 = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.w3 = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)


    def forward(self, x: torch.tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x


class Decoder(nn.Module):

    def __init__(self, args:LlamaConfig):
        super().__init__()
        dim = args.hidden_size

        self.attn = SelfAttention(args)
        self.ffn = FFN(args)
        self.norm_1 = RMSNorm(dim, args.rms_norm_eps)
        self.norm_2 = RMSNorm(dim, args.rms_norm_eps)

    def forward(self, x: torch.tensor ,freqs_comlex: torch.tensor, start_pos):

        h = x + self.attn.forward(
            self.norm_1(x), freqs_comlex, start_pos)

        out = h + self.ffn.forward(self.norm_2(h))
        return out

class LLaMA2(nn.Module):

    def __init__(self, args:LlamaConfig):
        super().__init__()
        assert args.vocab_size != -1, "Vocab size must be set"
        self.name: str = "LlaMa2"
        self.dim = args.hidden_size
        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size
        self.args = args
        self.token_embd_table = nn.Embedding(self.vocab_size, self.dim)
        self.linear = nn.Linear(self.dim, self.vocab_size, bias=False)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(Decoder(args))

        self.norm = RMSNorm(self.dim, args.rms_norm_eps)
        self.linear = nn.Linear(args.hidden_size, self.vocab_size)

        self.linear.weight = self.token_embd_table.weight
        self.freqs_complex = precompute_theta_freqs(
            seq_len=self.args.max_position_embeddings * 2,
            head_dim=self.args.hidden_size // self.args.n_heads,
            device=self.args.device
        )

    def forward(self, x: torch.tensor, start_pos: int=0):
        _, seq_len = x.shape
        h = self.token_embd_table(x)
        freqs_copmlex = self.freqs_complex[start_pos: start_pos+seq_len]

        for layer in self.layers:
            h = layer(h,freqs_copmlex, start_pos)
        h = self.norm(h).float()
        out = self.linear(h)

        return out
