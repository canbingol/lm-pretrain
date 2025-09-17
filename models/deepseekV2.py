import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os

world_size = 1
rank = 0
@dataclass
class DeepseekConfig:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_name = "DeepSeek"
    batch_size: int = 2
    max_position_embeddings: int = 128
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 32_768
    hidden_size: int = 512
    inter_dim: int = 2 * hidden_size
    moe_inter_dim: int = 704
    n_layers: int = 2
    n_dense_layers: int = 1
    n_heads: int = 8
    # moe
    n_routed_experts: int = 2
    n_shared_experts: int = 1
    n_activated_experts: int = 1
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 256
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    qk_norm = False

    # yarn
    original_seq_len: int = 256
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.

    # data preparing
    shuffle: bool = True
    drop_last: bool = True

    # training
    train:bool = True

def precompute_freqs_cis(args) -> torch.Tensor:
    """
Precomputes frequency-based complex exponential values for rotary positional embeddings.

Parameters (Args):
    args (ModelArgs): Model arguments containing positional embedding parameters.

Returns:
    torch.Tensor: A tensor containing complex exponential values corresponding to positions.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_position_embeddings
    beta_fast = args.beta_fast # frequency limits
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    #? Computes the dimension index where the rotation angle exceeds the threshold 2π·num_rot for the specified number of rotations
    def find_correction_dim(num_rotations, dim, base, max_seq_len):

        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    #? Determines the dimension range where the rotation angle starts to degrade and where it fully degrades
    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):

        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    #? Creates a linearly increasing transition (ramp) vector in the range [0,1] for the specified interval
    def linear_ramp_factor(min, max, dim):

        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    
    #? If the sequence length exceeds the pretraining limit, smoothly adjust the frequencies
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:

    assert x.shape[-1] % 2 == 0, "Rotary dim must be divisible by 2!"
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).reshape(*x.shape[:-1], -1)
    return y.to(dtype)

class RMSNorm(nn.Module):

    def __init__(self, dim:int, eps:float=1e-3):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x:torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+ self.eps)

    def forward(self, x:torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)

class MLA(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.dim = args.hidden_size
        self.n_head = args.n_heads
        self.n_local_head = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.isTrain = args.train

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_head * self.qk_head_dim)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank) # W_DQ
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_head * self.qk_head_dim) # in features: c_t^Q  out features: q_t^C
        
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim) # here, W^DKV and W_ht^Kr computations are combined
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_head * (self.qk_nope_head_dim + self.v_head_dim)) # here, W^uk  x c_t^kv and W^uv x c_t^kv computations are combined
        self.wo = nn.Linear(self.n_head * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5

        if args.max_position_embeddings > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale


        self.register_buffer('kv_cache', torch.zeros(args.batch_size, args.max_position_embeddings, self.kv_lora_rank), persistent=False) # latent space generated for K/V heads
        self.register_buffer('pe_cache', torch.zeros(args.batch_size, args.max_position_embeddings, self.qk_rope_head_dim), persistent=False) # storing positional information in memory

    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor, mask:Optional[torch.Tensor]):
        batch_size, seq_len, _ = x.size()
        end_pos = start_pos + seq_len
        
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x))) # full q_t^c query vector

        q = q.view(batch_size,seq_len, self.n_local_head, self.qk_head_dim) # Divide q into heads
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) # we separate the jointly computed q and q_rope values
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim],dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis) # since k_pe does not include batch_size, we add the batch_size dimension
        # deepseek-style attention computation
        wkv_b = self.wkv_b.weight
        wkv_b = wkv_b.view(self.n_local_head, -1, self.kv_lora_rank)
        q_nope = torch.einsum('bshd,hdc->bshc', q_nope, wkv_b[:, :self.qk_nope_head_dim])
        if not self.isTrain:
                    
            self.kv_cache[:batch_size, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:batch_size, start_pos:end_pos] = k_pe.squeeze(2)

        assert q_nope.shape[-1] == self.kv_cache.shape[-1], "Head dim mismatch between q_nope and kv_cache" 
        kv = self.kv_cache[:batch_size, :end_pos].unsqueeze(2)  # -> [B, T, 1, R]
        pe = self.pe_cache[:batch_size, :end_pos].unsqueeze(2)  # -> [B, T, 1, R]
        scores = (
             torch.einsum('bshr,bthr->bsht', q_nope, kv) +
             torch.einsum('bshr,bthr->bsht', q_pe, pe)
            ) * self.softmax_scale

        if mask is None and end_pos > 1:
            mask = torch.full((end_pos, end_pos), float('-inf'), device=x.device).triu(1)

        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        x = torch.einsum('bsht,btc->bshc',scores, self.kv_cache[:batch_size, :end_pos])
        x = torch.einsum('bshc,hdc->bshd',x,wkv_b[:,-self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x

class Gate(nn.Module):
    def __init__(self, args): 
        super().__init__()
        self.dim = args.hidden_size
        self.topk = args.n_activated_experts
        self.topk_groups = args.n_limited_groups
        self.n_groups = args.n_expert_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.hidden_size))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts))

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute scores: each token produces a score against all experts
        scores = F.linear(x, self.weight, self.bias)
        if self.score_func == 'softmax':
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()

        original_scores = scores
        #? Used to decrease scores of frequently selected experts (or boost rarely selected ones).
        if self.bias is not None:
            scores = scores + self.bias
        
        #? If experts are divided into groups, select top-k groups at the group level
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2,dim=-1)[0].sum(dim=-1)
            # Select the highest-scoring groups
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            # Mask the remaining groups (their experts do not run)
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float('-inf')).flatten(1)
        
        # Top-k selection among experts (e.g., activate 2 experts)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        # Retrieve the scores of the selected experts from the original score matrix
        weights = original_scores.gather(1, indices)

        if self.score_func == 'sigmoid':
            weights = weights /( weights.sum(dim=-1, keepdim=True))
        
        weights =weights* self.route_scale

        self.last_scores = original_scores.detach()  # [B*T, N_r]
        self.last_topk = indices.detach()  
        return weights.type_as(x), indices


class MLP(nn.Module):
    def __init__(self, dim:int, inter_dim:int):

        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x:torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, args):

        super().__init__()
        self.dim = args.hidden_size
        assert args.n_routed_experts % world_size == 0
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        # for distributed MoE
        self.n_local_experts = args.n_routed_experts // world_size
        self.expert_start_idx = rank * self.n_local_experts
        self.expert_end_idx = self.expert_start_idx + self.n_local_experts
        self.gate = Gate(args)
        
        # routed experts
        self.experts = nn.ModuleList([MLP(args.hidden_size, args.moe_inter_dim) if self.expert_start_idx <= i < self.expert_end_idx else None for i in range(self.n_routed_experts)])
        # shared experts that every input passes through
        self.shared_experts = MLP(args.hidden_size, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x:torch.Tensor)-> torch.Tensor:

        shape = x.size()
        x = x.view(-1, self.dim)
        # Gating mechanism → scores (weights) and indices of selected experts
        weights, indices = self.gate(x)
        
        # Empty tensor (same shape) to accumulate expert outputs
        y = torch.zeros_like(x)
        
        # Counting how many times each expert is selected. This prevents an expert from being used too much or too little within the MoE
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()

        for i in range(self.expert_start_idx, self.expert_end_idx):
            expert = self.experts[i]

            if expert is None or counts[i] == 0:
                continue

            idx, top = torch.where(indices == i)
            # running routed experts
            y = y.index_add(0, idx, expert(x[idx]) * weights[idx, top, None])
        # running shared experts
        z = self.shared_experts(x)

        self.last_gate_scores = self.gate.last_scores
        self.last_gate_topk = self.gate.last_topk
        return (y + z + x).view(shape)


class Block(nn.Module):

    def __init__(self, layer_ids:int, args):
        
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.hidden_size, args.inter_dim) if layer_ids < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.hidden_size)
        self.ffn_norm = RMSNorm(args.hidden_size)

    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor, mask:torch.Tensor)-> torch.Tensor:

        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Deepseek(nn.Module):

    def __init__(self, args):
        self.name = "deepseek"
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        super().__init__()
        self.max_seq_len = args.max_position_embeddings
        self.embed = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.hidden_size)
        self.head = nn.Linear(args.hidden_size, args.vocab_size)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0, return_gate_info=False):

        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        logits = self.head(h)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        moe_layer = None
        for l in reversed(self.layers):
            if isinstance(l.ffn, MoE):
                moe_layer = l.ffn
                break
        if return_gate_info:
            moe_layer = None
            for l in reversed(self.layers):
                if isinstance(l.ffn, MoE):
                    moe_layer = l.ffn
                    break
            return logits, moe_layer.last_gate_scores, moe_layer.last_gate_topk

        return logits