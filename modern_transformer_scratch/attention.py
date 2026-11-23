"""Grouped-query attention with RoPE and QK-RMSNorm.

n_kv_heads == n_heads is ordinary multi-head attention.
n_kv_heads == 1 is multi-query attention.
Anything in between is grouped-query attention: several query heads share one
key/value head. The shipped model uses this, because that is how current
decoder-only models cut the key/value cost.

The mask uses True for positions that must not be attended to.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from modern_transformer_scratch.rmsnorm import RMSNorm
from modern_transformer_scratch.rope import RotaryEmbedding, apply_rope


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, n_kv, time, dim = x.shape
    x = x[:, :, None, :, :].expand(batch, n_kv, n_rep, time, dim)
    return x.reshape(batch, n_kv * n_rep, time, dim)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must divide evenly by n_heads ({n_heads})")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must divide evenly by n_kv_heads ({n_kv_heads})")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.rope = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, time, _ = x.shape
        q = self.q_proj(x).view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope.cos_sin(time, q.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # Norm the vectors that actually enter the dot product. RoPE keeps
        # length, so doing this after the rotation stabilizes the scores.
        q = self.q_norm(q)
        k = self.k_norm(k)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(batch, time, -1)
        return self.out_proj(context)
