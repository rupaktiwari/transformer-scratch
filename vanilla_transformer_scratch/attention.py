"""Scaled dot-product multi-head attention from Vaswani et al. 2017.

The mask uses True for positions that must not be attended to.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must divide evenly by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = query.size(0)
        q = self._split(self.q_proj(query), batch)
        k = self._split(self.k_proj(key), batch)
        v = self._split(self.v_proj(value), batch)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(batch, -1, self.n_heads * self.d_k)
        return self.out_proj(context)

    def _split(self, x: torch.Tensor, batch: int) -> torch.Tensor:
        x = x.view(batch, -1, self.n_heads, self.d_k)
        return x.transpose(1, 2)
