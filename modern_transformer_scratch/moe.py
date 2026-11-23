"""Sparse mixture of SwiGLU experts, in place of one dense feed-forward layer.

Each token is sent to its top-k experts. Two extra terms keep the router alive:

Load balance (Switch Transformer), with assignment mass normalized by k so
top-1 and top-2 share the same scale:

    n_experts * sum_i (fraction of tokens assigned to i) * (mean gate probability of i)

Router z-loss (ST-MoE): the square of logsumexp(router logits). It stops the
router logits from drifting to huge values.

Tokens are not dropped when an expert is busy. Every selected token is
computed. That uses more memory than a capacity limit, and it keeps the
routing math visible while you are learning it.
"""

import torch
from torch import nn

from modern_transformer_scratch.ffn import SwiGLU


class SparseMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int,
        top_k: int,
        dropout: float,
        aux_coef: float = 0.01,
        z_loss_coef: float = 1e-3,
    ):
        super().__init__()
        if top_k < 1 or top_k > n_experts:
            raise ValueError(f"top_k must be between 1 and n_experts ({n_experts})")
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_coef = aux_coef
        self.z_loss_coef = z_loss_coef
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ff, dropout) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time, dim = x.shape
        flat = x.reshape(-1, dim)
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.router(flat.float())
            probs = torch.softmax(logits, dim=-1)
            z_loss = torch.mean(torch.logsumexp(logits, dim=-1).square())
        top_prob, top_idx = torch.topk(probs, self.top_k, dim=-1)
        top_prob = top_prob / top_prob.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        combined = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            chosen = top_idx.eq(expert_id)
            token_mask = chosen.any(dim=-1)
            if not token_mask.any():
                continue
            token_ids = token_mask.nonzero(as_tuple=False).squeeze(-1)
            weights = (top_prob[token_ids] * chosen[token_ids].float()).sum(dim=-1)
            combined[token_ids] = combined[token_ids] + expert(flat[token_ids]) * weights.unsqueeze(-1)

        importance = probs.mean(dim=0)
        dispatch = torch.zeros_like(probs)
        dispatch.scatter_(1, top_idx, 1.0 / self.top_k)
        load = dispatch.mean(dim=0)
        balance = self.n_experts * torch.sum(importance * load.to(importance.dtype))
        aux = self.aux_coef * balance + self.z_loss_coef * z_loss.to(balance.dtype)
        return combined.view(batch, time, dim).to(dtype=x.dtype), aux.to(dtype=x.dtype)
