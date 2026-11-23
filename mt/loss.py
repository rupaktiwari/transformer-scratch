"""Token cross-entropy with optional label smoothing.

Smoothing follows the 2017 paper: the gold token keeps probability
1 - smoothing, and the remaining smoothing mass is shared by every other token.
Padding and, for the decoder-only model, source-prefix tokens are masked out.
"""

import torch
import torch.nn.functional as F


def token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    smoothing: float,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_gold = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if smoothing > 0:
        vocab = logits.size(-1)
        other = smoothing / (vocab - 1)
        gold = 1.0 - smoothing
        sum_log = log_probs.sum(dim=-1)
        loss = -(gold * log_gold + other * (sum_log - log_gold))
    else:
        loss = -log_gold
    weights = mask.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp(min=1.0)
