"""Learning-rate schedules.

Vaswani et al. set the Adam step size to 1 and multiply by

    d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})

The cosine schedule is the one used by most decoder-only training runs:
linear warmup, then a cosine decay down to min_lr.
"""

import math


def vaswani_lr(step: int, d_model: int, warmup: int) -> float:
    step = max(step, 1)
    return (d_model**-0.5) * min(step**-0.5, step * (warmup**-1.5))


def cosine_lr(step: int, warmup: int, total: int, base_lr: float, min_lr: float) -> float:
    step = max(step, 1)
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * scale
