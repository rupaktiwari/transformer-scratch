"""Input embeddings scaled by sqrt(d_model), plus fixed sinusoidal positions."""

import math

import torch
from torch import nn


def sinusoidal_positions(max_len: int, d_model: int) -> torch.Tensor:
    """PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(...)."""
    positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(positions * div)
    pe[:, 1::2] = torch.cos(positions * div)
    return pe


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_id: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.scale = math.sqrt(d_model)
        self.register_buffer("pe", sinusoidal_positions(max_len, d_model), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok(tokens) * self.scale
        x = x + self.pe[: tokens.size(1)].unsqueeze(0).to(dtype=x.dtype)
        return self.dropout(x)
