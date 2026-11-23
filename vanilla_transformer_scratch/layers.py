"""Post-norm encoder and decoder layers.

Each sublayer is LayerNorm(x + Dropout(Sublayer(x))), as in the 2017 paper.
"""

from torch import nn

from vanilla_transformer_scratch.attention import MultiHeadAttention
from vanilla_transformer_scratch.ffn import PositionwiseFFN


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attended = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, memory_mask):
        attended = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attended))
        attended = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(attended))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x
