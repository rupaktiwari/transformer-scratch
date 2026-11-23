"""One pre-norm decoder block.

    x = x + Attention(RMSNorm(x))
    x = x + MoE(RMSNorm(x))

There is no cross-attention. The source sentence is a prefix of the same
stream, and a causal mask stops a token from seeing the future.
"""

from torch import nn

from modern_transformer_scratch.attention import Attention
from modern_transformer_scratch.moe import SparseMoE
from modern_transformer_scratch.rmsnorm import RMSNorm


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        dropout: float,
        n_experts: int,
        top_k: int,
        aux_coef: float,
        z_loss_coef: float,
    ):
        super().__init__()
        self.norm_attn = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, n_kv_heads, dropout)
        self.norm_moe = RMSNorm(d_model)
        self.moe = SparseMoE(
            d_model, d_ff, n_experts, top_k, dropout, aux_coef, z_loss_coef
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, self_mask):
        x = x + self.dropout(self.attn(self.norm_attn(x), self_mask))
        expert_out, aux = self.moe(self.norm_moe(x))
        return x + expert_out, aux
