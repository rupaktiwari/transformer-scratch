"""Decoder-only translation model.

The block is the current language-model stack, not a second encoder:

- token embedding, no added positional vector
- pre-RMSNorm
- rotary positions
- grouped-query attention
- QK-RMSNorm
- sparse mixture of SwiGLU experts
- a final RMSNorm, which pre-norm stacks need because the last step is a residual
- output projection tied to the token embedding

English–German training packs each pair as: source tokens, <sep>, target tokens, EOS.
"""

import torch
from torch import nn

from modern_transformer_scratch.layers import DecoderBlock
from modern_transformer_scratch.rmsnorm import RMSNorm


def padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    return tokens.eq(pad_id).unsqueeze(1).unsqueeze(2)


def causal_mask(size: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


class ModernTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        n_layers: int = 6,
        d_ff: int = 512,
        dropout: float = 0.1,
        n_experts: int = 4,
        top_k: int = 2,
        aux_coef: float = 0.01,
        z_loss_coef: float = 1e-3,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_experts=n_experts,
                    top_k=top_k,
                    aux_coef=aux_coef,
                    z_loss_coef=z_loss_coef,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.proj.weight = self.tok.weight
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.proj:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.tok.weight[self.pad_id].zero_()

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout(self.tok(tokens))
        self_mask = causal_mask(tokens.size(1), tokens.device) | padding_mask(tokens, self.pad_id)
        aux_values = []
        for layer in self.layers:
            x, aux = layer(x, self_mask)
            aux_values.append(aux)
        logits = self.proj(self.final_norm(x))
        aux = torch.stack([value.reshape(()) for value in aux_values]).sum()
        return logits, aux
