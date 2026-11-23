"""Full encoder–decoder. The output projection shares weights with the token embedding."""

import torch
from torch import nn

from vanilla_transformer_scratch.embeddings import Embeddings
from vanilla_transformer_scratch.layers import DecoderLayer, EncoderLayer


def padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """True where tokens are padding. Shape [batch, 1, 1, time]."""
    return tokens.eq(pad_id).unsqueeze(1).unsqueeze(2)


def causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """True on future positions. Shape [1, 1, time, time]."""
    mask = torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


class VanillaTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_enc_layers: int = 6,
        n_dec_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.src_embed = Embeddings(vocab_size, d_model, pad_id, dropout, max_len)
        self.tgt_embed = Embeddings(vocab_size, d_model, pad_id, dropout, max_len)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_enc_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_dec_layers)]
        )
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.proj.weight = self.tgt_embed.tok.weight
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.proj:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.d_model**-0.5)
        with torch.no_grad():
            self.src_embed.tok.weight[self.pad_id].zero_()
            self.tgt_embed.tok.weight[self.pad_id].zero_()

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        x = self.src_embed(src)
        mask = padding_mask(src, self.pad_id)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        x = self.tgt_embed(tgt)
        self_mask = causal_mask(tgt.size(1), tgt.device) | padding_mask(tgt, self.pad_id)
        memory_mask = padding_mask(src, self.pad_id)
        for layer in self.decoder_layers:
            x = layer(x, memory, self_mask, memory_mask)
        return self.proj(x)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.encode(src)
        logits = self.decode(tgt_in, memory, src)
        aux = logits.new_zeros(())
        return logits, aux
