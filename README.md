# Transformer scratch: Vaswani and a modern decoder

Two translation models, trained on the same English–German data, scored with the same metrics.

| | `vanilla_transformer_scratch` | `modern_transformer_scratch` |
|---|---|---|
| Shape | Encoder–decoder | Decoder-only |
| Positions | Sinusoidal, added to the embeddings | RoPE, inside attention |
| Attention | Multi-head, plus cross-attention | Grouped-query (8 query heads, 2 key/value heads) |
| Extra attention norm | None | QK-RMSNorm after RoPE |
| Feed-forward | One ReLU network | Mixture of 4 SwiGLU experts, top-2 routing |
| Norm | Post-LayerNorm | Pre-RMSNorm, plus a final RMSNorm |
| Output | Tied to the target embedding | Tied to the token embedding |
| Objective | Next German token, given the encoder | Next token on `English <sep> German` |
| Schedule | Vaswani warmup | Cosine warmup, AdamW |

The modern folder is one stack, not a menu. SwiGLU, QK-norm, and the router z-loss are part of it because current decoder-only models use them, and they are small enough to read in one sitting.

## Metrics

After training, `mt.evaluate` writes:

- **BLEU** — sacreBLEU, 13a tokenization
- **chrF** and **chrF++** — character F-score; chrF++ also uses word order
- **METEOR** — NLTK. WordNet synonyms are English, so on German this is mostly a lexical score
- **COMET** — `Unbabel/wmt22-comet-da`
- **Perplexity** — exp of the teacher-forced token loss. The decoder-only model counts only the German tokens, not the English prefix

Greedy decoding is the default so a Colab eval finishes. Set `decode.strategy: beam` in the config for beam size 4 and the Vaswani length penalty.

## Train on Colab

Train the two models in two sessions. They share the Drive corpus and one Weights & Biases project, and neither notebook launches the other model.

1. Push is already at [rupaktiwari/transformer-scratch](https://github.com/rupaktiwari/transformer-scratch).
2. Colab → **T4 GPU**.
3. Upload or open `notebooks/train_vanilla.ipynb`. Run it from top to bottom. The login cell asks for your Wandb API key once.
4. When that run is finished, open `notebooks/train_modern.ipynb` the same way. If the Drive folder already has `meta.json`, it reuses the data.

Each notebook clones the repo, installs `requirements-colab.txt` on top of Colab's PyTorch, and writes checkpoints to `MyDrive/transformer-scratch`. Do not run `uv sync` on Colab.

In the Wandb project `transformer-scratch` the runs are `vanilla_en_de` and `modern_en_de`. Both log `train/loss`, `train/lr`, `valid/loss`, and `valid/bleu`, so the charts can be overlaid. The modern run also logs `train/aux`, the mixture-of-experts penalty. Final BLEU, chrF, chrF++, METEOR, COMET, and perplexity are logged by the evaluation cell.

If the modern run runs out of memory, rerun that cell with `--batch-size 8`. COMET is installed only in the evaluation cell.

## Local setup

uv is for this machine, not for the Colab runtime.

```bash
uv sync
uv run python -m mt.prepare_data
uv run python -m mt.train --config configs/vanilla_en_de.yaml
uv run python -m mt.evaluate --config configs/vanilla_en_de.yaml --checkpoint checkpoints/vanilla_en_de/best.pt
```

`prepare_data` downloads WMT14 English–German and keeps 200,000 training pairs plus the full validation and test sets. The first download is large. Both models share one SentencePiece vocabulary of 16,000 tokens. Run every command from the repository root.

## What to compare

Use the same data directory, the same SentencePiece model, and both checkpoints. The parameter counts printed at step 0 will not match: MoE stores every expert, and each token only runs through two of them. Read BLEU, chrF, chrF++, COMET, and perplexity together. A lower perplexity with a worse BLEU means the loss moved and the translations did not.

## Layout

```
vanilla_transformer_scratch/   2017 encoder–decoder
modern_transformer_scratch/    decoder-only stack
mt/                            data, loss, training, decoding, metrics
configs/                       one file per model
```

Low-resource pairs can reuse this code later: point `prepare_data` at a smaller TSV and keep the metrics as they are.
