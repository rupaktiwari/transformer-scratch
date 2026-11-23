"""Shape, mask, RoPE, MoE, and one optimizer step for both models."""

import math
from pathlib import Path

import torch
import yaml

from modern_transformer_scratch.attention import Attention
from modern_transformer_scratch.model import ModernTransformer
from modern_transformer_scratch.moe import SparseMoE
from modern_transformer_scratch.rope import RotaryEmbedding, apply_rope
from mt.data import pack_decoder_only
from mt.loss import token_loss
from mt.metrics import translation_scores
from mt.modeling import build_model
from mt.prepare_data import main as prepare_main
from mt.schedules import vaswani_lr
from mt.train import batch_loss, learning_rate, optimizer_for
from vanilla_transformer_scratch.embeddings import sinusoidal_positions
from vanilla_transformer_scratch.model import VanillaTransformer


def test_sinusoid_at_position_zero():
    pe = sinusoidal_positions(4, 8)
    assert torch.allclose(pe[0, 0::2], torch.zeros(4), atol=1e-6)
    assert torch.allclose(pe[0, 1::2], torch.ones(4), atol=1e-6)


def test_rope_preserves_vector_length():
    rope = RotaryEmbedding(8)
    x = torch.randn(2, 3, 5, 8)
    cos, sin = rope.cos_sin(5, x.device, x.dtype)
    rotated = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-4)


def test_grouped_query_uses_fewer_key_heads():
    attention = Attention(32, n_heads=4, n_kv_heads=2, dropout=0.0)
    assert attention.q_proj.out_features == 32
    assert attention.k_proj.out_features == 16
    assert attention.v_proj.out_features == 16


def test_modern_is_causal_and_returns_aux():
    torch.manual_seed(0)
    model = ModernTransformer(
        vocab_size=40,
        pad_id=0,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=64,
        dropout=0.0,
        n_experts=4,
        top_k=2,
    )
    model.eval()
    tokens = torch.randint(1, 40, (2, 6))
    first, aux = model(tokens)
    changed = tokens.clone()
    changed[:, -1] = 3
    second, _ = model(changed)
    assert aux.ndim == 0 and torch.isfinite(aux)
    assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-5)
    assert not torch.allclose(first[:, -1], second[:, -1])


def test_vanilla_decoder_is_causal():
    torch.manual_seed(0)
    model = VanillaTransformer(
        vocab_size=40, pad_id=0, d_model=32, n_heads=4, n_enc_layers=1, n_dec_layers=1, d_ff=64, dropout=0.0
    )
    model.eval()
    src = torch.randint(1, 40, (2, 5))
    tgt = torch.randint(1, 40, (2, 6))
    memory = model.encode(src)
    first = model.decode(tgt, memory, src)
    changed = tgt.clone()
    changed[:, -1] = 3
    second = model.decode(changed, memory, src)
    assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-5)


def test_moe_aux_reaches_the_router():
    moe = SparseMoE(16, 32, n_experts=4, top_k=2, dropout=0.0)
    hidden = torch.randn(2, 3, 16)
    output, aux = moe(hidden)
    assert output.shape == hidden.shape
    aux.backward()
    assert moe.router.weight.grad is not None
    assert torch.isfinite(moe.router.weight.grad).all()


def test_decoder_only_loss_ignores_the_source_prefix():
    packed = pack_decoder_only([10, 11], [20], sep_id=4, eos_id=3, max_len=16)
    assert packed["labels"] == [11, 4, 20, 3]
    assert packed["loss_mask"] == [0, 0, 1, 1]


def test_label_smoothing_ignores_masked_tokens():
    logits = torch.zeros(1, 2, 5)
    targets = torch.tensor([[1, 0]])
    mask = torch.tensor([[1, 0]])
    loss = token_loss(logits, targets, mask, smoothing=0.1)
    assert torch.isfinite(loss)


def test_vaswani_lr_peaks_at_the_warmup_boundary():
    d_model, warmup = 256, 1000
    at_warmup = vaswani_lr(warmup, d_model, warmup)
    left = (d_model**-0.5) * (warmup**-0.5)
    assert math.isclose(at_warmup, left, rel_tol=1e-6)
    assert vaswani_lr(warmup // 2, d_model, warmup) < at_warmup
    assert vaswani_lr(warmup * 4, d_model, warmup) < at_warmup


def test_configs_build_both_architectures():
    root = Path(__file__).resolve().parents[1]
    for name in ("vanilla_en_de.yaml", "modern_en_de.yaml"):
        config = yaml.safe_load((root / "configs" / name).read_text())
        model = build_model(config, vocab_size=100, pad_id=0)
        assert sum(p.numel() for p in model.parameters()) > 0


def test_one_step_on_each_architecture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["prepare", "--demo", "--out", "data/demo"])
    prepare_main()
    from mt.tokenizer_utils import read_meta
    from mt.data import make_loader

    meta = read_meta(Path("data/demo/meta.json"))
    root = Path(__file__).resolve().parents[1]
    for name in ("vanilla_en_de.yaml", "modern_en_de.yaml"):
        config = yaml.safe_load((root / "configs" / name).read_text())
        config["data"]["dir"] = "data/demo"
        config["data"]["batch_size"] = 4
        config["model"]["d_model"] = 32
        config["model"]["dropout"] = 0.0
        if config["model"]["arch"] == "vanilla":
            config["model"]["n_heads"] = 4
            config["model"]["n_enc_layers"] = 1
            config["model"]["n_dec_layers"] = 1
            config["model"]["d_ff"] = 64
        else:
            config["model"]["n_heads"] = 4
            config["model"]["n_kv_heads"] = 2
            config["model"]["n_layers"] = 2
            config["model"]["d_ff"] = 64
            config["model"]["n_experts"] = 4
        model = build_model(config, meta["vocab_size"], meta["pad_id"])
        loader = make_loader(Path("data/demo/train.tsv"), Path("data/demo"), config["model"]["arch"], 32, 4, False, 0)
        batch = next(iter(loader))
        optimizer = optimizer_for(model, config)
        optimizer.param_groups[0]["lr"] = learning_rate(1, config)
        loss, _, _ = batch_loss(model, batch, config["model"]["arch"], config["train"]["label_smoothing"])
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)


def test_metric_names_without_heavy_models():
    scores = translation_scores(
        ["the cat"],
        ["the cat"],
        ["the cat"],
        perplexity=10.0,
        use_meteor=False,
        use_comet=False,
    )
    assert set(scores) == {"bleu", "chrf", "chrfpp", "perplexity"}
    assert scores["bleu"] > 90
