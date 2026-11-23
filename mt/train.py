"""Train one config. Run this from the repository root on Colab.

    uv run python -m mt.train --config configs/vanilla_en_de.yaml
    uv run python -m mt.train --config configs/modern_en_de.yaml
"""

import argparse
import random
from pathlib import Path

import torch

from mt.config import load_config
from mt.data import make_loader, read_tsv
from mt.decode import greedy_modern, greedy_vanilla, ids_to_text
from mt.loss import token_loss
from mt.metrics import translation_scores
from mt.hub import publish_to_hub
from mt.modeling import build_model, count_parameters
from mt.schedules import cosine_lr, vaswani_lr
from mt.tokenizer_utils import load_sentencepiece, read_meta, special_ids
from mt.tracking import Tracker


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def batch_loss(model, batch: dict, arch: str, smoothing: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if arch == "modern":
        logits, aux = model(batch["input_ids"])
        ce = token_loss(logits, batch["labels"], batch["loss_mask"], smoothing)
    else:
        logits, aux = model(batch["src"], batch["tgt_in"])
        mask = batch["tgt_out"].ne(model.pad_id)
        ce = token_loss(logits, batch["tgt_out"], mask, smoothing)
    return ce + aux, ce, aux


def learning_rate(step: int, config: dict) -> float:
    train_cfg = config["train"]
    if train_cfg["schedule"] == "vaswani":
        return vaswani_lr(step, config["model"]["d_model"], train_cfg["warmup_steps"])
    if train_cfg["schedule"] == "cosine":
        return cosine_lr(step, train_cfg["warmup_steps"], train_cfg["max_steps"], train_cfg["lr"], train_cfg["min_lr"])
    raise ValueError(f"Unknown schedule {train_cfg['schedule']}")


def optimizer_for(model, config: dict):
    train_cfg = config["train"]
    if config["model"]["arch"] == "vanilla":
        return torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    decay, plain = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith("tok.weight"):
            plain.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": train_cfg["weight_decay"]},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=1.0,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


@torch.inference_mode()
def validation_loss(model, loader, arch: str, smoothing: float, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        loss, _, _ = batch_loss(model, move(batch, device), arch, smoothing)
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


@torch.inference_mode()
def sample_bleu(model, config: dict, meta: dict, processor, device: torch.device) -> float:
    rows = read_tsv(Path(config["data"]["dir"]) / "valid.tsv")
    rows = rows[: config["train"]["eval_sentences"]]
    arch = config["model"]["arch"]
    pad_id = meta["pad_id"]
    max_len = config["decode"]["max_new_tokens"]
    sources, hypotheses, references = [], [], []
    special = {meta["pad_id"], meta["bos_id"], meta["eos_id"], meta["sep_id"]}
    model.eval()
    for start in range(0, len(rows), 8):
        chunk = rows[start : start + 8]
        src_ids = [processor.encode(src, out_type=int)[: config["data"]["max_len"] - 2] for src, _ in chunk]
        width = max(len(ids) for ids in src_ids)
        src = torch.full((len(chunk), width), pad_id, dtype=torch.long, device=device)
        for row, ids in enumerate(src_ids):
            src[row, : len(ids)] = torch.tensor(ids, device=device)
        if arch == "modern":
            generated = greedy_modern(model, src, meta["sep_id"], meta["eos_id"], pad_id, max_len)
        else:
            generated = greedy_vanilla(model, src, meta["bos_id"], meta["eos_id"], pad_id, max_len)
        for (source, reference), ids in zip(chunk, generated):
            sources.append(source)
            references.append(reference)
            hypotheses.append(ids_to_text(processor, ids, meta["eos_id"], special))
    scores = translation_scores(sources, hypotheses, references, use_meteor=False, use_comet=False)
    model.train()
    return scores["bleu"]


def save_checkpoint(path: Path, model, optimizer, scaler, step: int, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "config": config,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default=None, help="Override data.dir. Use a Drive folder on Colab.")
    parser.add_argument("--save-dir", default=None, help="Override train.save_dir. Use a Drive folder on Colab.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_dir:
        config["data"]["dir"] = args.data_dir
    if args.save_dir:
        config["train"]["save_dir"] = args.save_dir
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU visible. On Colab: Runtime → Change runtime type → T4 GPU, then rerun.")
    data_dir = Path(config["data"]["dir"])
    meta = read_meta(data_dir / "meta.json")
    model = build_model(config, meta["vocab_size"], meta["pad_id"]).to(device)
    n_params = count_parameters(model)
    print(f"{config['experiment']}: {n_params:,} trainable parameters on {device}")
    tracker = Tracker(config, enabled=not args.no_wandb)
    tracker.log({"parameters": n_params}, step=0)
    save_dir = Path(config["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    if tracker.url():
        (save_dir / "wandb_run.url").write_text(tracker.url() + "\n", encoding="utf-8")
        publish_to_hub(config, tracker.url())

    train_loader = make_loader(
        data_dir / "train.tsv",
        data_dir,
        config["model"]["arch"],
        config["data"]["max_len"],
        config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
    )
    valid_loader = make_loader(
        data_dir / "valid.tsv",
        data_dir,
        config["model"]["arch"],
        config["data"]["max_len"],
        config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )
    optimizer = optimizer_for(model, config)
    use_amp = bool(config["train"]["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: learning_rate(step + 1, config))
    processor = load_sentencepiece(data_dir / "spm.model")
    # Touch special_ids so a bad SentencePiece model fails before step 1.
    special_ids(processor)

    best = float("inf")
    step = 0
    model.train()
    while step < config["train"]["max_steps"]:
        for batch in train_loader:
            step += 1
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss, ce, aux = batch_loss(model, batch, config["model"]["arch"], config["train"]["label_smoothing"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % config["train"]["log_every"] == 0 or step == 1:
                lr = optimizer.param_groups[0]["lr"]
                print(f"step {step}  loss {loss.item():.4f}  ce {ce.item():.4f}  aux {aux.item():.4f}  lr {lr:.6e}")
                tracker.log(
                    {"train/loss": loss.item(), "train/ce": ce.item(), "train/aux": aux.item(), "train/lr": lr},
                    step=step,
                )
            if step % config["train"]["eval_every"] == 0:
                val = validation_loss(model, valid_loader, config["model"]["arch"], config["train"]["label_smoothing"], device)
                bleu = sample_bleu(model, config, meta, processor, device)
                print(f"step {step}  valid_loss {val:.4f}  valid_bleu {bleu:.2f}")
                tracker.log({"valid/loss": val, "valid/bleu": bleu}, step=step)
                save_checkpoint(save_dir / "last.pt", model, optimizer, scaler, step, config)
                if val < best:
                    best = val
                    save_checkpoint(save_dir / "best.pt", model, optimizer, scaler, step, config)
                    print(f"saved {save_dir / 'best.pt'}")
            if step >= config["train"]["max_steps"]:
                break
    save_checkpoint(save_dir / "last.pt", model, optimizer, scaler, step, config)
    tracker.finish()
    print(f"finished at step {step}. checkpoints in {save_dir}")


if __name__ == "__main__":
    main()
