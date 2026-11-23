"""Decode a split and write BLEU, chrF, chrF++, METEOR, COMET, and perplexity.

    uv run python -m mt.evaluate --config configs/vanilla_en_de.yaml --checkpoint checkpoints/vanilla_en_de/best.pt
"""

import argparse
import json
from pathlib import Path

import torch

from mt.config import load_config
from mt.data import make_loader, read_tsv
from mt.decode import beam_modern, beam_vanilla, greedy_modern, greedy_vanilla, ids_to_text
from mt.metrics import perplexity, translation_scores
from mt.hub import publish_to_hub
from mt.modeling import build_model
from mt.tokenizer_utils import load_sentencepiece, read_meta


def generate(model, rows, config, meta, processor, device) -> tuple[list[str], list[str], list[str]]:
    arch = config["model"]["arch"]
    pad_id = meta["pad_id"]
    max_new = config["decode"]["max_new_tokens"]
    beam = config["decode"]["beam_size"] if config["decode"]["strategy"] == "beam" else 1
    alpha = config["decode"]["length_penalty"]
    special = {meta["pad_id"], meta["bos_id"], meta["eos_id"], meta["sep_id"]}
    sources, hypotheses, references = [], [], []
    batch_size = 1 if beam > 1 else 16
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        src_ids = [processor.encode(src, out_type=int)[: config["data"]["max_len"] - 2] for src, _ in chunk]
        width = max(len(ids) for ids in src_ids)
        src = torch.full((len(chunk), width), pad_id, dtype=torch.long, device=device)
        for row, ids in enumerate(src_ids):
            if ids:
                src[row, : len(ids)] = torch.tensor(ids, device=device)
        if arch == "modern":
            if beam > 1:
                generated = beam_modern(model, src, meta["sep_id"], meta["eos_id"], beam, max_new, alpha)
            else:
                generated = greedy_modern(model, src, meta["sep_id"], meta["eos_id"], pad_id, max_new)
        elif beam > 1:
            generated = beam_vanilla(model, src, meta["bos_id"], meta["eos_id"], beam, max_new, alpha)
        else:
            generated = greedy_vanilla(model, src, meta["bos_id"], meta["eos_id"], pad_id, max_new)
        for (source, reference), ids in zip(chunk, generated):
            sources.append(source)
            references.append(reference)
            hypotheses.append(ids_to_text(processor, ids, meta["eos_id"], special))
    return sources, hypotheses, references


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N sentences. 0 means all.")
    parser.add_argument("--data-dir", default=None, help="Override data.dir. Use the same Drive folder as training.")
    parser.add_argument("--skip-comet", action="store_true", help="Kept so older notebooks still run. COMET is off.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_dir:
        config["data"]["dir"] = args.data_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(config["data"]["dir"])
    meta = read_meta(data_dir / "meta.json")
    model = build_model(config, meta["vocab_size"], meta["pad_id"]).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    rows = read_tsv(data_dir / f"{args.split}.tsv")
    if args.limit:
        rows = rows[: args.limit]
    processor = load_sentencepiece(data_dir / "spm.model")
    sources, hypotheses, references = generate(model, rows, config, meta, processor, device)
    loader = make_loader(
        data_dir / f"{args.split}.tsv",
        data_dir,
        config["model"]["arch"],
        config["data"]["max_len"],
        config["data"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    if args.limit:
        # Perplexity stays on the full split file when a generation limit is set,
        # unless the file itself is already the demo set. Rebuild a short loader
        # by scoring only the generated rows' teacher-forced loss inside generate's rows.
        from mt.data import ParallelDataset, make_collate
        from torch.utils.data import DataLoader, Subset

        dataset = ParallelDataset(data_dir / f"{args.split}.tsv", data_dir, config["model"]["arch"], config["data"]["max_len"])
        subset = Subset(dataset, range(min(args.limit, len(dataset))))
        loader = DataLoader(subset, batch_size=config["data"]["batch_size"], shuffle=False, collate_fn=make_collate(config["model"]["arch"], meta["pad_id"]))
    ppl = perplexity(model, loader, config["model"]["arch"], device)
    scores = translation_scores(
        sources,
        hypotheses,
        references,
        perplexity=ppl,
        use_comet=False,
        device=device,
    )
    out = Path(args.checkpoint).with_suffix(f".{args.split}.metrics.json")
    out.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    training_run = Path(args.checkpoint).parent / "wandb_run.url"
    if training_run.is_file():
        publish_to_hub(config, training_run.read_text(encoding="utf-8").strip(), scores, args.checkpoint)
    for name, value in scores.items():
        print(f"{name}: {value:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
