"""Download an English–German slice and train a shared SentencePiece model.

The default corpus is WMT14. Two hundred thousand training pairs is enough
for a from-scratch Colab run to learn a real mapping, and the official
validation and test sets are kept whole.

    uv run python -m mt.prepare_data
    uv run python -m mt.prepare_data --demo --out data/demo
"""

import argparse
from pathlib import Path

from mt.tokenizer_utils import special_ids, train_sentencepiece, write_meta


def write_tsv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["src\ttgt"]
    lines.extend(f"{src}\t{tgt}" for src, tgt in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def demo_rows() -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    train = [
        ("the cat sat on the mat", "die katze sass auf der matte"),
        ("the dog sat on the mat", "der hund sass auf der matte"),
        ("a cat sat here", "eine katze sass hier"),
        ("a dog sat here", "ein hund sass hier"),
        ("the bird sat on the roof", "der vogel sass auf dem dach"),
        ("a bird sat here", "ein vogel sass hier"),
        ("the cat is small", "die katze ist klein"),
        ("the dog is small", "der hund ist klein"),
    ]
    valid = [("the bird is small", "der vogel ist klein")]
    test = [("a cat is small", "eine katze ist klein")]
    return train, valid, test


def wmt_rows(train_samples: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    from datasets import load_dataset

    def take(split: str, limit: int | None) -> list[tuple[str, str]]:
        dataset = load_dataset("wmt14", "de-en", split=split)
        rows = []
        for example in dataset:
            src = example["translation"]["en"].strip()
            tgt = example["translation"]["de"].strip()
            if src and tgt:
                rows.append((src, tgt))
            if limit is not None and len(rows) >= limit:
                break
        return rows

    return take(f"train[:{train_samples}]", train_samples), take("validation", None), take("test", None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/en-de")
    parser.add_argument("--train-samples", type=int, default=200_000)
    parser.add_argument("--vocab-size", type=int, default=16_000)
    parser.add_argument("--demo", action="store_true", help="Write a tiny synthetic corpus for a smoke test")
    args = parser.parse_args()

    out = Path(args.out)
    train, valid, test = demo_rows() if args.demo else wmt_rows(args.train_samples)
    write_tsv(out / "train.tsv", train)
    write_tsv(out / "valid.tsv", valid)
    write_tsv(out / "test.tsv", test)

    texts = [side for pair in train + valid for side in pair]
    vocab_size = min(args.vocab_size, 80) if args.demo else args.vocab_size
    train_sentencepiece(texts, str(out / "spm"), vocab_size)
    from mt.tokenizer_utils import load_sentencepiece

    meta = special_ids(load_sentencepiece(out / "spm.model"))
    meta.update({"src_lang": "en", "tgt_lang": "de", "train_rows": len(train), "valid_rows": len(valid), "test_rows": len(test)})
    write_meta(out / "meta.json", meta)
    print(f"Wrote {len(train)} train, {len(valid)} valid, {len(test)} test pairs to {out}")
    print(f"Vocabulary size {meta['vocab_size']}, sep id {meta['sep_id']}")


if __name__ == "__main__":
    main()
