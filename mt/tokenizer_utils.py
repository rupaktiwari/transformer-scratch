"""Shared SentencePiece vocabulary for English and German."""

import json
from pathlib import Path

import sentencepiece as spm


def train_sentencepiece(texts: list[str], model_prefix: str, vocab_size: int) -> None:
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(model_prefix).with_suffix(".corpus.txt")
    corpus_path.write_text("\n".join(texts), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        user_defined_symbols=["<sep>"],
        hard_vocab_limit=False,
    )
    corpus_path.unlink(missing_ok=True)


def load_sentencepiece(model_path: str | Path):
    processor = spm.SentencePieceProcessor()
    processor.load(str(model_path))
    return processor


def special_ids(processor) -> dict:
    sep_id = processor.piece_to_id("<sep>")
    if sep_id < 0:
        raise RuntimeError("SentencePiece model is missing the <sep> token")
    return {
        "pad_id": processor.pad_id(),
        "unk_id": processor.unk_id(),
        "bos_id": processor.bos_id(),
        "eos_id": processor.eos_id(),
        "sep_id": sep_id,
        "vocab_size": processor.vocab_size(),
    }


def write_meta(path: str | Path, meta: dict) -> None:
    Path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_meta(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
