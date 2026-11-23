"""Parallel English–German batches.

Encoder–decoder batches are (source, decoder input, decoder target).
Decoder input is BOS + target. Decoder target is target + EOS.

Decoder-only batches pack source <sep> target EOS into one stream.
The loss mask is 1 only on the target tokens and the final EOS, so the model
is scored on the translation and not on copying the English prefix.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from mt.tokenizer_utils import load_sentencepiece, read_meta


def read_tsv(path: str | Path) -> list[tuple[str, str]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        header = handle.readline()
        if header.strip().lower() != "src\ttgt":
            raise ValueError(f"{path} must start with a 'src\\ttgt' header")
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            src, tgt = line.split("\t")
            if src and tgt:
                rows.append((src, tgt))
    return rows


def pack_decoder_only(
    src: list[int],
    tgt: list[int],
    sep_id: int,
    eos_id: int,
    max_len: int,
) -> dict:
    max_body = max_len - 2
    if len(src) + len(tgt) > max_body:
        src_budget = max(1, max_body // 2)
        src = src[:src_budget]
        tgt = tgt[: max_body - len(src)]
    seq = src + [sep_id] + tgt + [eos_id]
    inputs = seq[:-1]
    labels = seq[1:]
    src_len = len(src)
    loss_mask = [1 if index >= src_len else 0 for index in range(len(labels))]
    return {"input_ids": inputs, "labels": labels, "loss_mask": loss_mask}


class ParallelDataset(Dataset):
    def __init__(self, path: str | Path, data_dir: str | Path, arch: str, max_len: int):
        self.rows = read_tsv(path)
        self.sp = load_sentencepiece(Path(data_dir) / "spm.model")
        self.meta = read_meta(Path(data_dir) / "meta.json")
        self.arch = arch
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        src_text, tgt_text = self.rows[index]
        room = self.max_len - 2
        src = self.sp.encode(src_text, out_type=int)[:room]
        tgt = self.sp.encode(tgt_text, out_type=int)[:room]
        if self.arch == "modern":
            return pack_decoder_only(src, tgt, self.meta["sep_id"], self.meta["eos_id"], self.max_len)
        return {
            "src": src,
            "tgt_in": [self.meta["bos_id"]] + tgt,
            "tgt_out": tgt + [self.meta["eos_id"]],
        }


def _pad(seqs: list[list[int]], pad_id: int) -> torch.Tensor:
    width = max(len(seq) for seq in seqs)
    out = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    for row, seq in enumerate(seqs):
        out[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def make_collate(arch: str, pad_id: int):
    def collate(items: list[dict]) -> dict:
        if arch == "modern":
            return {
                "input_ids": _pad([item["input_ids"] for item in items], pad_id),
                "labels": _pad([item["labels"] for item in items], pad_id),
                "loss_mask": _pad([item["loss_mask"] for item in items], 0),
            }
        return {
            "src": _pad([item["src"] for item in items], pad_id),
            "tgt_in": _pad([item["tgt_in"] for item in items], pad_id),
            "tgt_out": _pad([item["tgt_out"] for item in items], pad_id),
        }

    return collate


def make_loader(path: Path, data_dir: Path, arch: str, max_len: int, batch_size: int, shuffle: bool, num_workers: int):
    meta = read_meta(data_dir / "meta.json")
    dataset = ParallelDataset(path, data_dir, arch, max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate(arch, meta["pad_id"]),
    )
