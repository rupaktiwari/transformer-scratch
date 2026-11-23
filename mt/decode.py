"""Greedy and beam search.

Encoder–decoder search encodes the source once, then grows the target from BOS.
Decoder-only search starts from the source tokens plus <sep>, then grows the target.
Beam ranking uses the Vaswani length penalty: ((5 + length) / 6) ** alpha.
"""

import torch
import torch.nn.functional as F

from vanilla_transformer_scratch.model import VanillaTransformer


def length_penalty(length: int, alpha: float) -> float:
    return ((5.0 + length) / 6.0) ** alpha


def ids_to_text(processor, ids: list[int], eos_id: int, special_ids: set[int]) -> str:
    kept = []
    for token_id in ids:
        if token_id == eos_id:
            break
        if token_id in special_ids:
            continue
        kept.append(token_id)
    return processor.decode(kept)


@torch.inference_mode()
def greedy_vanilla(model: VanillaTransformer, src: torch.Tensor, bos_id: int, eos_id: int, pad_id: int, max_len: int) -> list[list[int]]:
    memory = model.encode(src)
    tokens = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    generated = [[] for _ in range(src.size(0))]
    for _ in range(max_len):
        next_id = model.decode(tokens, memory, src)[:, -1].argmax(dim=-1)
        next_id = torch.where(finished, torch.full_like(next_id, pad_id), next_id)
        for row, token_id in enumerate(next_id.tolist()):
            if not finished[row]:
                generated[row].append(token_id)
        finished = finished | next_id.eq(eos_id)
        tokens = torch.cat([tokens, next_id.unsqueeze(1)], dim=1)
        if bool(finished.all()):
            break
    return generated


@torch.inference_mode()
def greedy_modern(model, src: torch.Tensor, sep_id: int, eos_id: int, pad_id: int, max_new: int) -> list[list[int]]:
    sep = torch.full((src.size(0), 1), sep_id, dtype=torch.long, device=src.device)
    tokens = torch.cat([src, sep], dim=1)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    generated = [[] for _ in range(src.size(0))]
    for _ in range(max_new):
        next_id = model(tokens)[0][:, -1].argmax(dim=-1)
        next_id = torch.where(finished, torch.full_like(next_id, pad_id), next_id)
        for row, token_id in enumerate(next_id.tolist()):
            if not finished[row]:
                generated[row].append(token_id)
        finished = finished | next_id.eq(eos_id)
        tokens = torch.cat([tokens, next_id.unsqueeze(1)], dim=1)
        if bool(finished.all()):
            break
    return generated


def _pick(scored: list[tuple[list[int], float]], beam_size: int, alpha: float):
    scored.sort(key=lambda item: item[1] / length_penalty(max(len(item[0]), 1), alpha), reverse=True)
    return scored[:beam_size]


@torch.inference_mode()
def beam_vanilla(model, src: torch.Tensor, bos_id: int, eos_id: int, beam_size: int, max_len: int, alpha: float) -> list[list[int]]:
    memory = model.encode(src)
    results = []
    for row in range(src.size(0)):
        mem = memory[row : row + 1]
        src_row = src[row : row + 1]
        beams = [([bos_id], 0.0)]
        finished = []
        for _ in range(max_len):
            candidates = []
            for seq, score in beams:
                if seq[-1] == eos_id:
                    finished.append((seq[1:], score))
                    continue
                prefix = torch.tensor(seq, dtype=torch.long, device=src.device).view(1, -1)
                log_probs = F.log_softmax(model.decode(prefix, mem, src_row)[:, -1], dim=-1).squeeze(0)
                values, indices = torch.topk(log_probs, beam_size)
                for value, index in zip(values.tolist(), indices.tolist()):
                    candidates.append((seq + [index], score + value))
            if not candidates:
                break
            beams = _pick(candidates, beam_size, alpha)
        finished.extend((seq[1:], score) for seq, score in beams if seq[-1] != eos_id)
        best = _pick(finished, 1, alpha)[0][0]
        results.append(best)
    return results


@torch.inference_mode()
def beam_modern(model, src: torch.Tensor, sep_id: int, eos_id: int, beam_size: int, max_new: int, alpha: float) -> list[list[int]]:
    results = []
    for row in range(src.size(0)):
        prefix = src[row].tolist() + [sep_id]
        beams = [(prefix, 0.0)]
        finished = []
        for _ in range(max_new):
            candidates = []
            for seq, score in beams:
                if seq[-1] == eos_id and len(seq) > len(prefix):
                    finished.append((seq[len(prefix) :], score))
                    continue
                tokens = torch.tensor(seq, dtype=torch.long, device=src.device).view(1, -1)
                log_probs = F.log_softmax(model(tokens)[0][:, -1], dim=-1).squeeze(0)
                values, indices = torch.topk(log_probs, beam_size)
                for value, index in zip(values.tolist(), indices.tolist()):
                    candidates.append((seq + [index], score + value))
            if not candidates:
                break
            beams = _pick(candidates, beam_size, alpha)
        start = len(prefix)
        finished.extend((seq[start:], score) for seq, score in beams if seq[-1] != eos_id)
        best = _pick(finished, 1, alpha)[0][0]
        results.append(best)
    return results
