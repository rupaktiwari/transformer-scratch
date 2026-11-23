"""Translation metrics.

BLEU, chrF, and chrF++ come from sacreBLEU (chrF++ is chrF with word order 2).
METEOR is the NLTK WordNet score. WordNet synonyms are English, so on German
output METEOR is mostly a stemmed lexical score. It is still reported.
COMET is Unbabel/wmt22-comet-da, a neural metric that reads source, hypothesis,
and reference. Perplexity is exp(token cross-entropy) under teacher forcing.
For the decoder-only model only the German tokens contribute to that loss.
"""

import math

import sacrebleu
import torch

from mt.loss import token_loss


def _meteor(hypotheses: list[str], references: list[str]) -> float:
    import nltk
    from nltk.translate.meteor_score import meteor_score

    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    scores = [
        meteor_score([reference.split()], hypothesis.split())
        for hypothesis, reference in zip(hypotheses, references)
    ]
    return float(sum(scores) / max(len(scores), 1))


def _comet(sources: list[str], hypotheses: list[str], references: list[str], device: torch.device) -> float:
    from comet import download_model, load_from_checkpoint

    checkpoint = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(checkpoint)
    rows = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(sources, hypotheses, references)]
    gpus = 1 if device.type == "cuda" else 0
    prediction = model.predict(rows, batch_size=8, gpus=gpus, progress_bar=False)
    if hasattr(prediction, "system_score"):
        return float(prediction.system_score)
    return float(prediction["system_score"])


def translation_scores(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    perplexity: float | None = None,
    use_meteor: bool = True,
    use_comet: bool = True,
    device: torch.device | None = None,
) -> dict:
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=0)
    chrfpp = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    scores = {
        "bleu": bleu.score,
        "chrf": chrf.score,
        "chrfpp": chrfpp.score,
    }
    if use_meteor:
        scores["meteor"] = _meteor(hypotheses, references)
    if use_comet:
        if device is None:
            device = torch.device("cpu")
        scores["comet"] = _comet(sources, hypotheses, references, device)
    if perplexity is not None:
        scores["perplexity"] = perplexity
    return scores


@torch.inference_mode()
def perplexity(model, loader, arch: str, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        if arch == "modern":
            logits, _ = model(batch["input_ids"])
            mask = batch["loss_mask"]
            targets = batch["labels"]
        else:
            logits, _ = model(batch["src"], batch["tgt_in"])
            targets = batch["tgt_out"]
            mask = targets.ne(model.pad_id)
        loss = token_loss(logits, targets, mask, smoothing=0.0)
        tokens = mask.sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
    mean = total_loss / max(total_tokens, 1.0)
    return math.exp(mean)
