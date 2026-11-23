"""Publish a Hugging Face model page that points at the Weights & Biases charts."""

import os
from io import BytesIO
from pathlib import Path


def _card(title: str, arch: str, run_url: str, metrics: dict | None) -> str:
    descriptions = {
        "vanilla": "Vaswani 2017 encoder–decoder: sinusoidal positions, multi-head attention, cross-attention, a ReLU feed-forward, and post-layer-norm.",
        "modern": "Decoder-only stack: Pre-RMSNorm, RoPE, grouped-query attention, QK-norm, and a top-2 mixture of SwiGLU experts.",
    }
    project_url = run_url.split("/runs/")[0] if "/runs/" in run_url else run_url
    if metrics:
        metric_lines = "\n".join(f"- **{name}**: {value:.4f}" for name, value in metrics.items())
    else:
        metric_lines = "Evaluation has not finished yet. The charts above update while training runs."
    return f"""---
library_name: pytorch
license: mit
language:
- en
- de
tags:
- machine-translation
- pytorch
- from-scratch
datasets:
- wmt14
metrics:
- bleu
- chrf
- comet
---

# {title}

{descriptions.get(arch, "")}

Trained from scratch on 200,000 WMT14 English–German pairs.

## Training curves

Open the live Weights & Biases charts for this run:

[{run_url}]({run_url})

Both architectures log to the same project, so the curves can be overlaid:

[{project_url}]({project_url})

## Test metrics

{metric_lines}
"""


def publish_to_hub(config: dict, run_url: str, metrics: dict | None = None, checkpoint: str | None = None) -> None:
    repo_id = os.environ.get("HF_REPO_ID", "").strip()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not repo_id:
        return
    if not token:
        raise RuntimeError("HF_REPO_ID is set but HF_TOKEN is empty. Paste your Hugging Face write token in the notebook.")
    if not run_url:
        raise RuntimeError("No Weights & Biases run URL, so the Hugging Face page cannot link the charts.")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    readme = _card(config["experiment"], config["model"]["arch"], run_url, metrics)
    api.upload_file(
        path_or_fileobj=BytesIO(readme.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Link the Weights & Biases training charts",
    )
    if checkpoint is not None and Path(checkpoint).is_file():
        api.upload_file(
            path_or_fileobj=checkpoint,
            path_in_repo="best.pt",
            repo_id=repo_id,
            commit_message="Upload the best checkpoint",
        )
    print(f"Hugging Face model page: https://huggingface.co/{repo_id}")
