"""Build the model named by a config file."""

from modern_transformer_scratch import ModernTransformer
from vanilla_transformer_scratch import VanillaTransformer


def build_model(config: dict, vocab_size: int, pad_id: int):
    model_cfg = config["model"]
    arch = model_cfg["arch"]
    if arch == "vanilla":
        return VanillaTransformer(
            vocab_size=vocab_size,
            pad_id=pad_id,
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_enc_layers=model_cfg["n_enc_layers"],
            n_dec_layers=model_cfg["n_dec_layers"],
            d_ff=model_cfg["d_ff"],
            dropout=model_cfg["dropout"],
            max_len=config["data"]["max_len"] + 8,
        )
    if arch == "modern":
        return ModernTransformer(
            vocab_size=vocab_size,
            pad_id=pad_id,
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_kv_heads=model_cfg["n_kv_heads"],
            n_layers=model_cfg["n_layers"],
            d_ff=model_cfg["d_ff"],
            dropout=model_cfg["dropout"],
            n_experts=model_cfg["n_experts"],
            top_k=model_cfg["top_k"],
            aux_coef=model_cfg["aux_coef"],
            z_loss_coef=model_cfg["z_loss_coef"],
        )
    raise ValueError(f"Unknown model.arch: {arch}. Use 'vanilla' or 'modern'.")


def count_parameters(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
