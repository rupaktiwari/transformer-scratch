"""Weights & Biases logging. Both models use the same metric names so their curves overlay."""

class Tracker:
    def __init__(self, config: dict, enabled: bool):
        self.enabled = enabled
        self.run = None
        self._wandb = None
        if not enabled:
            return
        import wandb

        self._wandb = wandb
        self.run = wandb.init(
            project=config.get("wandb_project", "transformer-scratch"),
            name=config["experiment"],
            config={key: value for key, value in config.items() if not str(key).startswith("_")},
        )

    def log(self, metrics: dict, step: int) -> None:
        if self.run is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
