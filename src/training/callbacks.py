from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.trainer import Trainer


class Callback(ABC):
    def on_eval_end(self, trainer: "Trainer") -> bool:
        return False


# class ModelMonitorCallback(Callback):
#    def __init__(self, history_key: str, minimize: bool):
#        super().__init__()
#        self.history_key = history_key
#        self.best = float("inf") if minimize else 0.0
#        self.minimize = minimize
#
#    def on_eval_end(self, trainer: "Trainer") -> bool:
#        val = trainer.history.get(self.history_key)
#        if val is None:
#            logger.warning(f"{self.history_key} not found in history; skipping.")
#            return False
#
#        try:
#            val = float(val)
#        except ValueError:
#            logger.warning(f"{self.history_key} value {val} is not a float; skipping.")
#            return False
#
#        improved = val < self.best if self.minimize else val > self.best
#        if improved:
#            self.best = val
#            return True
#
#        return False
#
#
# class EarlyStoppingCallback(ModelMonitorCallback):
#    def __init__(self, history_key: str, minimize: bool, patience: int):
#        super().__init__(history_key, minimize)
#        self.patience = patience
#        self.counter = 0
#
#    def on_eval_end(self, trainer: "Trainer") -> bool:
#        improved = super().on_eval_end(trainer)
#
#        if improved:
#            self.counter = 0
#            logger.info("Improved. Counter reset.")
#        else:
#            self.counter += 1
#            logger.info(f"No improvement for {self.counter}/{self.patience}.")
#
#            if self.counter >= self.patience:
#                trainer.stop_training = True
#
#        return improved
#
#
# class ModelSavingCallback(ModelMonitorCallback):
#    def __init__(self, history_key: str, minimize: bool, out_dir: str):
#        super().__init__(history_key, minimize)
#        self.out_dir = Path(out_dir)
#        self.out_dir.mkdir(parents=True, exist_ok=True)
#        self.best_model: Optional[Dict[str, Tensor]] = None
#
#    def on_eval_end(self, trainer: "Trainer") -> bool:
#        improved = super().on_eval_end(trainer)
#
#        if improved:
#            self.best_model = trainer.model.state_dict()
#
#        return improved
#
#    def on_train_end(self, trainer: "Trainer") -> None:
#        if wandb.run is None:
#            logger.warning("No active wandb run detected; skipping model upload.")
#            return
#
#        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#        base_name = f"{trainer.config.wandb_base_run_name}_{timestamp}_{self.history_key}_{self.best:0.4f}"
#
#        base_name = base_name.replace(" ", "_").replace("/", "-")  # sanitize filename
#
#        checkpoint_path = self.out_dir / f"{base_name}.pth"
#
#        torch.save(self.best_model, checkpoint_path)
#        logger.info(f"Saved model checkpoint to {checkpoint_path}")
#
#        artifact = wandb.Artifact(name=base_name, type="model")
#        artifact.add_file(str(checkpoint_path))
#        wandb.run.log_artifact(artifact)
#        logger.info(f"Uploaded model artifact to wandb: {base_name}")
