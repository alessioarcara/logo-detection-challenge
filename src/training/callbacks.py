from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from loguru import logger
from torch import Tensor

import wandb
from src.utils.misc import sanitize
from src.utils.typings import PathLike

if TYPE_CHECKING:
    from src.training.trainer import Trainer


class Callback(ABC):
    def on_eval_end(self, trainer: "Trainer") -> None:
        pass

    def on_all_evals_end(self, trainer: "Trainer") -> bool:
        return False


class ModelMonitorCallback(Callback):
    def __init__(self, history_key: str, minimize: bool) -> None:
        self.history_key = history_key
        self.minimize = minimize
        self.best = float("inf") if minimize else -float("inf")

    def on_all_evals_end(self, trainer: "Trainer") -> bool:
        value = self._get_history_value(trainer)
        if value is None:
            return False

        return self._update_best(value)

    def _get_history_value(self, trainer: "Trainer") -> float | None:
        value = trainer.history.get(self.history_key)
        if value is None:
            logger.warning(
                "{} not found in history; skipping model monitoring", self.history_key
            )
        return value

    def _update_best(self, value: float) -> bool:
        improved = value < self.best if self.minimize else value > self.best
        if improved:
            self.best = value

        return improved


class ModelSavingCallback(ModelMonitorCallback):
    def __init__(self, history_key: str, minimize: bool, save_dir: PathLike) -> None:
        super().__init__(history_key=history_key, minimize=minimize)
        self.out_dir = Path(save_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def on_all_evals_end(self, trainer: "Trainer") -> bool:
        improved = super().on_all_evals_end(trainer)
        if not improved:
            return False

        checkpoint_path = self._checkpoint_path(trainer)
        state_dict = {
            name: tensor.detach().cpu()
            for name, tensor in trainer.model.state_dict().items()
        }
        torch.save(state_dict, checkpoint_path)
        logger.info(
            "Saved improved checkpoint to {} ({}={:.6f})",
            checkpoint_path,
            self.history_key,
            self.best,
        )
        return True

    def _checkpoint_path(self, trainer: "Trainer") -> Path:
        run_name = sanitize(trainer.cfg.wandb_run_name)
        metric_name = sanitize(self.history_key)
        return self.out_dir / f"{run_name}_{metric_name}_best.pth"


class EarlyStoppingCallback(ModelMonitorCallback):
    def __init__(self, history_key: str, minimize: bool, patience: int) -> None:
        super().__init__(history_key=history_key, minimize=minimize)
        if patience < 1:
            raise ValueError("patience must be at least 1")
        self.patience = patience
        self.counter = 0

    def on_all_evals_end(self, trainer: "Trainer") -> bool:
        value = self._get_history_value(trainer)
        if value is None:
            return False

        improved = self._update_best(value)
        if improved:
            self.counter = 0
            logger.info(
                "Early stopping monitor improved ({}={:.6f}); counter reset",
                self.history_key,
                value,
            )
            return True

        self.counter += 1
        logger.info(
            "No improvement for {} ({}/{} evals)",
            self.history_key,
            self.counter,
            self.patience,
        )

        if self.counter >= self.patience:
            trainer.stop_training = True
            logger.info(
                "Early stopping triggered for {} after {} evals without improvement",
                self.history_key,
                self.counter,
            )

        return False


class VisualizeCallback(Callback):
    def __init__(
        self,
        mean: list[float],
        std: list[float],
        num_samples: int = 8,
        radius: int = 6,
    ) -> None:
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.num_samples = num_samples
        self.radius = radius

    def on_all_evals_end(self, trainer: "Trainer") -> bool:
        batch = next(iter(trainer.train_loader))
        images, target_keypoints, valid_mask = trainer._prepare_input(batch)

        with torch.inference_mode():
            pred_keypoints, _ = trainer.model(images)

        n = min(self.num_samples, images.size(0))
        mean = self.mean.to(images.device)
        std = self.std.to(images.device)

        wandb_images = []
        for i in range(n):
            img = self._denormalize(images[i], mean, std)
            h, w = img.shape[:2]

            if valid_mask[i].item():
                gt = target_keypoints[i].cpu().numpy()
                cv2.circle(
                    img, self._to_px(gt, w, h), self.radius, (0, 255, 0), -1
                )  # Green for GT

            pred = pred_keypoints[i].cpu().numpy()
            cv2.circle(
                img, self._to_px(pred, w, h), self.radius, (255, 0, 0), -1
            )  # Red for pred

            wandb_images.append(wandb.Image(img))

        wandb.log({"visualizations/keypoints": wandb_images})
        return False

    def _denormalize(self, tensor: Tensor, mean: Tensor, std: Tensor) -> np.ndarray:
        img = (tensor * std + mean).clamp(0, 1) * 255
        return img.byte().permute(1, 2, 0).cpu().numpy().copy()

    def _to_px(self, kp: np.ndarray, w: int, h: int) -> tuple[int, int]:
        return int(kp[0] * w), int(kp[1] * h)
