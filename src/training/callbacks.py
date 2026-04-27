from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from loguru import logger
from torch import Tensor

import wandb
from src.models.detector import decode_heatmap
from src.utils.misc import sanitize
from src.utils.typings import PathLike, Stage

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
        value = trainer.history.get(self.history_key)
        if value is None:
            logger.warning(
                "{} not found in history; skipping model monitoring", self.history_key
            )
            return False

        try:
            value = float(value)
        except (TypeError, ValueError):
            logger.warning(
                "{} in history is not a number (got {}); skipping model monitoring",
                self.history_key,
                value,
            )
            return False

        improved = value < self.best if self.minimize else value > self.best
        if improved:
            self.best = value
            return True

        return False


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
        improved = super().on_all_evals_end(trainer)

        if improved:
            self.counter = 0
            logger.info(
                "Improvement detected for {}={:.6f}; resetting early stopping counter",
                self.history_key,
                self.best,
            )
        else:
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

        return improved


class VisualizeCallback(Callback):
    def __init__(
        self,
        stage: str,
        mean: list[float],
        std: list[float],
        num_samples: int = 8,
        radius: int = 6,
        objectness_threshold: float = 0.5,
    ) -> None:
        self.stage = Stage(stage)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.num_samples = num_samples
        self.radius = radius
        self.objectness_threshold = objectness_threshold

    def on_all_evals_end(self, trainer: "Trainer") -> bool:
        loader = trainer._get_loader(self.stage)
        if loader is None:
            return False

        batch = next(iter(loader))
        images, target_keypoints, valid_mask = trainer._prepare_input(batch)

        with torch.inference_mode():
            heatmap = trainer.model(images)

        pred_keypoints, pred_logits = decode_heatmap(heatmap)
        pred_probs = torch.sigmoid(pred_logits)

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
                )

            if pred_probs[i].item() > self.objectness_threshold:
                pred = pred_keypoints[i].cpu().numpy()
                cv2.circle(
                    img, self._to_px(pred, w, h), self.radius, (255, 0, 0), -1
                )

            wandb_images.append(wandb.Image(img))

        wandb.log({
            f"visualizations/{self.stage.value}_keypoints": wandb_images,
        })
        return False

    def _denormalize(self, tensor: Tensor, mean: Tensor, std: Tensor) -> np.ndarray:
        img = (tensor * std + mean).clamp(0, 1) * 255
        return img.byte().permute(1, 2, 0).cpu().numpy().copy()

    def _to_px(self, kp: np.ndarray, w: int, h: int) -> tuple[int, int]:
        return int(kp[0] * w), int(kp[1] * h)
