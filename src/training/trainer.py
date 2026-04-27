from collections import defaultdict

import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.generated import ConfigModel
from src.utils.misc import generate_run_name
from src.utils.typings import Stage


class Trainer:
    def __init__(self, cfg: ConfigModel) -> None:
        self.cfg = cfg
        self.model = cfg.model.to(cfg.device)
        self.train_loader = cfg.train_loader
        self.val_loader = cfg.val_loader
        self.metrics = cfg.metrics
        self.optim = cfg.optim
        self.scheduler = cfg.scheduler
        self.criterion = cfg.criterion
        self.callbacks = cfg.callbacks
        self.history: dict[str, float] = {}
        self.stop_training = False
        self.grad_scaler = torch.amp.GradScaler(enabled=self.cfg.use_mixed_precision)

    def _get_loader(self, stage: Stage) -> DataLoader | None:
        match stage:
            case Stage.TRAIN:
                return self.train_loader
            case Stage.VAL:
                return self.val_loader
            case _:
                return None

    def _prepare_input(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        images = batch["image"].to(self.cfg.device, non_blocking=True)
        keypoints = batch["keypoint"].to(self.cfg.device, non_blocking=True)
        valid_mask = batch["has_target"].to(self.cfg.device, non_blocking=True)
        return images, keypoints, valid_mask

    def _train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        images, target_keypoints, valid_mask = self._prepare_input(batch)

        self.optim.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=self.cfg.device,
            dtype=torch.bfloat16,
            enabled=self.cfg.use_mixed_precision,
        ):
            pred_keypoints, pred_objectness = self.model(images)
            loss_dict = self.criterion(
                pred_keypoints, pred_objectness, target_keypoints, valid_mask
            )
            loss = loss_dict["loss"]

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optim)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.grad_scaler.step(self.optim)
        self.grad_scaler.update()

        return {key: value.item() for key, value in loss_dict.items()}

    def train(self) -> None:
        wandb.init(
            project=self.cfg.wandb_project,
            entity=self.cfg.wandb_entity,
            name=generate_run_name(self.cfg.wandb_run_name),
            config=self.cfg.model_dump(),
        )

        try:
            for epoch in tqdm(range(1, self.cfg.num_epochs + 1)):
                self.model.train()

                for batch in tqdm(
                    self.train_loader, desc=f"Epoch {epoch}", leave=False
                ):
                    batch_log = self._train_step(batch)
                    wandb.log(
                        {
                            f"train/batch_{key}": value
                            for key, value in batch_log.items()
                        }
                        | {"train/lr": self.scheduler.get_last_lr()[0]}
                    )

                self.scheduler.step()

                if epoch % self.cfg.eval_freq == 0:
                    train_results = self.eval(Stage.TRAIN)
                    val_results = self.eval(Stage.VAL)

                    epoch_log: dict[str, float | int] = {}
                    epoch_log.update(train_results)
                    epoch_log.update(val_results)

                    self.history.update(epoch_log)

                    wandb.log(epoch_log)

                    self.on_all_evals_end()

                if self.stop_training:
                    logger.info("Stopping training early at epoch {}", epoch)
                    break
        finally:
            wandb.finish()

    def _eval_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        images, target_keypoints, valid_mask = self._prepare_input(batch)

        with torch.autocast(
            device_type=self.cfg.device,
            dtype=torch.bfloat16,
            enabled=self.cfg.use_mixed_precision,
        ):
            pred_keypoints, pred_objectness = self.model(images)
            loss_dict = self.criterion(
                pred_keypoints, pred_objectness, target_keypoints, valid_mask
            )

        self.metrics.update(
            target_keypoint=target_keypoints,
            pred_keypoint=pred_keypoints,
            target_objectness=valid_mask,
            pred_objectness=pred_objectness,
        )

        return {key: value.item() for key, value in loss_dict.items()}

    def _on_eval_end(self) -> None:
        for callback in self.callbacks:
            callback.on_eval_end(self)

    def on_all_evals_end(self) -> None:
        for callback in self.callbacks:
            callback.on_all_evals_end(self)

    @torch.inference_mode()
    def eval(self, stage: Stage) -> dict[str, float]:
        loader = self._get_loader(stage)

        if loader is None:
            logger.warning("No data loader for stage {}, skipping evaluation", stage)
            self._on_eval_end()
            return {}

        self.model.eval()
        self.metrics.reset()

        loss_totals: defaultdict[str, float] = defaultdict(float)
        num_batches = len(loader)

        for batch in tqdm(loader, desc=f"Evaluating {stage}", leave=False):
            loss_dict = self._eval_step(batch)
            for key, value in loss_dict.items():
                loss_totals[key] += value

        results = {
            f"{stage.value}/{key}": value / num_batches
            for key, value in loss_totals.items()
        }
        results.update(
            {
                f"{stage.value}/{key}": value
                for key, value in self.metrics.compute().items()
            }
        )

        self._on_eval_end()
        return results
