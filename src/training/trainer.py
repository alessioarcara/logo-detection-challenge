import torch
import torch.nn.functional as F
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
        self.metrics = cfg.metrics
        self.optim = cfg.optim
        self.scheduler = cfg.scheduler
        self.grad_scaler = torch.amp.GradScaler(enabled=self.cfg.use_mixed_precision)

    def _get_loader(self, stage: Stage) -> DataLoader | None:
        match stage:
            case Stage.TRAIN:
                return self.train_loader
            case Stage.VAL:
                return None  # TODO: implement val loader
            case _:
                return None

    def _prepare_input(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        images = batch["image"].to(self.cfg.device, non_blocking=True)
        keypoints = batch["keypoint"].to(self.cfg.device, non_blocking=True)
        valid_mask = batch["has_target"].to(self.cfg.device, non_blocking=True)
        return images, keypoints, valid_mask

    def _train_step(self, batch: dict[str, Tensor]) -> float:
        images, targets, valid_mask = self._prepare_input(batch)

        self.optim.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=self.cfg.device,
            dtype=torch.bfloat16,
            enabled=self.cfg.use_mixed_precision,
        ):
            y_pred = self.model(images)

            if valid_mask.any():
                loss = F.mse_loss(y_pred[valid_mask], targets[valid_mask])
            else:
                loss = y_pred.sum() * 0.0

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optim)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.grad_scaler.step(self.optim)
        self.grad_scaler.update()

        return loss.item()

    def _eval_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        return {}

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            batch_loss = self._train_step(batch)
            wandb.log({"train/batch_loss": batch_loss})

    @torch.no_grad()
    def _eval_epoch(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        self.metrics.reset()

        # TODO: add val_loader to config and iterate here
        results = self.metrics.compute()
        wandb.log({f"val/{k}": v for k, v in results.items()})
        return results

    def train(self) -> None:
        wandb.init(
            project=self.cfg.wandb_project,
            entity=self.cfg.wandb_entity,
            name=generate_run_name(self.cfg.wandb_run_name),
            config=self.cfg.model_dump(),
        )

        try:
            for epoch in tqdm(range(1, self.cfg.num_epochs + 1)):
                self._train_epoch(epoch)
                self.scheduler.step()

                if epoch % self.cfg.eval_freq == 0:
                    self._eval_epoch(epoch)
        finally:
            wandb.finish()

    @torch.inference_mode()
    def eval(self, stage: Stage) -> dict[str, float]:
        loader = self._get_loader(stage)

        if loader is None:
            logger.warning("No data loader for stage {}, skipping evaluation", stage)
            return {}

        self.model.eval()
        self.metrics.reset()

        for batch in tqdm(loader, desc=f"Evaluating {stage}", leave=False):
            loss_dict = self._eval_step(batch)

        return {}
