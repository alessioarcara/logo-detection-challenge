import torch.nn as nn
from loguru import logger
from torch import Tensor


class LogoLoss(nn.Module):
    def __init__(self, keypoint_loss: nn.Module) -> None:
        super(LogoLoss, self).__init__()
        self.keypoint_loss = keypoint_loss

    def forward(
        self,
        pred_keypoint: Tensor,  # [B, 2]
        pred_objectness: Tensor,  # [B]
        target_keypoint: Tensor,  # [B, 2]
        valid_mask: Tensor,  # [B]
    ) -> dict[str, Tensor]:
        objectness_loss = nn.functional.binary_cross_entropy_with_logits(
            pred_objectness.squeeze(-1),
            valid_mask.float(),
        )

        if valid_mask.any():
            keypoint_loss = self.keypoint_loss(
                pred_keypoint[valid_mask], target_keypoint[valid_mask]
            )
        else:
            logger.warning("No valid targets in batch, skipping keypoint loss")
            keypoint_loss = pred_keypoint.sum() * 0.0

        return {
            "loss": objectness_loss + keypoint_loss,
            "objectness_loss": objectness_loss,
            "keypoint_loss": keypoint_loss,
        }
