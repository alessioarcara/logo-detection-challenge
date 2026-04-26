from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from loguru import logger


class BaseModel(nn.Module):
    def __init__(self, checkpoint_path: str | None = None) -> None:
        super().__init__()
        if checkpoint_path is not None:
            try:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                self.load_state_dict(state_dict)
            except Exception as e:
                logger.exception(
                    f"Failed to load checkpoint from {checkpoint_path}: {e}"
                )


class Detector(BaseModel):
    def __init__(self, backbone: nn.Module, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        embed_dim = cast(int, backbone.embed_dim)

        # TODO: consider to add FPN neck like DPT does for better precision

        self.keypoint_head = nn.Linear(embed_dim, 2)
        self.objectness_head = nn.Linear(embed_dim, 1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.backbone(x)
        # Discard the [CLS] token
        pooled = features[:, 1:].mean(dim=1)

        keypoint = torch.sigmoid(self.keypoint_head(pooled))
        objectness = self.objectness_head(pooled)

        return keypoint, objectness
