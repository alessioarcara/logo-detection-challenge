from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from src.models.neck import DPTNeck
from src.utils.checkpoint import load_weights


class BaseModel(nn.Module):
    def _load_checkpoint(self, checkpoint: Path | None) -> None:
        if checkpoint is None:
            return
        sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
        load_weights(self, sd)


class Detector(BaseModel):
    def __init__(
        self,
        backbone: nn.Module,
        freeze_backbone: bool,
        neck_features: int,
        layer_indices: list[int] = [2, 5, 8, 11],
        reassemble_scales: list[float] = [4.0, 2.0, 1.0, 0.5],
        head_dropout: float = 0.0,
        checkpoint: Path | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        embed_dim = cast(int, backbone.embed_dim)

        self.neck = DPTNeck(
            embed_dim=embed_dim,
            features=neck_features,
            layer_indices=layer_indices,
            reassemble_scales=reassemble_scales,
        )

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(neck_features, neck_features // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(neck_features // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Conv2d(neck_features // 4, 1, 1),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self._load_checkpoint(checkpoint)

    def forward(self, x: Tensor) -> Tensor:
        hidden_states = self.backbone(x)
        grid_size = cast(int, self.backbone.num_patches_per_side)
        features = self.neck(hidden_states, grid_size)
        return self.heatmap_head(features)  # (B, 1, H, W) logits


def decode_heatmap(heatmap: Tensor) -> tuple[Tensor, Tensor]:
    """
    It takes a heatmap (B, 1, H, W) of logits,
    finds the index of the maximum logit for each batch item,
    and converts it to normalized (x, y) coordinates.
    """
    B, _, H, W = heatmap.shape
    flat = heatmap.view(B, -1)  # (B, H*W)
    max_logit, idx = flat.max(dim=-1, keepdim=True)
    x = ((idx % W).float() + 0.5) / W
    y = ((idx // W).float() + 0.5) / H
    return torch.cat([x, y], dim=-1), max_logit
