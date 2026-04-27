from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from src.models.neck import DPTNeck


class Detector(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        freeze_backbone: bool,
        neck_features: int,
        layer_indices: list[int] = [2, 5, 8, 11],
        reassemble_scales: list[float] = [2.0, 1.0, 0.5, 0.25],
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
            nn.Conv2d(neck_features // 4, 1, 1),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        hidden_states = self.backbone(x)
        grid_size = cast(int, self.backbone.num_patches_per_side)
        features = self.neck(hidden_states, grid_size)
        return self.heatmap_head(features)  # (B, 1, H, W) logits


def decode_heatmap(heatmap: Tensor) -> tuple[Tensor, Tensor]:
    """
    (B,1,H,W) logits -> (B,2) -> (x,y) normalized, (B,1) peak logit."""
    B, _, H, W = heatmap.shape
    flat = heatmap.view(B, -1)
    max_logit, idx = flat.max(dim=-1, keepdim=True)
    x = ((idx % W).float() + 0.5) / W
    y = ((idx // W).float() + 0.5) / H
    return torch.cat([x, y], dim=-1), max_logit
