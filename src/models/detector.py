from typing import cast

import torch.nn as nn
from torch import Tensor


class Detector(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        embed_dim = cast(int, backbone.embed_dim)
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        # Discard the [CLS] token
        pooled = features[:, 1:].mean(dim=1)
        return self.head(pooled)
