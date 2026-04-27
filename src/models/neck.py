import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(features)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(features)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(F.relu(x))))
        out = self.bn2(self.conv2(out))
        return x + out


class ReassembleBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale: float) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_features, out_features, 1, bias=False)
        if scale > 1.0:
            s = int(scale)
            self.spatial: nn.Module = nn.ConvTranspose2d(
                out_features,
                out_features,
                3,
                stride=s,
                padding=1,
                output_padding=s - 1,
                bias=False,
            )
        elif scale < 1.0:
            s = int(1.0 / scale)
            self.spatial = nn.Conv2d(
                out_features,
                out_features,
                3,
                stride=s,
                padding=1,
                bias=False,
            )
        else:
            self.spatial = nn.Identity()

    def forward(self, tokens: Tensor, grid_size: int) -> Tensor:
        B, N, D = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, D, grid_size, grid_size)
        return self.spatial(self.project(x))


class FusionBlock(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.skip_rcu = ResidualConvUnit(features)
        self.out_rcu = ResidualConvUnit(features)
        self.upsample = nn.ConvTranspose2d(
            features,
            features,
            3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = x + self.skip_rcu(skip)
        x = self.out_rcu(x)
        return self.upsample(x)


class DPTNeck(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        features: int,
        layer_indices: list[int],
        reassemble_scales: list[float],
    ) -> None:
        super().__init__()
        self.layer_indices = layer_indices

        self.reassembles = nn.ModuleList(
            [ReassembleBlock(embed_dim, features, scale) for scale in reassemble_scales]
        )

        self.fusions = nn.ModuleList(
            [FusionBlock(features) for _ in range(len(layer_indices))]
        )

    def forward(self, hidden_states: list[Tensor], grid_size: int) -> Tensor:
        spatial = [
            self.reassembles[i](hidden_states[idx], grid_size)
            for i, idx in enumerate(self.layer_indices)
        ]

        out = self.fusions[-1](spatial[-1])
        for i in range(len(spatial) - 2, -1, -1):
            out = self.fusions[i](out, spatial[i])

        return out
