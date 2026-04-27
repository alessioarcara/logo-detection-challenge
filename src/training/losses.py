import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HeatmapLoss(nn.Module):
    """CenterNet-style modified focal loss for keypoint heatmaps.

    Positive pixels (GT == 1) get ``-(1-p)^alpha * log(p)``.
    Negative pixels get ``-(1-y)^beta * p^alpha * log(1-p)``,
    which down-weights cells near the peak so the Gaussian
    neighbourhood is not penalised harshly.
    """

    def __init__(
        self,
        sigma: float = 0.01,
        alpha: float = 2.0,
        beta: float = 4.0,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

    def _build_gt_heatmap(
        self, keypoints: Tensor, valid_mask: Tensor, height: int, width: int
    ) -> Tensor:
        B = keypoints.shape[0]
        device = keypoints.device

        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )

        cx = (keypoints[:, 0] * width).view(B, 1, 1)
        cy = (keypoints[:, 1] * height).view(B, 1, 1)

        sigma_px = self.sigma * min(height, width)
        g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma_px**2))
        g = g / g.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        #! if valid_mask is False, set the heatmap to all zeros
        #! so the model is forced to predict low probability everywhere
        g = g * valid_mask.float().view(B, 1, 1)

        return g.unsqueeze(1)

    def forward(
        self,
        pred_heatmap: Tensor,  # (B, 1, H, W) logits
        target_keypoint: Tensor,  # (B, 2)
        valid_mask: Tensor,  # (B,)
    ) -> dict[str, Tensor]:
        _, _, H, W = pred_heatmap.shape
        gt = self._build_gt_heatmap(target_keypoint, valid_mask, H, W)

        pred = torch.sigmoid(pred_heatmap)

        loss = torch.where(
            gt.eq(1),
            -((1 - pred) ** self.alpha) * F.logsigmoid(pred_heatmap),
            -((1 - gt) ** self.beta) * (pred**self.alpha) * F.logsigmoid(-pred_heatmap),
        )

        num_pos = valid_mask.float().sum().clamp(min=1)
        loss = loss.sum() / num_pos

        return {"loss": loss, "heatmap_loss": loss}
