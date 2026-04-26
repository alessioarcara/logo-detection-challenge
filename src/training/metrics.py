from abc import ABC, abstractmethod

import torch
from torch import Tensor

from src.utils.typings import MetricResults


class Metric(ABC):
    @abstractmethod
    def update(
        self,
        *,
        target_keypoint: Tensor,
        pred_keypoint: Tensor,
        target_objectness: Tensor,
        pred_objectness: Tensor,
    ) -> None: ...

    @abstractmethod
    def compute(self) -> MetricResults: ...

    @abstractmethod
    def reset(self) -> None: ...


class LocalizationAccuracy(Metric):
    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold + 1e-6
        self.reset()

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    @torch.no_grad()
    def update(
        self,
        *,
        target_keypoint: Tensor,
        pred_keypoint: Tensor,
        target_objectness: Tensor,
        pred_objectness: Tensor,
    ) -> None:
        mask = target_objectness.bool()
        if not mask.any():
            return

        delta = (pred_keypoint[mask] - target_keypoint[mask]).abs()
        localized = (delta <= self.threshold).all(dim=-1)

        self.correct += localized.sum().item()
        self.total += localized.numel()

    def compute(self) -> MetricResults:
        accuracy = self.correct / self.total if self.total > 0 else 0.0
        return {"localization_accuracy": accuracy}


class MeanAbsoluteError(Metric):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_error = 0.0
        self.total_count = 0

    @torch.no_grad()
    def update(
        self,
        *,
        target_keypoint: Tensor,
        pred_keypoint: Tensor,
        target_objectness: Tensor,
        pred_objectness: Tensor,
    ) -> None:
        mask = target_objectness.bool()
        if not mask.any():
            return

        error = (pred_keypoint[mask] - target_keypoint[mask]).abs()
        self.total_error += error.sum().item()
        self.total_count += error.numel()

    def compute(self) -> MetricResults:
        mae = self.total_error / self.total_count if self.total_count > 0 else 0.0
        return {"mae": mae}


class LogoPresenceAccuracy(Metric):
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.presence_correct = 0
        self.presence_total = 0
        self.absence_correct = 0
        self.absence_total = 0

    @torch.no_grad()
    def update(
        self,
        *,
        target_keypoint: Tensor,
        pred_keypoint: Tensor,
        target_objectness: Tensor,
        pred_objectness: Tensor,
    ) -> None:
        probs = torch.sigmoid(pred_objectness).reshape(-1)
        targets = target_objectness.bool().reshape(-1)
        preds = probs >= self.threshold

        self.presence_correct += preds[targets].sum().item()
        self.presence_total += targets.sum().item()

        self.absence_correct += (~preds[~targets]).sum().item()
        self.absence_total += (~targets).sum().item()

    def compute(self) -> MetricResults:
        presence = self.presence_correct / self.presence_total if self.presence_total > 0 else 0.0
        absence = self.absence_correct / self.absence_total if self.absence_total > 0 else 0.0
        return {
            "logo_presence_accuracy": presence,
            "logo_absence_accuracy": absence,
        }


class MetricCollection:
    def __init__(self, metrics: list[Metric]) -> None:
        self.metrics = metrics

    @torch.no_grad()
    def update(
        self,
        *,
        target_keypoint: Tensor,
        pred_keypoint: Tensor,
        target_objectness: Tensor,
        pred_objectness: Tensor,
    ) -> None:
        for metric in self.metrics:
            metric.update(
                target_keypoint=target_keypoint,
                pred_keypoint=pred_keypoint,
                target_objectness=target_objectness,
                pred_objectness=pred_objectness,
            )

    def compute(self) -> MetricResults:
        combined: MetricResults = {}
        for metric in self.metrics:
            combined.update(metric.compute())
        return combined

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()
