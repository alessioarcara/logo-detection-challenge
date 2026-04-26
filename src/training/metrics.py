from abc import ABC, abstractmethod

import torch
from torch import Tensor

from src.utils.typings import MetricResults


class Metric(ABC):
    @abstractmethod
    def update(self, y_true: Tensor, y_pred: Tensor) -> None: ...

    @abstractmethod
    def compute(self) -> MetricResults: ...

    @abstractmethod
    def reset(self): ...


class LocalizationAccuracy(Metric):
    def __init__(self):
        super().__init__()

    def reset(self):
        return super().reset()

    @torch.no_grad()
    def update(self, y_true: Tensor, y_pred: Tensor) -> None: ...

    def compute(self) -> MetricResults:
        return {}


class MetricCollection:
    def __init__(self, metrics: list[Metric]) -> None:
        self.metrics = metrics

    @torch.no_grad()
    def update(self, y_true: Tensor, y_pred: Tensor) -> None:
        for metric in self.metrics:
            metric.update(y_true, y_pred)

    def compute(self) -> MetricResults:
        combined_metrics = {}
        for metric in self.metrics:
            metric_dict = metric.compute()
            for k, v in metric_dict.items():
                combined_metrics[k] = v

        return combined_metrics

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()
