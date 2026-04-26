import torch

from src.training.metrics import (
    LocalizationAccuracy,
    LogoPresenceAccuracy,
    MeanAbsoluteError,
)


def test_localization_accuracy() -> None:
    metric = LocalizationAccuracy(threshold=0.1)

    metric.update(
        target_keypoint=torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
        pred_keypoint=torch.tensor([[0.5, 0.5], [0.9, 0.9]]),
        target_objectness=torch.tensor([1, 1]),
        pred_objectness=torch.zeros(2),
    )

    result = metric.compute()
    assert result["localization_accuracy"] == 0.5


def test_mean_absolute_error() -> None:
    metric = MeanAbsoluteError()

    metric.update(
        target_keypoint=torch.tensor([[0.0], [1.0]]),
        pred_keypoint=torch.tensor([[0.5], [1.0]]),
        target_objectness=torch.tensor([1, 1]),
        pred_objectness=torch.zeros(2),
    )

    result = metric.compute()
    assert result["mae"] == 0.25


def test_logo_presence_accuracy() -> None:
    metric = LogoPresenceAccuracy()

    metric.update(
        target_keypoint=torch.zeros(4, 2),
        pred_keypoint=torch.zeros(4, 2),
        target_objectness=torch.tensor([True, True, False, False]),
        pred_objectness=torch.tensor([2.0, -2.0, -2.0, 2.0]),
    )

    result = metric.compute()
    assert result["logo_presence_accuracy"] == 0.5
    assert result["logo_absence_accuracy"] == 0.5
