from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from pipelime.sequences import SamplesSequence  # type: ignore[import-untyped]

import albumentations as A
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.utils.typings import PathLike


class DetectionDataset(Dataset):
    def __init__(self, dataset: Any, transform: A.Compose) -> None:
        self.dataset = dataset
        self.transform = transform

    @classmethod
    def from_sequence(cls, sequence: SamplesSequence, transform: A.Compose) -> Self:
        return cls(dataset=sequence.torch_dataset(), transform=transform)

    @classmethod
    def from_underfolder(cls, folder: PathLike, transform: A.Compose) -> Self:
        from pipelime.sequences import SamplesSequence  # type: ignore[import-untyped]

        sequence = SamplesSequence.from_underfolder(folder)
        return cls.from_sequence(sequence=sequence, transform=transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = self.dataset[index]
        transformed = self.transform(image=np.asarray(sample["image"]))
        image = transformed["image"]

        if not isinstance(image, Tensor):
            raise TypeError(
                "DetectionDataset transform must return a torch Tensor image"
            )

        return {
            "image": image,
            "keypoint": torch.as_tensor(sample["keypoint"], dtype=torch.float32),
            "has_target": torch.as_tensor(sample["has_target"], dtype=torch.bool),
        }


class DetectionCollateFn:
    def __call__(self, batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        return {
            "image": torch.stack([item["image"] for item in batch]),
            "keypoint": torch.stack([item["keypoint"] for item in batch]),
            "has_target": torch.stack([item["has_target"] for item in batch]),
        }
