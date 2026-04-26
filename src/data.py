from collections.abc import Sized
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import albumentations as A


class DetectionDataset(Dataset):
    def __init__(self, dataset: Sized, transform: A.Compose) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        if not isinstance(self.dataset, Sized):
            raise TypeError("DetectionDataset source dataset must implement __len__")
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
