from pathlib import Path
from typing import TypeAlias
from enum import StrEnum

import torch

StateDict: TypeAlias = dict[str, torch.Tensor]
PathLike: TypeAlias = Path | str
BBox: TypeAlias = tuple[float, float, float, float]
Point: TypeAlias = tuple[float, float]
MetricResults: TypeAlias = dict[str, float]


class Stage(StrEnum):
    TRAIN = "train"
    VAL = "val"
