from pathlib import Path
from typing import TypeAlias

import torch

StateDict: TypeAlias = dict[str, torch.Tensor]
PathLike: TypeAlias = Path | str
BBox = tuple[float, float, float, float]
