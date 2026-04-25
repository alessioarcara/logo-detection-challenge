import torch
import torch.nn as nn
from loguru import logger

from src.utils.typings import StateDict


def load_weights(model: nn.Module, sd: StateDict) -> None:
    model_sd = model.state_dict()
    before = {k: v.clone() for k, v in model_sd.items()}

    compatible = {
        k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape
    }
    model.load_state_dict(compatible, strict=False)

    for name, tensor in model.state_dict().items():
        changed = not torch.allclose(before[name], tensor)
        logger.info("{} {}", "changed" if changed else "unchanged", name)
