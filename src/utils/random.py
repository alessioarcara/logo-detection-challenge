import random

import numpy as np
import torch


def fix_random(seed: int) -> None:
    """
    Fix all the possible sources of randomness.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_for(
    base_seed: int,
    idx: int,
    attempt: int = 0,
    stream: int = 0,
) -> int:
    """
    Derive a deterministic per-sample seed from a base seed and sample coordinates.
    Uses numpy's SeedSequence to ensure independent, collision-free seeds.
    """
    sequence = np.random.SeedSequence([base_seed, idx, attempt, stream])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])
