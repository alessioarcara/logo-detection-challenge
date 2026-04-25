from pathlib import Path

import cv2 as cv
import numpy as np

from src.utils.constants import VALID_IMAGE_EXTENSIONS
from src.utils.typings import PathLike


def get_image_paths_in_dir(
    directory: PathLike,
    extensions: frozenset[str] = VALID_IMAGE_EXTENSIONS,
) -> list[Path]:
    directory = Path(directory)

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    return [
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]


def load_rgb(
    path: PathLike,
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    image = cv.imread(str(path), cv.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    if size is not None:
        image = cv.resize(image, (size[1], size[0]), interpolation=cv.INTER_LINEAR)
    return image


def load_rgba(
    path: PathLike,
) -> tuple[np.ndarray, np.ndarray]:
    image = cv.imread(str(path), cv.IMREAD_UNCHANGED)

    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    if image.ndim != 3 or image.shape[-1] != 4:
        raise RuntimeError(f"Expected RGBA image: {path}, got shape {image.shape}")

    image = cv.cvtColor(image, cv.COLOR_BGRA2RGBA)
    return image[..., :3], image[..., 3]
