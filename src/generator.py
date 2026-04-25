from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from pipelime.items import (  # type: ignore[import-untyped]
    NpyNumpyItem,
    PngImageItem,
    YamlMetadataItem,
)
from pipelime.sequences import Sample, SamplesSequence  # type: ignore[import-untyped]

from src.utils.bbox import bbox_from_alpha, shift_bbox, to_array
from src.utils.io import get_image_paths_in_dir, load_rgb, load_rgba
from src.utils.random import seed_for
from src.utils.typings import BBox, PathLike

if TYPE_CHECKING:
    from generated import GeneratorConfig


def _alpha_blend(
    bg_roi: np.ndarray,
    logo_rgb: np.ndarray,
    logo_alpha: np.ndarray,
) -> np.ndarray:
    alpha = (logo_alpha.astype(np.float32) / 255.0)[..., None]
    foreground = logo_rgb.astype(np.float32)
    background = bg_roi.astype(np.float32)

    blended = alpha * foreground + (1.0 - alpha) * background
    return blended.astype(np.uint8)


class LogoDatasetGenerator:
    def __init__(self, config: "GeneratorConfig") -> None:
        self._config = config
        self._length = config.length

        self.logo_paths = get_image_paths_in_dir(config.logos_dir)
        self.bg_paths = get_image_paths_in_dir(config.backgrounds_dir)

        if not self.logo_paths:
            raise RuntimeError(f"No logos found in {config.logos_dir}")

        if not self.bg_paths:
            raise RuntimeError(f"No backgrounds found in {config.backgrounds_dir}")

        self._logos = [load_rgba(path) for path in self.logo_paths]

        logger.info(
            "Found {} logos, {} backgrounds",
            len(self.logo_paths),
            len(self.bg_paths),
        )

    def __len__(self) -> int:
        return self._length

    def as_sequence(self) -> SamplesSequence:
        return SamplesSequence.from_callable(
            generator_fn=self.generate, length=len(self)
        )

    def write_underfolder(self, folder: PathLike, exists_ok: bool = False) -> Path:
        folder = Path(folder)

        self.as_sequence().to_underfolder(folder=folder, exists_ok=exists_ok).run()

        logger.info(f"Wrote {len(self)} samples to {folder}")
        return folder

    def _prepare_sample_generation(
        self,
        idx: int,
        attempt: int,
    ) -> tuple[int, np.random.Generator]:
        """
        Fix all randomness needed to generate a reproducible sample
        """
        seed = seed_for(self._config.seed, idx, attempt)
        rng = np.random.default_rng(seed)

        self._config.logo_transform.set_random_seed(seed)
        self._config.composition_transform.set_random_seed(seed + 1)

        return seed, rng

    def generate(self, idx: int) -> Sample:
        seed, rng = self._prepare_sample_generation(idx, attempt=0)
        should_include_logo = rng.random() >= self._config.negative_ratio

        if not should_include_logo:
            return self._try_generate_sample(
                idx=idx,
                rng=rng,
                seed=seed,
                with_logo=False,
            )

        for attempt in range(self._config.max_retries):
            seed, rng = self._prepare_sample_generation(idx, attempt)

            sample = self._try_generate_sample(
                idx=idx,
                rng=rng,
                seed=seed,
                with_logo=True,
            )

            # Is the logo still visible after the composition transform?
            if sample["bboxes"]().shape[0] > 0:
                return sample

        raise RuntimeError(
            f"Sample {idx}: logo was cropped out in all "
            f"{self._config.max_retries} attempts. "
            "Consider relaxing min_visibility or the composition transform."
        )

    def _try_generate_sample(
        self,
        idx: int,
        rng: np.random.Generator,
        seed: int,
        with_logo: bool,
    ) -> Sample:
        bg_path = self.bg_paths[rng.integers(len(self.bg_paths))]
        bg = load_rgb(bg_path, size=self._config.output_size)

        image = bg
        bboxes: list[BBox] = []

        if with_logo:
            logo_rgb, logo_alpha = self._logos[rng.integers(len(self._logos))]

            transformed = self._config.logo_transform(
                image=logo_rgb,
                mask=logo_alpha,
            )

            logo_rgb = transformed["image"]
            logo_alpha = transformed["mask"]

            image, bbox = self._paste(
                bg=bg,
                logo_rgb=logo_rgb,
                logo_alpha=logo_alpha,
                rng=rng,
            )

            if bbox is not None:
                bboxes.append(bbox)

        augmented = self._config.composition_transform(
            image=image,
            bboxes=bboxes,
            labels=[0] * len(bboxes),
        )

        return Sample(
            {
                "image": PngImageItem(augmented["image"]),
                "bboxes": NpyNumpyItem(to_array(augmented.get("bboxes", []))),
                "metadata": YamlMetadataItem(
                    {
                        "index": int(idx),
                        "seed": int(seed),
                        "background_source": bg_path.name,
                        "with_logo": bool(with_logo),
                    }
                ),
            }
        )

    @staticmethod
    def _paste(
        bg: np.ndarray,
        logo_rgb: np.ndarray,
        logo_alpha: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, BBox | None]:
        bg_h, bg_w = bg.shape[:2]
        logo_h, logo_w = logo_rgb.shape[:2]

        if logo_h > bg_h or logo_w > bg_w:
            raise RuntimeError(
                "Logo is larger than background after logo transform. "
                "Fix the logo_transform parameters"
            )

        x = int(rng.integers(0, bg_w - logo_w + 1))
        y = int(rng.integers(0, bg_h - logo_h + 1))

        image = bg.copy()
        image[y : y + logo_h, x : x + logo_w] = _alpha_blend(
            bg_roi=image[y : y + logo_h, x : x + logo_w],
            logo_rgb=logo_rgb,
            logo_alpha=logo_alpha,
        )

        bbox = bbox_from_alpha(logo_alpha)
        if bbox is None:
            return image, None

        return image, shift_bbox(bbox, dx=x, dy=y)
