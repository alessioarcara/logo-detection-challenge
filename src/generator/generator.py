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
from torch.utils.data import Dataset as TorchDataset

from src.generator.blender import Blender
from src.utils.bbox import to_array
from src.utils.io import get_image_paths_in_dir, load_rgb, load_rgba
from src.utils.random import seed_for
from src.utils.typings import BBox, PathLike

if TYPE_CHECKING:
    from src.generated import GeneratorConfig


class LogoDatasetGenerator:
    def __init__(self, config: "GeneratorConfig") -> None:
        self._config = config
        self._blender = Blender(
            blends=config.blends,
            same_scene_variants=config.same_scene_blend_variants,
        )

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
        return self._config.length

    def as_sequence(self) -> SamplesSequence:
        return SamplesSequence.from_callable(
            generator_fn=self.generate, length=len(self)
        )

    def as_torch_dataset(self) -> TorchDataset:
        return self.as_sequence().torch_dataset()

    def write_underfolder(self, folder: PathLike, exists_ok: bool = False) -> Path:
        folder = Path(folder)
        self.as_sequence().to_underfolder(folder=folder, exists_ok=exists_ok).run()
        logger.info(f"Wrote {len(self)} samples to {folder}")
        return folder

    def _prepare_seed_rng(self, scene_idx: int, attempt: int) -> tuple[int, np.random.Generator]:
        """
        Derive a deterministic seed from (base_seed, scene, attempt) and align
        numpy rng + albumentations transforms to the same random state.
        """
        seed = seed_for(self._config.seed, scene_idx, attempt)
        rng = np.random.default_rng(seed)
        self._config.logo_transform.set_random_seed(seed)
        # * +1 so composition transform is independent from logo transform
        self._config.composition_transform.set_random_seed(seed + 1)
        return seed, rng

    def generate(self, idx: int) -> Sample:
        scene_idx = self._blender.scene_index(idx)
        seed, rng = self._prepare_seed_rng(scene_idx, attempt=0)
        is_negative = rng.random() < self._config.negative_ratio

        if is_negative:
            return self._try_generate_sample(idx, scene_idx, seed, rng, with_logo=False)

        for attempt in range(self._config.max_retries):
            seed, rng = self._prepare_seed_rng(scene_idx, attempt)
            sample = self._try_generate_sample(idx, scene_idx, seed, rng, with_logo=True)

            # * logo may be cropped out by composition transform, retry with a new attempt seed
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
        scene_idx: int,
        seed: int,
        rng: np.random.Generator,
        with_logo: bool,
    ) -> Sample:
        bg_path = self.bg_paths[rng.integers(len(self.bg_paths))]
        output_h, output_w = self._config.output_size
        bg = load_rgb(bg_path, size=(output_h, output_w))

        image = bg
        bboxes: list[BBox] = []
        blend = None

        if with_logo:
            blend = self._blender.pick_blend(idx, rng)
            logo_rgb, logo_alpha = self._logos[rng.integers(len(self._logos))]

            transformed = self._config.logo_transform(
                image=logo_rgb,
                mask=logo_alpha,
            )

            image, bbox = self._blender.paste(
                bg=bg,
                logo_rgb=transformed["image"],
                logo_alpha=transformed["mask"],
                rng=rng,
                blend=blend,
            )

            if bbox is not None:
                bboxes.append(bbox)

        augmented = self._config.composition_transform(
            image=image,
            bboxes=bboxes,
            labels=[0] * len(bboxes),
        )

        aug_bboxes = augmented.get("bboxes", [])
        has_target = len(aug_bboxes) > 0

        # * keypoint = bbox center in pixels and normalized [0, 1] coordinates
        keypoint_px = np.zeros(2, dtype=np.float32)
        keypoint = np.zeros(2, dtype=np.float32)

        if has_target:
            x, y, w, h = aug_bboxes[0]
            cx, cy = x + w / 2, y + h / 2
            keypoint_px = np.array([cx, cy], dtype=np.float32)
            keypoint = np.array([cx / output_w, cy / output_h], dtype=np.float32)

        return Sample(
            {
                "image": PngImageItem(augmented["image"]),
                "bboxes": NpyNumpyItem(to_array(aug_bboxes)),
                "keypoint_px": NpyNumpyItem(keypoint_px),
                "keypoint": NpyNumpyItem(keypoint),
                "has_target": NpyNumpyItem(np.array(has_target)),
                "metadata": YamlMetadataItem(
                    {
                        "index": int(idx),
                        "scene_index": int(scene_idx),
                        "seed": int(seed),
                        "background_source": bg_path.name,
                        "with_logo": bool(with_logo),
                        "blend": type(blend).__name__ if with_logo else "none",
                    }
                ),
            }
        )
