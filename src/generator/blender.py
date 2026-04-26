from typing import NamedTuple

import numpy as np

from src.generator.blends import Blend
from src.utils.bbox import bbox_from_alpha, shift_bbox
from src.utils.typings import BBox, Point


class PasteResult(NamedTuple):
    image: np.ndarray
    bbox: BBox | None
    origin: Point


class Blender:
    def __init__(
        self,
        blends: list[Blend],
        same_scene_variants: bool,
    ) -> None:
        """
        Args:
        ``blends``: list of Blends to use for blending logos onto backgrounds
        ``same_scene_variants``: when True, each scene (background + logo) is generated for every blend in the blends list
        """
        self._blends = tuple(blends)
        self.same_scene_variants = same_scene_variants

    def scene_index(self, idx: int) -> int:
        """
        When ``same_scene_variants`` is True, consecutive indices cycle through
        blends for the same scene.

        example: idx 0,1 -> scene 0; idx 2,3 -> scene 1; etc.
        """
        if self.same_scene_variants:
            return idx // len(self._blends)
        return idx

    def pick_blend(self, idx: int, rng: np.random.Generator) -> Blend:
        """
        When ``same_scene_variants`` is True, the blend is determined by idx.
        Otherwise, a random blend is picked.
        """
        if self.same_scene_variants:
            return self._blends[idx % len(self._blends)]
        return self._blends[int(rng.integers(len(self._blends)))]

    def paste(
        self,
        bg: np.ndarray,
        logo_rgb: np.ndarray,
        logo_alpha: np.ndarray,
        rng: np.random.Generator,
        blend: Blend,
    ) -> PasteResult:
        bg_h, bg_w = bg.shape[:2]
        logo_h, logo_w = logo_rgb.shape[:2]

        if logo_h > bg_h or logo_w > bg_w:
            raise RuntimeError(
                "Logo is larger than background after logo transform. "
                "Fix the logo_transform parameters"
            )

        x = int(rng.integers(0, bg_w - logo_w + 1))
        y = int(rng.integers(0, bg_h - logo_h + 1))
        origin = float(x), float(y)

        bbox = bbox_from_alpha(logo_alpha)
        if bbox is None:
            return PasteResult(image=bg.copy(), bbox=None, origin=origin)

        image = blend.paste(
            bg=bg,
            logo_rgb=logo_rgb,
            logo_alpha=logo_alpha,
            x=x,
            y=y,
        )

        return PasteResult(image=image, bbox=shift_bbox(bbox, origin), origin=origin)
