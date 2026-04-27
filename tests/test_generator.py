from types import SimpleNamespace

import albumentations as A
import numpy as np
import pytest
from albumentations import BboxParams, KeypointParams
from PIL import Image

from src.generator.blends import AlphaBlend
from src.generator.generator import LogoDatasetGenerator


@pytest.fixture()
def generator(tmp_path):
    logos_dir = tmp_path / "logos"
    logos_dir.mkdir()
    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()

    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(logos_dir / "logo.png")
    Image.new("RGB", (128, 128), (0, 0, 255)).save(bg_dir / "bg.png")

    cfg = SimpleNamespace(
        logos_dir=str(logos_dir),
        backgrounds_dir=str(bg_dir),
        output_size=(128, 128),
        length=5,
        negative_ratio=0.0,
        seed=42,
        max_retries=10,
        blends=[AlphaBlend()],
        same_scene_blend_variants=False,
        logo_transform=A.Compose(
            [A.LongestMaxSize(max_size=128)],
            keypoint_params=KeypointParams(format="xy", remove_invisible=True),
        ),
        composition_transform=A.Compose(
            [],
            bbox_params=BboxParams(
                format="coco", min_visibility=0.3, label_fields=["labels"]
            ),
            keypoint_params=KeypointParams(format="xy", remove_invisible=True),
        ),
    )
    return LogoDatasetGenerator(cfg)


def test_same_images_across_epochs(generator: LogoDatasetGenerator):
    gen = generator

    epoch1 = [np.asarray(gen.generate(i)["image"]()) for i in range(len(gen))]
    epoch2 = [np.asarray(gen.generate(i)["image"]()) for i in range(len(gen))]

    for i, (a, b) in enumerate(zip(epoch1, epoch2)):
        np.testing.assert_array_equal(
            a, b, err_msg=f"Sample {i} differs between epochs"
        )
