import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger
from pipelime.items import (  # type: ignore[import-untyped]
    NpyNumpyItem,
    PngImageItem,
    YamlMetadataItem,
)
from pipelime.sequences import Sample, SamplesSequence  # type: ignore[import-untyped]

from src.utils.io import load_rgb


def convert_labelstudio_data_to_underfolder(
    json_path: Path,
    images_dir: Path,
    output_dir: Path,
    size: int,
) -> None:
    with open(json_path) as f:
        annotations = json.load(f)
    logger.info("Loaded {} annotations from {}", len(annotations), json_path)

    samples: list[Sample] = []
    for entry in annotations:
        image_path = images_dir / Path(entry["img"]).name

        if not image_path.exists():
            logger.warning("Image not found, skipping: {}", image_path)
            continue

        kp = entry["kp-1"][0]
        kp_norm = np.array([kp["x"] / 100.0, kp["y"] / 100.0], dtype=np.float32)

        samples.append(
            Sample(
                {
                    "image": PngImageItem(load_rgb(image_path, size=(size, size))),
                    "keypoint": NpyNumpyItem(kp_norm),
                    "keypoint_px": NpyNumpyItem(kp_norm * size),
                    "has_target": NpyNumpyItem(np.array(True)),
                    "metadata": YamlMetadataItem(
                        {
                            "source": image_path.name,
                            "annotation_id": entry["annotation_id"],
                            "original_width": kp["original_width"],
                            "original_height": kp["original_height"],
                        }
                    ),
                }
            )
        )

    logger.info("Created {} samples, writing to {}", len(samples), output_dir)
    SamplesSequence.from_list(samples).to_underfolder(
        folder=output_dir, exists_ok=True
    ).run()
    logger.info("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", type=Path, required=True, help="Label Studio JSON export")
    p.add_argument("--images", type=Path, required=True, help="Raw images directory")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation"),
        help="Output underfolder directory",
    )
    p.add_argument("--size", type=int, required=True, help="Output image size")
    args = p.parse_args()
    convert_labelstudio_data_to_underfolder(
        args.json, args.images, args.output, args.size
    )
