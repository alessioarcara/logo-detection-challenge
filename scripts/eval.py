import argparse
from pathlib import Path

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from ezconfy import ConfigBuilder
from loguru import logger
from tqdm import tqdm

from src.generated import ConfigModel
from src.models.detector import decode_heatmap
from src.training.trainer import Trainer
from src.utils.io import get_image_paths_in_dir
from src.utils.typings import Stage


@torch.inference_mode()
def predict(cfg: ConfigModel, src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)

    model = cfg.model.to(cfg.device)
    model.eval()

    transform = A.Compose(
        [A.Resize(cfg.image_size, cfg.image_size), A.Normalize(), ToTensorV2()]
    )
    paths = get_image_paths_in_dir(src)

    for path in tqdm(paths, desc="Predicting"):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if image is None:
            logger.warning("Failed to read image: {}", path)
            continue

        tensor = transform(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))["image"]
        heatmap = model(tensor.unsqueeze(0).to(cfg.device))
        kp, logit = decode_heatmap(heatmap)

        h, w = image.shape[:2]
        x, y = int(kp[0, 0] * w), int(kp[0, 1] * h)
        prob = torch.sigmoid(logit).item()

        out = image.copy()
        cv2.circle(out, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            out,
            f"{prob:.2f}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        cv2.imwrite(str(dst / path.name), out)

    logger.info("Saved {} predictions to {}", len(paths), dst)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", type=str, nargs="+", default=["configs/base.yaml"])
    p.add_argument("--schema", type=str, default="configs/schema.yaml")
    p.add_argument("--images_dir", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    args = p.parse_args()

    cfg: ConfigModel = ConfigBuilder.from_files(
        config_paths=args.configs,
        schema_path=args.schema,
    )  # type: ignore

    if args.images_dir is not None:
        dst = (
            args.output_dir
            or args.images_dir.parent / f"{args.images_dir.name}_predictions"
        )
        predict(cfg, args.images_dir, dst)
    else:
        trainer = Trainer(cfg)
        results = trainer.eval(Stage.VAL)
        for name, value in results.items():
            logger.info("{}: {:.4f}", name, value)


if __name__ == "__main__":
    main()
