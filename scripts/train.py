import argparse
from pathlib import Path

from ezconfy import ConfigBuilder

from src.generated import ConfigModel
from src.utils.random import fix_random


def main(config_paths: list[str | Path], schema_path: str) -> None:
    cfg: ConfigModel = ConfigBuilder.from_files(
        config_paths=config_paths,
        schema_path=schema_path,
    )
    fix_random(cfg.seed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--configs",
        type=str,
        default=["configs/base.yaml"],
        nargs="+",
        help="Path(s) to YAML config file(s)",
    )
    p.add_argument(
        "--schema",
        type=str,
        default="configs/schema.yaml",
        help="Path to the YAML schema file",
    )
    args = p.parse_args()
    main(args.configs, args.schema)
