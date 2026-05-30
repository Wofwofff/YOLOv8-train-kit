import argparse
import logging
from pathlib import Path

from yolokit.config import Config
from yolokit.split import split_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main(args) -> None:
    config = Config.from_yaml(args.config)

    export_dir = args.yolo_export
    if not (export_dir / "images").exists() or not (export_dir / "labels").exists():
        raise FileNotFoundError(
            f"{export_dir} must contain images/ and labels/ sub-folders."
        )

    log.info("Splitting into train/val/test …")
    yaml_path = split_dataset(source_dir=export_dir, cfg=config.data)
    log.info("Done. Use this file for training:\n  %s", yaml_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for YOLOv8 training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml")
    )
    parser.add_argument(
        "--yolo-export",
        type=Path,
        required=True,
        metavar="DIR",
        help='Path to YOLO export from Label Studio (must contain images/ and labels/).',
    )
    args = parser.parse_args()
    main(args)
