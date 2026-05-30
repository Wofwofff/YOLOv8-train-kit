import argparse
import logging
import sys
from pathlib import Path

from yolokit.config import Config
from yolokit.train import train

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main(args) -> None:
    cfg = Config.from_yaml(args.config)

    if not args.dataset.exists():
        log.critical(f"dataset.yaml not found: {args.dataset}")
        log.warning("Run prepare_dataset.py first.")
        sys.exit(1)

    log.info("Starting to train...")
    best_weights = train(dataset_yaml=args.dataset, cfg=cfg.train, aug=cfg.augment)
    log.info("Training complete!")
    log.info("Best weights saved to: %s", best_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on dataset"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml")
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/dataset/dataset.yaml"),
        help="Path to dataset.yaml (produced by prepare_dataset.py).",
    )

    args: argparse.Namespace = parser.parse_args()

    main(args)
