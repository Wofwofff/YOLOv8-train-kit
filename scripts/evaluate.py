import argparse
import logging
import sys
from pathlib import Path

from yodet.config import Config
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main(args) -> None:
    cfg = Config.from_yaml(args.config)
    dataset_yaml = cfg.data.dataset_dir.resolve() / "dataset.yaml"

    if not dataset_yaml.exists():
        log.critical(f"dataset.yaml not found: {dataset_yaml}")
        sys.exit(1)

    model = YOLO(str(args.weights))

    log.info("Оценка качества модели на test выборке")
    _ = model.val(
        data=str(dataset_yaml),
        split="test",
        conf=0.3,
        iou=0.35,
        plots=True,
        save_json=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8 on test split"
    )
    parser.add_argument("--config",
                        type=Path,
                        default=Path("configs/default.yaml"))
    parser.add_argument("--weights",
                        type=Path,
                        required=True)
    args = parser.parse_args()
    main(args)
