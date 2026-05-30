import logging
from pathlib import Path

import torch
from ultralytics import YOLO

from yodet.config import AugmentConfig, TrainConfig

log = logging.getLogger(__name__)


def train(dataset_yaml: Path, cfg: TrainConfig, aug: AugmentConfig) -> Path:

    log.info(
        "Training yolov8%s for %d epochs, imgsz=%d, batch=%d",
        cfg.model_variant, cfg.epochs, cfg.imgsz, cfg.batch,
    )

    model = YOLO(cfg.pretrained_weights)

    results = model.train(
        data=str(dataset_yaml.resolve()),
        name="yodet",
        device=0 if torch.cuda.is_available() else "cpu",
        mosaic=1.0,
        workers=2,
        **cfg.to_kwargs(),
        **aug.to_kwargs(),
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    log.info("Training complete. Best weights: %s", best_weights)

    return best_weights
