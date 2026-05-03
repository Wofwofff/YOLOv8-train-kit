import logging
from pathlib import Path

from ultralytics import YOLO

from yodet.config import AugmentConfig, TrainConfig

log = logging.getLogger(__name__)


def train(dataset_yaml: Path, cfg: TrainConfig, aug: AugmentConfig) -> Path:

    log.info(
        "Training yolov8%s for %d epochs, image_size=%d, batch=%d",
        cfg.model_variant, cfg.epochs, cfg.image_size, cfg.batch_size,
    )

    model = YOLO(cfg.pretrained_weights)

    results = model.train(
        data=str(dataset_yaml.resolve()),
        epochs=cfg.epochs,
        imgsz=cfg.image_size,
        batch=cfg.batch_size,
        patience=cfg.patience,
        name="yodet",
        # Rotation, flip, perspective
        degrees=aug.rotate_limit,
        fliplr=aug.flip_lr,
        perspective=aug.perspective,
        hsv_v=aug.brightness_limit,
        # Mosaic + MixUp + CopyPaste
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.1,
        copy_paste=0.1,
        workers=2,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    log.info("Training complete. Best weights: %s", best_weights)

    return best_weights
