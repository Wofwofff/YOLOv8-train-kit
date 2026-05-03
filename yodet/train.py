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
        device=0 if __import__("torch").cuda.is_available() else "cpu",
        # Augmentation
        degrees=aug.rotate_limit,
        fliplr=aug.flip_lr,
        flipud=aug.flip_ud,
        perspective=aug.perspective,
        shear=aug.shear,
        scale=aug.scale,
        erasing=aug.erasing,
        hsv_v=aug.brightness_limit,
        hsv_s=aug.saturation_limit,
        # Mosaic + MixUp + CopyPaste
        mosaic=1.0,
        close_mosaic=cfg.close_mosaic,
        mixup=0.1,
        copy_paste=0.1,
        # Training
        cos_lr=cfg.cos_lr,
        label_smoothing=cfg.label_smoothing,
        multi_scale=cfg.multi_scale,
        workers=2,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    log.info("Training complete. Best weights: %s", best_weights)

    return best_weights
