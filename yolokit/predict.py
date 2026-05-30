import logging
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from yolokit.config import InferenceConfig

log = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def predict(source: Path,
            weights: Path,
            output_dir: Path,
            cfg: InferenceConfig) -> None:
    """Run detection on `source` (image or directory) and save annotated results.

    Two modes:
    - Standard (use_slicing=False): fast, suitable for images ≤1280px.
    - Sliced / SAHI  (use_slicing=True): splits large images into overlapping
        tiles before inference, then merges results. Recommended for high-res
        tunnel scans with small or dense defects.

    Args:
        source:     Path to a single image or a directory of images.
        weights:    Path to best.pt from training.
        output_dir: Directory where annotated images will be saved.
        cfg:        Inference parameters (confidence, IOU, slicing settings).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.use_slicing:
        _predict_sliced(source, weights, output_dir, cfg)
    else:
        _predict_standard(source, weights, output_dir, cfg)


def _predict_standard(source: Path,
                      weights: Path,
                      output_dir: Path,
                      cfg: InferenceConfig) -> None:

    log.info("Standard inference on %s", source)
    model = YOLO(str(weights))
    model.predict(
        source=str(source),
        conf=cfg.confidence,
        iou=cfg.iou_threshold,
        save=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
    )
    log.info("Results saved to %s", output_dir)


def _predict_sliced(source: Path,
                    weights: Path,
                    output_dir: Path,
                    cfg: InferenceConfig) -> None:
    """SAHI sliced inference for small/dense defects in large images"""

    log.info("Sliced inference (SAHI) on %s, slice=%d, overlap=%.2f",
             source, cfg.slice_size, cfg.overlap_ratio)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(weights),
        confidence_threshold=cfg.confidence,
    )

    image_paths = _collect_images(source)

    for img_path in image_paths:
        result = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=cfg.slice_size,
            slice_width=cfg.slice_size,
            overlap_height_ratio=cfg.overlap_ratio,
            overlap_width_ratio=cfg.overlap_ratio,
            postprocess_match_threshold=cfg.iou_threshold,
        )

        out_path = output_dir / img_path.name
        result.export_visuals(export_dir=str(output_dir), file_name=img_path.stem)
        log.info("  %s → %d detections", img_path.name, len(result.object_prediction_list))

    log.info("Results saved to %s", output_dir)


def _collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
