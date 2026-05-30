import logging
import random
import shutil
from pathlib import Path

import yaml

from yolokit.config import DataConfig

log = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def split_dataset(source_dir: Path, cfg: DataConfig, seed: int = 42) -> Path:
    """Split source_dir into train/val/test and write dataset.yaml.

    Args:
        source_dir: Directory with images/, labels/, and classes.txt.
        cfg:        DataConfig — provides dataset_dir and ratios.
        seed:       Random seed for reproducible splits.

    Returns:
        Path to the generated dataset.yaml file.
    """
    src_images = source_dir / "images"
    src_labels = source_dir / "labels"
    dataset_dir = cfg.dataset_dir.resolve()

    image_paths = sorted(
        p for p in src_images.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {src_images}")

    rng = random.Random(seed)
    rng.shuffle(image_paths)

    n = len(image_paths)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)

    splits = {
        "train": image_paths[:n_train],
        "val": image_paths[n_train : n_train + n_val],
        "test": image_paths[n_train + n_val :],
    }

    for split_name, paths in splits.items():
        (dataset_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        for img_path in paths:
            label_path = src_labels / (img_path.stem + ".txt")

            shutil.copy(img_path, dataset_dir / "images" / split_name / img_path.name)

            if label_path.exists():
                shutil.copy(
                    label_path,
                    dataset_dir / "labels" / split_name / label_path.name,
                )
            else:
                # Write empty label file so YOLO counts the image as a negative
                (dataset_dir / "labels" / split_name / (img_path.stem + ".txt")).touch()

        log.info("  %s: %d images", split_name, len(paths))

    classes = (source_dir / "classes.txt").read_text(encoding="utf-8").splitlines()
    yaml_path = _write_dataset_yaml(dataset_dir, classes, cfg)

    log.info(
        "Split complete: %d total → train=%d val=%d test=%d. YAML: %s",
        n, len(splits["train"]), len(splits["val"]), len(splits["test"]),
        yaml_path,
    )
    return yaml_path


def _write_dataset_yaml(dataset_dir: Path, classes: list[str], cfg: DataConfig) -> Path:
    """Write dataset.yaml with absolute paths (required by ultralytics)."""
    yaml_path = dataset_dir / "dataset.yaml"

    content = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": classes,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, allow_unicode=True, default_flow_style=False)

    return yaml_path
