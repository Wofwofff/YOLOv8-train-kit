from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    dataset_dir: Path = Path("data/dataset")
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    def __post_init__(self) -> None:
        self.dataset_dir = Path(self.dataset_dir)

    @property
    def test_ratio(self) -> float:
        return round(1.0 - self.train_ratio - self.val_ratio, 10)


@dataclass
class AugmentConfig:
    rotate_limit: int = 15
    brightness_limit: float = 0.5
    saturation_limit: float = 0.5
    flip_lr: float = 0.5
    flip_ud: float = 0.3
    perspective: float = 0.0003
    shear: float = 10.0
    scale: float = 0.6
    erasing: float = 0.4


@dataclass
class TrainConfig:
    model_variant: str = "m"
    epochs: int = 100
    image_size: int = 1280
    batch_size: int = 8
    patience: int = 50
    cos_lr: bool = True
    label_smoothing: float = 0.1
    multi_scale: bool = True
    close_mosaic: int = 20

    @property
    def pretrained_weights(self) -> str:
        return f"yolov8{self.model_variant}.pt"


@dataclass
class InferenceConfig:
    confidence: float = 0.30
    iou_threshold: float = 0.35
    use_slicing: bool = True
    slice_size: int = 640
    overlap_ratio: float = 0.20


@dataclass
class Config:
    data: DataConfig
    augment: AugmentConfig
    train: TrainConfig
    inference: InferenceConfig

    @classmethod
    def from_yaml(cls, path: Path | str) -> 'Config':
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            data=DataConfig(**raw.get("data", {})),
            augment=AugmentConfig(**raw.get("augment", {})),
            train=TrainConfig(**raw.get("train", {})),
            inference=InferenceConfig(**raw.get("inference", {})),
        )
