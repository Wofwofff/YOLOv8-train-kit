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


class _ParamsConfig:
    def __init__(self, **kwargs: object) -> None:
        self._params = kwargs

    def __getattr__(self, name: str) -> object:
        try:
            return self._params[name]
        except KeyError:
            raise AttributeError(name)

    def to_kwargs(self) -> dict:
        return dict(self._params)


class AugmentConfig(_ParamsConfig):
    pass


class TrainConfig(_ParamsConfig):
    @property
    def pretrained_weights(self) -> str:
        return f"yolov8{self._params['model_variant']}.pt"

    def to_kwargs(self) -> dict:
        return {k: v for k, v in self._params.items() if k != "model_variant"}


class InferenceConfig(_ParamsConfig):
    pass


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
