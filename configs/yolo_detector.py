"""Ultralytics YOLO 탐지기 학습 설정과 YAML 로더."""

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from configs.yaml_loading import (
    build_dataclass,
    load_task_yaml,
    validate_boolean,
    validate_device_name,
    validate_positive_integer,
    validate_run_name,
)


@dataclass(slots=True, kw_only=True)
class YoloDetectorConfig:
    """단일 pill 클래스 YOLO 탐지기의 학습 설정."""

    run_name: str
    output_dir: Path
    data_dir: Path = Path("data/detector")
    model_name: str = "yolo26s.pt"
    device: str = "auto"
    seed: int = 42
    image_size: int = 640
    batch_size: int = 4
    max_epochs: int = 30
    early_stopping_patience: int = 7
    workers: int = 0
    cache_images: bool = False
    optimizer: str = "auto"
    initial_learning_rate: float | None = None
    weight_decay: float = 0.0005
    score_threshold: float = 0.25
    nms_threshold: float = 0.45
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.5
    hsv_hue: float = 0.015
    hsv_saturation: float = 0.4
    hsv_value: float = 0.2
    translate: float = 0.1
    scale: float = 0.3
    mosaic: float = 0.5
    close_mosaic_epochs: int = 5
    deterministic: bool = True
    show_progress: bool = True

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.data_dir = Path(self.data_dir)

        validate_run_name(self.run_name)
        validate_device_name(self.device)
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if not isinstance(self.optimizer, str) or not self.optimizer.strip():
            raise ValueError("optimizer must not be empty")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name in (
            "image_size",
            "batch_size",
            "max_epochs",
            "early_stopping_patience",
        ):
            validate_positive_integer(getattr(self, name), name=name)
        if not isinstance(self.workers, int) or isinstance(self.workers, bool):
            raise TypeError("workers must be an integer")
        if self.workers < 0:
            raise ValueError("workers must be non-negative")
        if not isinstance(self.close_mosaic_epochs, int) or isinstance(
            self.close_mosaic_epochs, bool
        ):
            raise TypeError("close_mosaic_epochs must be an integer")
        if not 0 <= self.close_mosaic_epochs <= self.max_epochs:
            raise ValueError("close_mosaic_epochs must be between zero and max_epochs")

        if self.initial_learning_rate is not None:
            if self.optimizer.strip().lower() == "auto":
                raise ValueError(
                    "optimizer must be explicit when initial_learning_rate is set"
                )
            self._validate_number(
                self.initial_learning_rate,
                name="initial_learning_rate",
                minimum=0.0,
                minimum_inclusive=False,
            )
        self._validate_number(self.weight_decay, name="weight_decay", minimum=0.0)
        for name in (
            "score_threshold",
            "nms_threshold",
            "horizontal_flip",
            "vertical_flip",
            "hsv_hue",
            "hsv_saturation",
            "hsv_value",
            "translate",
            "scale",
            "mosaic",
        ):
            self._validate_number(
                getattr(self, name),
                name=name,
                minimum=0.0,
                maximum=1.0,
            )
        for name in ("cache_images", "deterministic", "show_progress"):
            validate_boolean(getattr(self, name), name=name)

    @staticmethod
    def _validate_number(
        value: float,
        *,
        name: str,
        minimum: float,
        maximum: float | None = None,
        minimum_inclusive: bool = True,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{name} must be a finite number")
        if not isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        below_minimum = value < minimum if minimum_inclusive else value <= minimum
        if below_minimum:
            comparison = (
                "greater than or equal to" if minimum_inclusive else "greater than"
            )
            raise ValueError(f"{name} must be {comparison} {minimum}")
        if maximum is not None and value > maximum:
            raise ValueError(f"{name} must be less than or equal to {maximum}")

    @property
    def run_dir(self) -> Path:
        """현재 YOLO 학습 실행 폴더를 반환한다."""

        return self.output_dir / self.run_name

    def to_dict(self) -> dict[str, Any]:
        """JSON으로 저장할 수 있는 설정 사본을 반환한다."""

        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        values["data_dir"] = str(self.data_dir)
        return values


def load_yolo_detector_config(path: Path) -> YoloDetectorConfig:
    """YOLO 탐지기 YAML을 실행 설정으로 변환한다."""

    raw = load_task_yaml(path, expected_task="yolo_detector")
    values = {key: value for key, value in raw.items() if key != "task"}
    return build_dataclass(
        YoloDetectorConfig,
        values,
        context="YOLO detector config",
    )
