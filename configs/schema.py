"""작업별 YAML을 기존 모델 config와 실행 설정으로 변환한다."""

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from classifier.config import TrainConfig
from detector.config import DetectorConfig
from pilldata.validation import normalize_probability

_DEVICE_NAMES = {"auto", "cpu", "mps", "cuda"}


@dataclass(frozen=True, slots=True)
class ClassifierExperimentConfig:
    """분류기 YAML에서 읽은 학습 및 실행 설정."""

    training: TrainConfig
    data_factory_path: str | None
    expected_num_classes: int = 118
    pretrained: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        _validate_factory_path(self.data_factory_path)
        _validate_positive_integer(
            self.expected_num_classes,
            name="expected_num_classes",
        )
        _validate_boolean(self.pretrained, name="pretrained")
        _validate_device_name(self.device)


@dataclass(frozen=True, slots=True)
class DetectorExperimentConfig:
    """탐지기 YAML에서 읽은 학습 및 실행 설정."""

    training: DetectorConfig
    data_factory_path: str | None
    pretrained_backbone: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        _validate_factory_path(self.data_factory_path)
        _validate_boolean(self.pretrained_backbone, name="pretrained_backbone")
        _validate_device_name(self.device)


@dataclass(frozen=True, slots=True)
class PipelineExperimentConfig:
    """통합 추론 YAML에서 읽은 checkpoint, loader와 출력 설정."""

    inference_factory_path: str | None
    detector_checkpoint_path: Path
    classifier_checkpoint_path: Path
    detector_label_map_path: Path
    classifier_class_map_path: Path
    output_dir: Path
    detector_name: str
    batch_size: int = 4
    score_threshold: float = 0.25
    show_progress: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        _validate_factory_path(self.inference_factory_path)
        for field_name in (
            "detector_checkpoint_path",
            "classifier_checkpoint_path",
            "detector_label_map_path",
            "classifier_class_map_path",
            "output_dir",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))
        if not isinstance(self.detector_name, str) or not self.detector_name.strip():
            raise ValueError("detector_name must be a non-empty string")
        _validate_positive_integer(self.batch_size, name="batch_size")
        normalize_probability(self.score_threshold, name="score_threshold")
        _validate_boolean(self.show_progress, name="show_progress")
        _validate_device_name(self.device)


def load_classifier_experiment_config(path: Path) -> ClassifierExperimentConfig:
    """분류기 YAML을 읽고 기존 TrainConfig까지 검증한다."""

    raw = _load_task_yaml(path, expected_task="classifier")
    _validate_keys(
        raw,
        allowed={
            "task",
            "training",
            "data_factory_path",
            "expected_num_classes",
            "pretrained",
            "device",
        },
        required={"task", "training", "data_factory_path"},
        context="classifier config",
    )
    training = _build_model_config(
        TrainConfig,
        raw["training"],
        context="classifier config.training",
    )
    return ClassifierExperimentConfig(
        training=training,
        data_factory_path=raw["data_factory_path"],
        expected_num_classes=raw.get("expected_num_classes", 118),
        pretrained=raw.get("pretrained", True),
        device=raw.get("device", "auto"),
    )


def load_detector_experiment_config(path: Path) -> DetectorExperimentConfig:
    """탐지기 YAML을 읽고 기존 DetectorConfig까지 검증한다."""

    raw = _load_task_yaml(path, expected_task="detector")
    _validate_keys(
        raw,
        allowed={
            "task",
            "training",
            "data_factory_path",
            "pretrained_backbone",
            "device",
        },
        required={"task", "training", "data_factory_path"},
        context="detector config",
    )
    training = _build_model_config(
        DetectorConfig,
        raw["training"],
        context="detector config.training",
    )
    return DetectorExperimentConfig(
        training=training,
        data_factory_path=raw["data_factory_path"],
        pretrained_backbone=raw.get("pretrained_backbone", True),
        device=raw.get("device", "auto"),
    )


def load_pipeline_experiment_config(path: Path) -> PipelineExperimentConfig:
    """통합 추론 YAML을 읽고 경로와 실행 옵션을 검증한다."""

    raw = _load_task_yaml(path, expected_task="pipeline")
    allowed = {
        "task",
        "inference_factory_path",
        "detector_checkpoint_path",
        "classifier_checkpoint_path",
        "detector_label_map_path",
        "classifier_class_map_path",
        "output_dir",
        "detector_name",
        "batch_size",
        "score_threshold",
        "show_progress",
        "device",
    }
    _validate_keys(
        raw,
        allowed=allowed,
        required={
            "task",
            "inference_factory_path",
            "detector_checkpoint_path",
            "classifier_checkpoint_path",
            "detector_label_map_path",
            "classifier_class_map_path",
            "output_dir",
            "detector_name",
        },
        context="pipeline config",
    )
    values = {key: value for key, value in raw.items() if key != "task"}
    return PipelineExperimentConfig(**values)


def _load_task_yaml(path: Path, *, expected_task: str) -> dict[str, Any]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("config root must be a mapping")
    if any(not isinstance(key, str) for key in raw):
        raise TypeError("config keys must be strings")
    values = dict(raw)
    if values.get("task") != expected_task:
        raise ValueError(f"config task must be {expected_task!r}")
    return values


def _build_model_config[ConfigT: (TrainConfig, DetectorConfig)](
    config_type: type[ConfigT],
    value: Any,
    *,
    context: str,
) -> ConfigT:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    allowed = {field.name for field in fields(config_type)}
    required = {
        field.name
        for field in fields(config_type)
        if field.default is MISSING and field.default_factory is MISSING
    }
    _validate_keys(value, allowed=allowed, required=required, context=context)
    return config_type(**dict(value))


def _validate_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    context: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {unknown}")
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}")


def _validate_factory_path(value: str | None) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError("factory path must be a string or null")
    if not value.strip():
        raise ValueError("factory path must not be empty; use null when unavailable")
    module_name, separator, function_name = value.partition(":")
    if separator != ":" or not module_name or not function_name or ":" in function_name:
        raise ValueError("factory path must use 'module:function' format")


def _validate_positive_integer(value: int, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _validate_boolean(value: bool, *, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _validate_device_name(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("device must be a string")
    if value.strip().lower() not in _DEVICE_NAMES:
        choices = ", ".join(sorted(_DEVICE_NAMES))
        raise ValueError(f"device must be one of: {choices}")
