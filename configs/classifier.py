"""분류기 학습 실행 설정과 YAML 로더."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from configs.yaml_loading import (
    build_dataclass,
    load_task_yaml,
    validate_boolean,
    validate_device_name,
    validate_factory_path,
    validate_keys,
)


@dataclass(slots=True)
class ClassifierConfig:
    """분류기 데이터 준비부터 학습까지 필요한 설정."""

    run_name: str
    output_dir: Path
    class_map_path: Path
    data_summary_path: Path
    data_factory_path: str | None = None
    expected_num_classes: int = 118
    pretrained: bool = True
    device: str = "auto"
    seed: int = 42
    batch_size: int = 32
    max_epochs: int = 30
    head_only_epochs: int = 3
    lr_head_only: float = 1e-3
    lr_layer4: float = 1e-4
    lr_fc: float = 5e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 7
    resume_from: Path | None = None
    show_progress: bool = True

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.class_map_path = Path(self.class_map_path)
        self.data_summary_path = Path(self.data_summary_path)
        if self.resume_from is not None:
            self.resume_from = Path(self.resume_from)

        if not self.run_name.strip():
            raise ValueError("run_name must not be empty")
        if Path(self.run_name).name != self.run_name:
            raise ValueError("run_name must not contain path separators")
        validate_factory_path(self.data_factory_path, name="data_factory_path")
        if not isinstance(self.expected_num_classes, int) or isinstance(
            self.expected_num_classes, bool
        ):
            raise TypeError("expected_num_classes must be an integer")
        if self.expected_num_classes <= 0:
            raise ValueError("expected_num_classes must be greater than zero")
        validate_boolean(self.pretrained, name="pretrained")
        validate_device_name(self.device)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be greater than zero")
        if not 0 <= self.head_only_epochs <= self.max_epochs:
            raise ValueError("head_only_epochs must be between zero and max_epochs")
        if self.lr_head_only <= 0 or self.lr_layer4 <= 0 or self.lr_fc <= 0:
            raise ValueError("learning rates must be greater than zero")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be greater than zero")
        validate_boolean(self.show_progress, name="show_progress")

    def to_dict(self) -> dict[str, Any]:
        """JSON으로 저장할 수 있는 설정 사본을 반환한다."""

        values = asdict(self)
        for field_name in (
            "output_dir",
            "class_map_path",
            "data_summary_path",
            "resume_from",
        ):
            value = values[field_name]
            values[field_name] = str(value) if value is not None else None
        return values


def load_classifier_config(path: Path) -> ClassifierConfig:
    """분류기 YAML을 읽어 하나의 실행 설정으로 변환한다."""

    raw = load_task_yaml(path, expected_task="classifier")
    validate_keys(
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
    runtime_values = {
        "data_factory_path": raw["data_factory_path"],
        "expected_num_classes": raw.get("expected_num_classes", 118),
        "pretrained": raw.get("pretrained", True),
        "device": raw.get("device", "auto"),
    }
    return build_dataclass(
        ClassifierConfig,
        raw["training"],
        context="classifier config.training",
        extra_values=runtime_values,
    )
