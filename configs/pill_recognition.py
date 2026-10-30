"""알약 탐지·분류 통합 추론 설정과 YAML 로더."""

from dataclasses import dataclass
from pathlib import Path

from configs.yaml_loading import (
    load_task_yaml,
    validate_boolean,
    validate_device_name,
    validate_factory_path,
    validate_keys,
    validate_positive_integer,
)
from pilldata.validation import normalize_probability


@dataclass(frozen=True, slots=True)
class PillRecognitionConfig:
    """탐지기와 분류기를 연결하는 통합 추론 설정."""

    inference_factory_path: str | None
    input_dir: Path
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
        validate_factory_path(
            self.inference_factory_path,
            name="inference_factory_path",
        )
        for field_name in (
            "detector_checkpoint_path",
            "classifier_checkpoint_path",
            "detector_label_map_path",
            "classifier_class_map_path",
            "output_dir",
            "input_dir",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))
        if not isinstance(self.detector_name, str) or not self.detector_name.strip():
            raise ValueError("detector_name must be a non-empty string")
        validate_positive_integer(self.batch_size, name="batch_size")
        normalize_probability(self.score_threshold, name="score_threshold")
        validate_boolean(self.show_progress, name="show_progress")
        validate_device_name(self.device)


def load_pill_recognition_config(path: Path) -> PillRecognitionConfig:
    """통합 추론 YAML을 읽고 경로와 실행 옵션을 검증한다."""

    raw = load_task_yaml(path, expected_task="pill_recognition")
    allowed = {
        "task",
        "inference_factory_path",
        "input_dir",
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
    validate_keys(
        raw,
        allowed=allowed,
        required={
            "task",
            "inference_factory_path",
            "input_dir",
            "detector_checkpoint_path",
            "classifier_checkpoint_path",
            "detector_label_map_path",
            "classifier_class_map_path",
            "output_dir",
            "detector_name",
        },
        context="pill recognition config",
    )
    values = {key: value for key, value in raw.items() if key != "task"}
    return PillRecognitionConfig(**values)
