"""알약 탐지·분류 통합 추론 설정과 YAML 로더."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from configs.yaml_loading import (
    build_dataclass,
    load_task_yaml,
    validate_boolean,
    validate_device_name,
    validate_factory_path,
    validate_positive_integer,
    validate_run_name,
)
from data_loading.validation import normalize_probability


@dataclass(frozen=True, slots=True)
class PillRecognitionConfig:
    """탐지기와 분류기를 연결하는 통합 추론 설정."""

    run_name: str
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
        validate_run_name(self.run_name)
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

    @property
    def run_dir(self) -> Path:
        """작업별 출력 루트 아래의 현재 통합 추론 실행 폴더를 반환한다."""

        return self.output_dir / self.run_name

    def to_dict(self) -> dict[str, Any]:
        """JSON으로 저장할 수 있는 설정 사본을 반환한다."""

        values = asdict(self)
        for field_name in (
            "input_dir",
            "detector_checkpoint_path",
            "classifier_checkpoint_path",
            "detector_label_map_path",
            "classifier_class_map_path",
            "output_dir",
        ):
            values[field_name] = str(values[field_name])
        return values


def load_pill_recognition_config(path: Path) -> PillRecognitionConfig:
    """통합 추론 YAML을 읽고 경로와 실행 옵션을 검증한다."""

    raw = load_task_yaml(path, expected_task="pill_recognition")
    values = {key: value for key, value in raw.items() if key != "task"}
    return build_dataclass(
        PillRecognitionConfig,
        values,
        context="pill recognition config",
    )
