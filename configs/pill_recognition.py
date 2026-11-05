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

DETECTOR_BACKENDS = {"torchvision", "ultralytics"}


@dataclass(frozen=True, slots=True)
class PillRecognitionConfig:
    """탐지기와 분류기를 연결하는 통합 추론 설정."""

    run_name: str
    inference_factory_path: str | None
    input_dir: Path
    detector_run_dir: Path
    classifier_run_dir: Path
    output_dir: Path
    detector_backend: str = "torchvision"
    detector_image_size: int = 640
    detector_nms_threshold: float = 0.45
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
            "detector_run_dir",
            "classifier_run_dir",
            "output_dir",
            "input_dir",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))
        for field_name in ("detector_run_dir", "classifier_run_dir"):
            if not getattr(self, field_name).name:
                raise ValueError(f"{field_name} must identify a run directory")
        if not isinstance(self.detector_backend, str):
            raise TypeError("detector_backend must be a string")
        normalized_backend = self.detector_backend.strip().lower()
        if normalized_backend not in DETECTOR_BACKENDS:
            choices = ", ".join(sorted(DETECTOR_BACKENDS))
            raise ValueError(f"detector_backend must be one of: {choices}")
        object.__setattr__(self, "detector_backend", normalized_backend)
        validate_positive_integer(
            self.detector_image_size,
            name="detector_image_size",
        )
        normalize_probability(
            self.detector_nms_threshold,
            name="detector_nms_threshold",
        )
        validate_positive_integer(self.batch_size, name="batch_size")
        normalize_probability(self.score_threshold, name="score_threshold")
        validate_boolean(self.show_progress, name="show_progress")
        validate_device_name(self.device)

    @property
    def run_dir(self) -> Path:
        """작업별 출력 루트 아래의 현재 통합 추론 실행 폴더를 반환한다."""

        return self.output_dir / self.run_name

    @property
    def detector_checkpoint_path(self) -> Path:
        if self.detector_backend == "ultralytics":
            return self.detector_run_dir / "weights" / "best.pt"
        return self.detector_run_dir / "best.pt"

    @property
    def classifier_checkpoint_path(self) -> Path:
        return self.classifier_run_dir / "best.pt"

    @property
    def detector_label_map_path(self) -> Path:
        return self.detector_run_dir / "label_map.json"

    @property
    def classifier_class_map_path(self) -> Path:
        return self.classifier_run_dir / "class_map.json"

    @property
    def detector_name(self) -> str:
        return self.detector_run_dir.name

    def to_dict(self) -> dict[str, Any]:
        """JSON으로 저장할 수 있는 설정 사본을 반환한다."""

        values = asdict(self)
        for field_name in (
            "input_dir",
            "detector_run_dir",
            "classifier_run_dir",
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
