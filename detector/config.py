"""탐지기 학습 설정."""

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any


@dataclass(slots=True, kw_only=True)
class DetectorConfig:
    """탐지기 학습과 추론에 공통으로 사용하는 설정을 보관한다."""

    run_name: str
    output_dir: Path
    label_map_path: Path
    data_summary_path: Path
    learning_rate: float
    weight_decay: float
    score_threshold: float
    nms_threshold: float
    early_stopping_patience: int
    model_name: str = "ssdlite320_mobilenet_v3_large"
    seed: int = 42
    batch_size: int = 8
    max_epochs: int = 50
    momentum: float = 0.9
    resume_from: Path | None = None
    show_progress: bool = True

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.label_map_path = Path(self.label_map_path)
        self.data_summary_path = Path(self.data_summary_path)
        if self.resume_from is not None:
            self.resume_from = Path(self.resume_from)

        if not self.run_name.strip():
            raise ValueError("run_name must not be empty")
        if Path(self.run_name).name != self.run_name:
            raise ValueError("run_name must not contain path separators")
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be greater than zero")
        self._validate_finite_number(
            self.learning_rate,
            name="learning_rate",
            minimum=0.0,
            minimum_inclusive=False,
        )
        self._validate_finite_number(
            self.momentum,
            name="momentum",
            minimum=0.0,
            maximum=1.0,
            maximum_inclusive=False,
        )
        self._validate_finite_number(
            self.weight_decay,
            name="weight_decay",
            minimum=0.0,
        )
        self._validate_finite_number(
            self.score_threshold,
            name="score_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        self._validate_finite_number(
            self.nms_threshold,
            name="nms_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be greater than zero")
        if not isinstance(self.show_progress, bool):
            raise TypeError("show_progress must be a boolean")

    @staticmethod
    def _validate_finite_number(
        value: float,
        *,
        name: str,
        minimum: float,
        maximum: float | None = None,
        minimum_inclusive: bool = True,
        maximum_inclusive: bool = True,
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

        if maximum is None:
            return
        above_maximum = value > maximum if maximum_inclusive else value >= maximum
        if above_maximum:
            comparison = "less than or equal to" if maximum_inclusive else "less than"
            raise ValueError(f"{name} must be {comparison} {maximum}")

    def to_dict(self) -> dict[str, Any]:
        """JSON으로 저장할 수 있는 설정 사본을 반환한다."""

        values = asdict(self)
        for field_name in (
            "output_dir",
            "label_map_path",
            "data_summary_path",
            "resume_from",
        ):
            value = values[field_name]
            values[field_name] = str(value) if value is not None else None
        return values
