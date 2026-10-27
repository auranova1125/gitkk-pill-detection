"""분류기 학습 설정."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainConfig:
    """분류기 학습에 필요한 재현 가능한 설정을 보관한다."""

    run_name: str
    output_dir: Path
    class_map_path: Path
    data_summary_path: Path
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
