"""분류기 checkpoint 저장과 복원."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from classifier.model import TrainingStage
from common.reproducibility import capture_rng_state, restore_rng_state

CHECKPOINT_VERSION = 1
_REQUIRED_CHECKPOINT_FIELDS = {
    "checkpoint_version",
    "completed_epoch",
    "training_stage",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "scaler_state",
    "best_epoch",
    "best_macro_f1",
    "best_top1_accuracy",
    "epochs_without_improvement",
    "rng_state",
    "class_map_hash",
    "split_hash",
}


def hash_class_map(class_map: Mapping[str, Any]) -> str:
    """키 순서와 무관한 class map SHA-256 hash를 계산한다."""

    canonical_json = json.dumps(
        class_map,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def save_checkpoint(
    path: Path,
    *,
    completed_epoch: int,
    training_stage: TrainingStage | str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    best_epoch: int,
    best_macro_f1: float,
    best_top1_accuracy: float,
    epochs_without_improvement: int,
    class_map_hash: str,
    split_hash: str,
) -> None:
    """학습 상태를 임시 파일에 쓴 뒤 원자적으로 checkpoint를 교체한다."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "completed_epoch": completed_epoch,
        "training_stage": TrainingStage(training_stage).value,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict()
        if scaler is not None and scaler.is_enabled()
        else None,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "best_top1_accuracy": best_top1_accuracy,
        "epochs_without_improvement": epochs_without_improvement,
        "rng_state": capture_rng_state(),
        "class_map_hash": class_map_hash,
        "split_hash": split_hash,
    }

    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """신뢰할 수 있는 프로젝트 checkpoint를 현재 장치로 읽는다."""

    checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a dictionary")

    missing_fields = sorted(_REQUIRED_CHECKPOINT_FIELDS - checkpoint.keys())
    if missing_fields:
        raise ValueError(f"checkpoint is missing fields: {missing_fields}")
    if checkpoint["checkpoint_version"] != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version: {checkpoint['checkpoint_version']!r}"
        )
    TrainingStage(checkpoint["training_stage"])
    return checkpoint


def validate_checkpoint_compatibility(
    checkpoint: Mapping[str, Any],
    *,
    expected_class_map_hash: str,
    expected_split_hash: str,
) -> None:
    """checkpoint의 클래스 매핑과 데이터 분할이 현재 실행과 같은지 확인한다."""

    if checkpoint["class_map_hash"] != expected_class_map_hash:
        raise ValueError("checkpoint class map hash does not match current class map")
    if checkpoint["split_hash"] != expected_split_hash:
        raise ValueError("checkpoint split hash does not match current data split")


def restore_checkpoint_state(
    checkpoint: Mapping[str, Any],
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    restore_optimizer: bool = True,
    restore_rng: bool = True,
) -> None:
    """model과 선택한 학습 상태를 checkpoint에서 복원한다."""

    model.load_state_dict(checkpoint["model_state"])
    if restore_optimizer:
        if optimizer is None:
            raise ValueError("optimizer is required when restore_optimizer is true")
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        scheduler_state = checkpoint["scheduler_state"]
        if scheduler_state is not None:
            if scheduler is None:
                raise ValueError("scheduler is required by the checkpoint")
            scheduler.load_state_dict(scheduler_state)
        elif scheduler is not None:
            raise ValueError("checkpoint does not contain scheduler state")

    scaler_state = checkpoint["scaler_state"]
    if scaler_state is not None and scaler is not None and scaler.is_enabled():
        scaler.load_state_dict(scaler_state)
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
