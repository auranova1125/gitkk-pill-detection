"""탐지기 checkpoint 저장, 호환성 확인과 상태 복원."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torchvision
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from utils.runtime import capture_rng_state, restore_rng_state

CHECKPOINT_VERSION = 1
_REQUIRED_CHECKPOINT_FIELDS = {
    "checkpoint_version",
    "completed_epoch",
    "model_name",
    "model_config",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "scaler_state",
    "best_epoch",
    "best_map_75_95",
    "best_mean_average_recall",
    "epochs_without_improvement",
    "rng_state",
    "label_map_hash",
    "split_hash",
    "torchvision_version",
}


def hash_label_map(label_map: Mapping[str, Any]) -> str:
    """키 순서와 무관한 label map SHA-256 hash를 계산한다."""

    canonical_json = json.dumps(
        label_map,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def save_checkpoint(
    path: Path,
    *,
    completed_epoch: int,
    model_name: str,
    model_config: Mapping[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    best_epoch: int,
    best_map_75_95: float,
    best_mean_average_recall: float,
    epochs_without_improvement: int,
    label_map_hash: str,
    split_hash: str,
) -> None:
    """학습 상태를 임시 파일에 저장한 뒤 checkpoint를 원자적으로 교체한다."""

    _validate_save_metadata(
        completed_epoch=completed_epoch,
        model_name=model_name,
        model_config=model_config,
        best_epoch=best_epoch,
        epochs_without_improvement=epochs_without_improvement,
        label_map_hash=label_map_hash,
        split_hash=split_hash,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "completed_epoch": completed_epoch,
        "model_name": model_name,
        "model_config": dict(model_config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict()
        if scaler is not None and scaler.is_enabled()
        else None,
        "best_epoch": best_epoch,
        "best_map_75_95": float(best_map_75_95),
        "best_mean_average_recall": float(best_mean_average_recall),
        "epochs_without_improvement": epochs_without_improvement,
        "rng_state": capture_rng_state(),
        "label_map_hash": label_map_hash,
        "split_hash": split_hash,
        "torchvision_version": torchvision.__version__,
    }

    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """신뢰할 수 있는 프로젝트 checkpoint를 지정 장치로 읽는다."""

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
    if not isinstance(checkpoint["model_config"], Mapping):
        raise TypeError("checkpoint model_config must be a mapping")
    return checkpoint


def validate_checkpoint_compatibility(
    checkpoint: Mapping[str, Any],
    *,
    expected_model_name: str,
    expected_model_config: Mapping[str, Any],
    expected_label_map_hash: str,
    expected_split_hash: str,
) -> None:
    """checkpoint가 현재 모델과 데이터 설정으로 재개 가능한지 확인한다."""

    if checkpoint["model_name"] != expected_model_name:
        raise ValueError("checkpoint model name does not match current model")
    if dict(checkpoint["model_config"]) != dict(expected_model_config):
        raise ValueError("checkpoint model config does not match current model config")
    if checkpoint["label_map_hash"] != expected_label_map_hash:
        raise ValueError("checkpoint label map hash does not match current label map")
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
        if scaler_state is not None:
            if scaler is None or not scaler.is_enabled():
                raise ValueError("enabled GradScaler is required by the checkpoint")
            scaler.load_state_dict(scaler_state)
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])


def _validate_save_metadata(
    *,
    completed_epoch: int,
    model_name: str,
    model_config: Mapping[str, Any],
    best_epoch: int,
    epochs_without_improvement: int,
    label_map_hash: str,
    split_hash: str,
) -> None:
    integer_values = {
        "completed_epoch": completed_epoch,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
    }
    for name, value in integer_values.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if completed_epoch < 0:
        raise ValueError("completed_epoch must be non-negative")
    if not -1 <= best_epoch <= completed_epoch:
        raise ValueError("best_epoch must be between -1 and completed_epoch")
    if epochs_without_improvement < 0:
        raise ValueError("epochs_without_improvement must be non-negative")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(model_config, Mapping) or not model_config:
        raise ValueError("model_config must be a non-empty mapping")
    for name, value in (
        ("label_map_hash", label_map_hash),
        ("split_hash", split_hash),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
