"""탐지기 optimizer, epoch 학습과 전체 학습 실행."""

import csv
import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from detector.checkpoint import (
    hash_label_map,
    load_checkpoint,
    restore_checkpoint_state,
    save_checkpoint,
    validate_checkpoint_compatibility,
)
from detector.config import DetectorConfig
from detector.input_validation import (
    validate_detector_data_summary,
    validate_detector_label_map,
)
from detector.model import build_detector_model, select_device


@dataclass(frozen=True, slots=True)
class TrainingEpochResult:
    """한 training epoch의 평균 loss와 처리량."""

    loss: float
    loss_components: dict[str, float]
    num_batches: int
    num_images: int

    def to_dict(self) -> dict[str, Any]:
        """직렬화 가능한 dict로 변환한다."""

        return asdict(self)


@dataclass(slots=True)
class EarlyStopping:
    """고 IoU mAP와 recall을 기준으로 best 및 조기 종료 상태를 관리한다."""

    patience: int
    best_epoch: int = -1
    best_map_75_95: float = float("-inf")
    best_mean_average_recall: float = float("-inf")
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.patience, int) or isinstance(self.patience, bool):
            raise TypeError("patience must be an integer")
        if self.patience <= 0:
            raise ValueError("patience must be greater than zero")

    def update(self, epoch: int, evaluation: Any) -> bool:
        """평가 결과로 best 상태를 갱신하고 개선 여부를 반환한다."""

        map_75_95 = float(evaluation.map_75_95)
        mean_average_recall = float(evaluation.mean_average_recall)
        if not isfinite(map_75_95) or not isfinite(mean_average_recall):
            raise ValueError("early stopping metrics must be finite")

        improved = map_75_95 > self.best_map_75_95 or (
            map_75_95 == self.best_map_75_95
            and mean_average_recall > self.best_mean_average_recall
        )
        if improved:
            self.best_epoch = epoch
            self.best_map_75_95 = map_75_95
            self.best_mean_average_recall = mean_average_recall
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """전체 학습의 checkpoint, best 지표와 epoch 기록."""

    run_dir: Path
    last_checkpoint_path: Path
    best_checkpoint_path: Path
    completed_epoch: int
    best_epoch: int
    best_map_75_95: float
    best_mean_average_recall: float
    stopped_early: bool
    metrics: tuple[dict[str, Any], ...]


def build_optimizer(model: nn.Module, config: DetectorConfig) -> SGD:
    """현재 학습 가능한 파라미터로 SGD optimizer를 생성한다."""

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("model must have at least one trainable parameter")
    return SGD(
        trainable_parameters,
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def build_scheduler(
    optimizer: Optimizer,
    config: DetectorConfig,
) -> CosineAnnealingLR:
    """전체 epoch 수를 주기로 하는 cosine scheduler를 생성한다."""

    return CosineAnnealingLR(optimizer, T_max=config.max_epochs)


def build_grad_scaler(device: torch.device) -> torch.amp.GradScaler:
    """CUDA에서만 활성화되는 AMP GradScaler를 생성한다."""

    return torch.amp.GradScaler(device.type, enabled=device.type == "cuda")


def set_seed(seed: int) -> None:
    """Python, NumPy와 PyTorch random seed를 설정한다."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def move_detection_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    """이미지와 target tensor를 선택한 장치로 이동한다."""

    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping")
    missing_fields = {"images", "targets"} - batch.keys()
    if missing_fields:
        raise ValueError(f"batch is missing required fields: {sorted(missing_fields)}")

    images = batch["images"]
    targets = batch["targets"]
    if not isinstance(images, list) or not isinstance(targets, list):
        raise TypeError("batch images and targets must be lists")
    if not images or len(images) != len(targets):
        raise ValueError("batch images and targets must have equal non-zero lengths")

    non_blocking = device.type == "cuda"
    moved_images: list[torch.Tensor] = []
    moved_targets: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"batch.images[{index}] must be a torch.Tensor")
        moved_images.append(image.to(device, non_blocking=non_blocking))

    for index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise TypeError(f"batch.targets[{index}] must be a mapping")
        moved_targets.append(
            {
                key: value.to(device, non_blocking=non_blocking)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in target.items()
            }
        )
    return moved_images, moved_targets


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    optimizer: Optimizer,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None = None,
    show_progress: bool = False,
    progress_description: str = "Detector training",
) -> TrainingEpochResult:
    """DataLoader를 한 번 순회해 탐지 모델을 학습한다."""

    model.train()
    use_amp = device.type == "cuda"
    active_scaler = scaler if scaler is not None else build_grad_scaler(device)
    if active_scaler.is_enabled() != use_amp:
        raise ValueError("GradScaler enabled state must match CUDA usage")
    if not isinstance(show_progress, bool):
        raise TypeError("show_progress must be a boolean")
    if not isinstance(progress_description, str) or not progress_description.strip():
        raise ValueError("progress_description must be a non-empty string")

    total_loss = 0.0
    component_totals: dict[str, float] = {}
    expected_loss_names: set[str] | None = None
    num_batches = 0
    num_images = 0

    progress = tqdm(
        data_loader,
        desc=progress_description,
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for batch in progress:
        images, targets = move_detection_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            raw_losses = model(images, targets)
            losses = _validate_losses(raw_losses)
            current_loss_names = set(losses)
            if expected_loss_names is None:
                expected_loss_names = current_loss_names
            elif current_loss_names != expected_loss_names:
                raise ValueError("model loss component names must remain consistent")
            loss = sum(losses.values())

        if active_scaler.is_enabled():
            active_scaler.scale(loss).backward()
            active_scaler.step(optimizer)
            active_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().item())
        for name, value in losses.items():
            component_totals[name] = component_totals.get(name, 0.0) + float(
                value.detach().item()
            )
        num_batches += 1
        num_images += len(images)
        progress.set_postfix(loss=f"{total_loss / num_batches:.4f}")

    if num_batches == 0:
        raise ValueError("data_loader must contain at least one batch")

    return TrainingEpochResult(
        loss=total_loss / num_batches,
        loss_components={
            name: total / num_batches
            for name, total in sorted(component_totals.items())
        },
        num_batches=num_batches,
        num_images=num_images,
    )


def run_training(
    config: DetectorConfig,
    train_loader: Iterable[Mapping[str, Any]],
    val_loader: Iterable[Mapping[str, Any]],
    label_map: Mapping[str, Any],
    data_summary: Mapping[str, Any],
    *,
    pretrained_backbone: bool = True,
    device: torch.device | None = None,
) -> TrainingResult:
    """설정 검증부터 checkpoint 저장까지 detector 학습을 실행한다."""

    from detector.evaluate import evaluate_model

    validate_detector_label_map(label_map)
    validate_detector_data_summary(data_summary, label_map)
    if not isinstance(pretrained_backbone, bool):
        raise TypeError("pretrained_backbone must be a boolean")

    selected_device = device if device is not None else select_device()
    set_seed(config.seed)
    model_config = {
        "num_classes": label_map["num_model_classes"],
        "pretrained_backbone": pretrained_backbone,
        "nms_threshold": config.nms_threshold,
    }
    model = build_detector_model(
        config.model_name,
        num_classes=model_config["num_classes"],
        pretrained_backbone=pretrained_backbone,
        nms_threshold=config.nms_threshold,
    ).to(selected_device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = build_grad_scaler(selected_device)
    early_stopping = EarlyStopping(config.early_stopping_patience)

    run_dir = config.output_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = run_dir / "last.pt"
    best_checkpoint_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.csv"
    _write_json(run_dir / "resolved_config.json", config.to_dict())
    _write_json(run_dir / "label_map.json", label_map)
    _write_json(run_dir / "data_summary.json", data_summary)
    if config.resume_from is None:
        metrics_path.write_text("", encoding="utf-8")

    label_map_hash = hash_label_map(label_map)
    split_hash = data_summary["split_hash"]
    start_epoch = 0
    if config.resume_from is not None:
        checkpoint = load_checkpoint(config.resume_from, selected_device)
        validate_checkpoint_compatibility(
            checkpoint,
            expected_model_name=config.model_name,
            expected_model_config=model_config,
            expected_label_map_hash=label_map_hash,
            expected_split_hash=split_hash,
        )
        restore_checkpoint_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        start_epoch = checkpoint["completed_epoch"] + 1
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            best_epoch=checkpoint["best_epoch"],
            best_map_75_95=checkpoint["best_map_75_95"],
            best_mean_average_recall=checkpoint["best_mean_average_recall"],
            epochs_without_improvement=checkpoint["epochs_without_improvement"],
        )
    if start_epoch >= config.max_epochs:
        raise ValueError("checkpoint already reached the configured max_epochs")

    metric_rows: list[dict[str, Any]] = []
    completed_epoch = start_epoch - 1
    stopped_early = False
    for epoch in range(start_epoch, config.max_epochs):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        training_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            selected_device,
            scaler=scaler,
            show_progress=config.show_progress,
            progress_description=f"Detector train {epoch + 1}/{config.max_epochs}",
        )
        evaluation_result = evaluate_model(
            model,
            val_loader,
            selected_device,
            score_threshold=config.score_threshold,
            show_progress=config.show_progress,
            progress_description=(
                f"Detector validation {epoch + 1}/{config.max_epochs}"
            ),
        )
        improved = early_stopping.update(epoch, evaluation_result)
        scheduler.step()
        completed_epoch = epoch

        metric_row = _build_metric_row(
            epoch=epoch,
            learning_rate=learning_rate,
            training_result=training_result,
            evaluation_result=evaluation_result,
        )
        metric_rows.append(metric_row)
        _append_metric_row(metrics_path, metric_row)

        checkpoint_arguments = {
            "completed_epoch": epoch,
            "model_name": config.model_name,
            "model_config": model_config,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "best_epoch": early_stopping.best_epoch,
            "best_map_75_95": early_stopping.best_map_75_95,
            "best_mean_average_recall": early_stopping.best_mean_average_recall,
            "epochs_without_improvement": early_stopping.epochs_without_improvement,
            "label_map_hash": label_map_hash,
            "split_hash": split_hash,
        }
        save_checkpoint(last_checkpoint_path, **checkpoint_arguments)
        if improved:
            save_checkpoint(best_checkpoint_path, **checkpoint_arguments)
        if early_stopping.should_stop:
            stopped_early = True
            break

    return TrainingResult(
        run_dir=run_dir,
        last_checkpoint_path=last_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        completed_epoch=completed_epoch,
        best_epoch=early_stopping.best_epoch,
        best_map_75_95=early_stopping.best_map_75_95,
        best_mean_average_recall=early_stopping.best_mean_average_recall,
        stopped_early=stopped_early,
        metrics=tuple(metric_rows),
    )


def _build_metric_row(
    *,
    epoch: int,
    learning_rate: float,
    training_result: TrainingEpochResult,
    evaluation_result: Any,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_loss": training_result.loss,
        "train_loss_components": json.dumps(
            training_result.loss_components,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "train_num_images": training_result.num_images,
        **evaluation_result.to_dict(),
    }


def _append_metric_row(path: Path, row: Mapping[str, Any]) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_losses(value: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError("model must return a non-empty loss mapping in training mode")
    for name, loss in value.items():
        if not isinstance(name, str) or not name:
            raise TypeError("model loss names must be non-empty strings")
        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"model loss {name!r} must be a torch.Tensor")
        if loss.ndim != 0:
            raise ValueError(f"model loss {name!r} must be a scalar")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"model loss {name!r} must be finite")
    return value
