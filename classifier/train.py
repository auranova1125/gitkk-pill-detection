"""분류기 optimizer 구성과 전체 학습 실행."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torchvision.models.resnet import ResNet
from tqdm.auto import tqdm

from classifier.checkpoint import (
    hash_class_map,
    load_checkpoint,
    restore_checkpoint_state,
    save_checkpoint,
    validate_checkpoint_compatibility,
)
from classifier.evaluate import EvaluationResult, evaluate_model
from classifier.input_validation import validate_class_map, validate_data_summary
from classifier.model import (
    TrainingStage,
    build_model,
    configure_model_for_stage,
    set_model_train_mode,
)
from classifier.training_data import ClassifierTrainingData, prepare_classifier_data
from configs.classifier import ClassifierConfig
from utils.io import append_csv_row, write_json
from utils.runtime import build_grad_scaler, select_device, set_seed


@dataclass(frozen=True, slots=True)
class TrainingEpochResult:
    """한 training epoch의 집계 지표."""

    loss: float
    top1_accuracy: float
    top5_accuracy: float
    num_samples: int


@dataclass(slots=True)
class EarlyStopping:
    """macro F1과 Top-1 기준의 best 선택 및 조기 종료 상태."""

    patience: int
    best_epoch: int = -1
    best_macro_f1: float = float("-inf")
    best_top1_accuracy: float = float("-inf")
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be greater than zero")

    def update(
        self,
        epoch: int,
        result: EvaluationResult,
        *,
        count_for_stopping: bool,
    ) -> bool:
        """best 상태를 갱신하고 현재 결과의 개선 여부를 반환한다."""

        improved = result.macro_f1 > self.best_macro_f1 or (
            result.macro_f1 == self.best_macro_f1
            and result.top1_accuracy > self.best_top1_accuracy
        )
        if improved:
            self.best_epoch = epoch
            self.best_macro_f1 = result.macro_f1
            self.best_top1_accuracy = result.top1_accuracy
            self.epochs_without_improvement = 0
        elif count_for_stopping:
            self.epochs_without_improvement += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """전체 학습 실행 결과와 생성된 artifact 위치."""

    run_dir: Path
    last_checkpoint_path: Path
    best_checkpoint_path: Path
    completed_epoch: int
    best_epoch: int
    best_macro_f1: float
    best_top1_accuracy: float
    stopped_early: bool
    metrics: tuple[dict[str, Any], ...]


def resolve_training_stage(epoch: int, head_only_epochs: int) -> TrainingStage:
    """현재 epoch가 head-only 단계인지 layer4 fine-tuning 단계인지 결정합니다."""

    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise TypeError("epoch must be an integer")
    if not isinstance(head_only_epochs, int) or isinstance(head_only_epochs, bool):
        raise TypeError("head_only_epochs must be an integer")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if head_only_epochs < 0:
        raise ValueError("head_only_epochs must be non-negative")
    if epoch < head_only_epochs:
        return TrainingStage.HEAD_ONLY
    return TrainingStage.LAYER4_FINE_TUNING


def build_optimizer(
    model: ResNet,
    config: ClassifierConfig,
    stage: TrainingStage | str,
) -> AdamW:
    """
    Head-only 단계에서는 model.fc만 AdamW로 학습합니다.
    Fine-tuning 단계에서는 layer4와 fc에 서로 다른 learning rate를 적용합니다.
    """

    resolved_stage = TrainingStage(stage)
    configure_model_for_stage(model, resolved_stage)

    if resolved_stage is TrainingStage.HEAD_ONLY:
        return AdamW(
            [
                {
                    "params": list(model.fc.parameters()),
                    "lr": config.lr_head_only,
                    "name": "fc",
                }
            ],
            weight_decay=config.weight_decay,
        )

    layer4_parameters = [
        parameter for parameter in model.layer4.parameters() if parameter.requires_grad
    ]
    return AdamW(
        [
            {
                "params": layer4_parameters,
                "lr": config.lr_layer4,
                "name": "layer4",
            },
            {
                "params": list(model.fc.parameters()),
                "lr": config.lr_fc,
                "name": "fc",
            },
        ],
        weight_decay=config.weight_decay,
    )


def build_scheduler(
    optimizer: Optimizer,
    config: ClassifierConfig,
    stage: TrainingStage | str,
) -> LRScheduler | None:
    """fine-tuning 단계에서만 cosine scheduler를 생성한다."""

    resolved_stage = TrainingStage(stage)
    if resolved_stage is TrainingStage.HEAD_ONLY:
        return None

    fine_tuning_epochs = config.max_epochs - config.head_only_epochs
    if fine_tuning_epochs <= 0:
        raise ValueError("fine-tuning stage requires at least one epoch")
    return CosineAnnealingLR(optimizer, T_max=fine_tuning_epochs)


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None = None,
    show_progress: bool = False,
    progress_description: str = "Classifier training",
) -> TrainingEpochResult:
    """한 epoch를 학습하고 loss와 Top-k accuracy를 반환한다."""

    use_amp = device.type == "cuda"
    if use_amp and scaler is None:
        raise ValueError("CUDA training requires a GradScaler")
    if not isinstance(show_progress, bool):
        raise TypeError("show_progress must be a boolean")
    if not isinstance(progress_description, str) or not progress_description.strip():
        raise ValueError("progress_description must be a non-empty string")

    set_model_train_mode(model)
    total_loss = 0.0
    total_samples = 0
    top1_correct = 0
    top5_correct = 0

    progress = tqdm(
        data_loader,
        desc=progress_description,
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for batch in progress:
        images = batch["images"].to(device, non_blocking=use_amp)
        targets = batch["targets"].to(device, non_blocking=use_amp)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if use_amp:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = targets.shape[0]
        top_k = min(5, logits.shape[1])
        predictions = logits.detach().topk(k=top_k, dim=1).indices
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        top1_correct += int((predictions[:, 0] == targets).sum().item())
        top5_correct += int(
            (predictions == targets.unsqueeze(1)).any(dim=1).sum().item()
        )
        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            top1=f"{top1_correct / total_samples:.4f}",
        )

    if total_samples == 0:
        raise ValueError("data_loader must contain at least one sample")

    return TrainingEpochResult(
        loss=total_loss / total_samples,
        top1_accuracy=top1_correct / total_samples,
        top5_accuracy=top5_correct / total_samples,
        num_samples=total_samples,
    )


def _prepare_run_directory(
    config: ClassifierConfig,
    *,
    device: torch.device,
    class_map: Mapping[str, Any],
    data_summary: Mapping[str, Any],
) -> Path:
    run_dir = config.output_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if config.resume_from is None:
        existing_artifacts = [
            path
            for path in (
                run_dir / "last.pt",
                run_dir / "best.pt",
                run_dir / "metrics.csv",
            )
            if path.exists()
        ]
        if existing_artifacts:
            names = ", ".join(path.name for path in existing_artifacts)
            raise FileExistsError(
                f"run directory already contains training artifacts: {names}"
            )

    resolved_config = config.to_dict()
    resolved_config["device"] = str(device)
    write_json(run_dir / "resolved_config.json", resolved_config, sort_keys=True)
    write_json(run_dir / "class_map.json", class_map, sort_keys=True)
    write_json(run_dir / "data_summary.json", data_summary, sort_keys=True)
    return run_dir


def _learning_rates(optimizer: Optimizer) -> dict[str, float | None]:
    rates = {group.get("name"): float(group["lr"]) for group in optimizer.param_groups}
    return {
        "lr_layer4": rates.get("layer4"),
        "lr_fc": rates.get("fc"),
    }


def _checkpoint_kwargs(
    *,
    epoch: int,
    stage: TrainingStage,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    scaler: torch.amp.GradScaler,
    early_stopping: EarlyStopping,
    current_class_map_hash: str,
    split_hash: str,
) -> dict[str, Any]:
    return {
        "completed_epoch": epoch,
        "training_stage": stage,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "best_epoch": early_stopping.best_epoch,
        "best_macro_f1": early_stopping.best_macro_f1,
        "best_top1_accuracy": early_stopping.best_top1_accuracy,
        "epochs_without_improvement": early_stopping.epochs_without_improvement,
        "class_map_hash": current_class_map_hash,
        "split_hash": split_hash,
    }


def train_classifier(
    config: ClassifierConfig,
    *,
    data: ClassifierTrainingData | None = None,
) -> TrainingResult:
    """설정 하나로 분류기 데이터를 준비하고 학습을 실행한다."""

    training_data = data or prepare_classifier_data(config)
    return run_training(
        config,
        training_data.train_loader,
        training_data.val_loader,
        training_data.class_map,
        training_data.data_summary,
        expected_num_classes=config.expected_num_classes,
        pretrained=config.pretrained,
        device=training_data.device,
    )


def run_training(
    config: ClassifierConfig,
    train_loader: Iterable[Mapping[str, Any]],
    val_loader: Iterable[Mapping[str, Any]],
    class_map: Mapping[str, Any],
    data_summary: Mapping[str, Any],
    *,
    expected_num_classes: int = 118,
    pretrained: bool = True,
    device: torch.device | None = None,
) -> TrainingResult:
    """새 학습 또는 checkpoint 재개를 실행하고 best/last artifact를 저장한다."""

    # 입력 검증
    validate_class_map(class_map, expected_num_classes=expected_num_classes)
    validate_data_summary(
        data_summary,
        class_map,
        expected_num_classes=expected_num_classes,
    )

    # 장치 선택
    selected_device = device if device is not None else select_device()

    # 실행 디렉터리와 설정파일 생성
    set_seed(config.seed)
    run_dir = _prepare_run_directory(
        config,
        device=selected_device,
        class_map=class_map,
        data_summary=data_summary,
    )
    last_checkpoint_path = run_dir / "last.pt"
    best_checkpoint_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.csv"
    current_class_map_hash = hash_class_map(class_map)
    split_hash = data_summary["split_hash"]

    # 모델 생성
    model = build_model(expected_num_classes, pretrained=pretrained).to(selected_device)
    criterion = nn.CrossEntropyLoss()
    scaler = build_grad_scaler(selected_device)
    early_stopping = EarlyStopping(config.early_stopping_patience)
    start_epoch = 0

    # 신규 학습 또는 체크포인트 재개
    if config.resume_from is None:
        current_stage = resolve_training_stage(0, config.head_only_epochs)
        optimizer = build_optimizer(model, config, current_stage)
        scheduler = build_scheduler(optimizer, config, current_stage)
    else:
        checkpoint = load_checkpoint(config.resume_from, selected_device)
        validate_checkpoint_compatibility(
            checkpoint,
            expected_class_map_hash=current_class_map_hash,
            expected_split_hash=split_hash,
        )
        start_epoch = checkpoint["completed_epoch"] + 1
        checkpoint_stage = TrainingStage(checkpoint["training_stage"])
        expected_checkpoint_stage = resolve_training_stage(
            checkpoint["completed_epoch"],
            config.head_only_epochs,
        )
        if checkpoint_stage is not expected_checkpoint_stage:
            raise ValueError(
                "checkpoint training stage does not match completed epoch and config"
            )
        current_stage = resolve_training_stage(start_epoch, config.head_only_epochs)
        optimizer = build_optimizer(model, config, current_stage)
        scheduler = build_scheduler(optimizer, config, current_stage)
        restore_checkpoint_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            restore_optimizer=current_stage is checkpoint_stage,
        )
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            best_epoch=checkpoint["best_epoch"],
            best_macro_f1=checkpoint["best_macro_f1"],
            best_top1_accuracy=checkpoint["best_top1_accuracy"],
            epochs_without_improvement=checkpoint["epochs_without_improvement"],
        )

    # epoch별 학습과 validation 수행
    metrics_records: list[dict[str, Any]] = []
    completed_epoch = start_epoch - 1
    stopped_early = False

    for epoch in range(start_epoch, config.max_epochs):
        stage = resolve_training_stage(epoch, config.head_only_epochs)
        if stage is not current_stage:
            current_stage = stage
            optimizer = build_optimizer(model, config, current_stage)
            scheduler = build_scheduler(optimizer, config, current_stage)

        learning_rates = _learning_rates(optimizer)
        training_result = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            selected_device,
            scaler=scaler,
            show_progress=config.show_progress,
            progress_description=(f"Classifier train {epoch + 1}/{config.max_epochs}"),
        )
        validation_result = evaluate_model(
            model,
            val_loader,
            criterion,
            selected_device,
            num_classes=expected_num_classes,
            show_progress=config.show_progress,
            progress_description=(
                f"Classifier validation {epoch + 1}/{config.max_epochs}"
            ),
        )
        if scheduler is not None:
            scheduler.step()

        # 조기 종료 횟수는 fine-tuning 단계에서만 계산
        improved = early_stopping.update(
            epoch,
            validation_result,
            count_for_stopping=stage is TrainingStage.LAYER4_FINE_TUNING,
        )

        # metrics.csv 기록
        metrics = {
            "epoch": epoch,
            "training_stage": stage.value,
            "train_loss": training_result.loss,
            "train_top1_accuracy": training_result.top1_accuracy,
            "train_top5_accuracy": training_result.top5_accuracy,
            "val_loss": validation_result.loss,
            "val_top1_accuracy": validation_result.top1_accuracy,
            "val_top5_accuracy": validation_result.top5_accuracy,
            "val_macro_f1": validation_result.macro_f1,
            "validation_class_coverage": validation_result.validation_class_coverage,
            **learning_rates,
        }
        metrics_records.append(metrics)
        append_csv_row(metrics_path, metrics)

        checkpoint_kwargs = _checkpoint_kwargs(
            epoch=epoch,
            stage=stage,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early_stopping,
            current_class_map_hash=current_class_map_hash,
            split_hash=split_hash,
        )
        save_checkpoint(last_checkpoint_path, **checkpoint_kwargs)
        if improved:
            save_checkpoint(best_checkpoint_path, **checkpoint_kwargs)

        completed_epoch = epoch
        if stage is TrainingStage.LAYER4_FINE_TUNING and early_stopping.should_stop:
            stopped_early = True
            break

    return TrainingResult(
        run_dir=run_dir,
        last_checkpoint_path=last_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        completed_epoch=completed_epoch,
        best_epoch=early_stopping.best_epoch,
        best_macro_f1=early_stopping.best_macro_f1,
        best_top1_accuracy=early_stopping.best_top1_accuracy,
        stopped_early=stopped_early,
        metrics=tuple(metrics_records),
    )
