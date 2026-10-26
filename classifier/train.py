"""분류기 optimizer 구성과 epoch 학습 함수."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torchvision.models.resnet import ResNet

from classifier.config import TrainConfig
from classifier.model import (
    TrainingStage,
    configure_model_for_stage,
    set_model_train_mode,
)


@dataclass(frozen=True, slots=True)
class TrainingEpochResult:
    """한 training epoch의 집계 지표."""

    loss: float
    top1_accuracy: float
    top5_accuracy: float
    num_samples: int


def select_device() -> torch.device:
    """CUDA, MPS, CPU 순서로 사용 가능한 장치를 선택한다."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_training_stage(epoch: int, head_only_epochs: int) -> TrainingStage:
    """0부터 시작하는 epoch 번호에 해당하는 학습 단계를 반환한다."""

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
    config: TrainConfig,
    stage: TrainingStage | str,
) -> AdamW:
    """학습 단계에 맞는 AdamW optimizer를 새로 생성한다."""

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
    config: TrainConfig,
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


def build_grad_scaler(device: torch.device) -> torch.amp.GradScaler:
    """CUDA에서만 활성화되는 GradScaler를 생성한다."""

    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None = None,
) -> TrainingEpochResult:
    """한 epoch를 학습하고 loss와 Top-k accuracy를 반환한다."""

    use_amp = device.type == "cuda"
    if use_amp and scaler is None:
        raise ValueError("CUDA training requires a GradScaler")

    set_model_train_mode(model)
    total_loss = 0.0
    total_samples = 0
    top1_correct = 0
    top5_correct = 0

    for batch in data_loader:
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

    if total_samples == 0:
        raise ValueError("data_loader must contain at least one sample")

    return TrainingEpochResult(
        loss=total_loss / total_samples,
        top1_accuracy=top1_correct / total_samples,
        top5_accuracy=top5_correct / total_samples,
        num_samples=total_samples,
    )
