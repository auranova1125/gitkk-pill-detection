"""탐지기 optimizer, scheduler와 한 epoch 학습."""

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from detector.config import DetectorConfig


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
) -> TrainingEpochResult:
    """DataLoader를 한 번 순회해 탐지 모델을 학습한다."""

    model.train()
    use_amp = device.type == "cuda"
    active_scaler = scaler if scaler is not None else build_grad_scaler(device)
    if active_scaler.is_enabled() != use_amp:
        raise ValueError("GradScaler enabled state must match CUDA usage")

    total_loss = 0.0
    component_totals: dict[str, float] = {}
    expected_loss_names: set[str] | None = None
    num_batches = 0
    num_images = 0

    for batch in data_loader:
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
