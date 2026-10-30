"""탐지기 학습 데이터 factory 호출, 입력 검증과 장치 이동."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from configs.detector import DetectorConfig
from detector.input_validation import (
    validate_detector_data_summary,
    validate_detector_label_map,
    validate_detector_training_batch,
)
from pilldata.factory import (
    TrainingFactoryResult,
    call_training_data_factory,
    first_batch,
)
from utils.runtime import resolve_device


@dataclass(frozen=True, slots=True)
class DetectorTrainingData:
    """검증을 마쳐 Notebook 확인과 학습에 재사용할 수 있는 입력."""

    train_loader: Iterable[Mapping[str, Any]]
    val_loader: Iterable[Mapping[str, Any]]
    label_map: Mapping[str, Any]
    data_summary: Mapping[str, Any]
    device: torch.device


def prepare_detector_data(config: DetectorConfig) -> DetectorTrainingData:
    """탐지기 factory를 한 번 호출하고 반환된 학습 입력을 검증한다."""

    device = resolve_device(config.device)
    training_data = call_training_data_factory(
        config.data_factory_path,
        config=config,
        device=device,
    )
    _validate_detector_training_data(training_data)
    train_loader, val_loader, label_map, data_summary = training_data
    return DetectorTrainingData(
        train_loader=train_loader,
        val_loader=val_loader,
        label_map=label_map,
        data_summary=data_summary,
        device=device,
    )


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


def _validate_detector_training_data(data: TrainingFactoryResult) -> None:
    train_loader, val_loader, label_map, data_summary = data
    validate_detector_label_map(label_map)
    validate_detector_data_summary(data_summary, label_map)
    validate_detector_training_batch(
        first_batch(train_loader, name="train_loader"),
        label_map,
    )
    validate_detector_training_batch(
        first_batch(val_loader, name="val_loader"),
        label_map,
    )
