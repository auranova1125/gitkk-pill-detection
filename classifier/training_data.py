"""분류기 학습 데이터 factory 호출과 입력 검증."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from classifier.input_validation import (
    validate_batch,
    validate_class_map,
    validate_data_summary,
)
from configs.classifier import ClassifierConfig
from pilldata.factory import load_data_factory
from utils.runtime import resolve_device

TrainingFactoryResult = tuple[
    Iterable[Mapping[str, Any]],
    Iterable[Mapping[str, Any]],
    Mapping[str, Any],
    Mapping[str, Any],
]


@dataclass(frozen=True, slots=True)
class ClassifierTrainingData:
    """검증을 마쳐 Notebook 확인과 학습에 재사용할 수 있는 입력."""

    train_loader: Iterable[Mapping[str, Any]]
    val_loader: Iterable[Mapping[str, Any]]
    class_map: Mapping[str, Any]
    data_summary: Mapping[str, Any]
    device: torch.device


def prepare_classifier_data(config: ClassifierConfig) -> ClassifierTrainingData:
    """분류기 factory를 한 번 호출하고 반환된 학습 입력을 검증한다."""

    device = resolve_device(config.device)
    factory = load_data_factory(config.data_factory_path)
    training_data = _call_training_factory(factory, config=config, device=device)
    _validate_classifier_training_data(
        training_data,
        expected_num_classes=config.expected_num_classes,
    )
    train_loader, val_loader, class_map, data_summary = training_data
    return ClassifierTrainingData(
        train_loader=train_loader,
        val_loader=val_loader,
        class_map=class_map,
        data_summary=data_summary,
        device=device,
    )


def _call_training_factory(
    factory: Any,
    *,
    config: ClassifierConfig,
    device: torch.device,
) -> TrainingFactoryResult:
    value = factory(config=config, device=device)
    if not isinstance(value, tuple) or len(value) != 4:
        raise TypeError(
            "training data factory must return "
            "(train_loader, val_loader, class_map, data_summary)"
        )
    train_loader, val_loader, class_map, data_summary = value
    if not isinstance(class_map, Mapping) or not isinstance(data_summary, Mapping):
        raise TypeError("factory class map and data summary must be mappings")
    return train_loader, val_loader, class_map, data_summary


def _validate_classifier_training_data(
    data: TrainingFactoryResult,
    *,
    expected_num_classes: int,
) -> None:
    train_loader, val_loader, class_map, data_summary = data
    validate_class_map(class_map, expected_num_classes=expected_num_classes)
    validate_data_summary(
        data_summary,
        class_map,
        expected_num_classes=expected_num_classes,
    )
    validate_batch(
        _first_batch(train_loader, name="train_loader"),
        class_map,
        expected_num_classes=expected_num_classes,
    )
    validate_batch(
        _first_batch(val_loader, name="val_loader"),
        class_map,
        expected_num_classes=expected_num_classes,
    )


def _first_batch(loader: Iterable[Any], *, name: str) -> Any:
    try:
        return next(iter(loader))
    except StopIteration as error:
        raise ValueError(f"{name} must contain at least one batch") from error
