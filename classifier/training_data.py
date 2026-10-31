"""분류기 학습 데이터 factory 호출과 입력 검증."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from classifier.input_validation import (
    validate_classifier_class_map,
    validate_classifier_data_summary,
    validate_classifier_training_batch,
)
from configs.classifier import ClassifierConfig
from data_loading.factory import (
    TrainingFactoryResult,
    call_training_data_factory,
    first_batch,
)
from utils.runtime import resolve_device


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
    training_data = call_training_data_factory(
        config.data_factory_path,
        config=config,
        device=device,
    )
    validate_classifier_training_data(
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


def validate_classifier_training_data(
    data: TrainingFactoryResult,
    *,
    expected_num_classes: int,
) -> None:
    """factory 반환값의 클래스 정보와 첫 학습·검증 batch를 검증한다."""

    train_loader, val_loader, class_map, data_summary = data
    validate_classifier_class_map(
        class_map,
        expected_num_classes=expected_num_classes,
    )
    validate_classifier_data_summary(
        data_summary,
        class_map,
        expected_num_classes=expected_num_classes,
    )
    validate_classifier_training_batch(
        first_batch(train_loader, name="train_loader"),
        class_map,
        expected_num_classes=expected_num_classes,
    )
    validate_classifier_training_batch(
        first_batch(val_loader, name="val_loader"),
        class_map,
        expected_num_classes=expected_num_classes,
    )
