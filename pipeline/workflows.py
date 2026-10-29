"""YAML 설정을 기존 모델 함수와 연결하는 실행 workflow."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from classifier.evaluate import (
    ValidationArtifacts,
    evaluate_with_predictions,
    save_validation_artifacts,
)
from classifier.inference import load_inference_model as load_classifier_model
from classifier.input_validation import (
    validate_batch as validate_classifier_batch,
)
from classifier.input_validation import validate_class_map, validate_data_summary
from classifier.train import TrainingResult as ClassifierTrainingResult
from classifier.train import run_training as run_classifier_training
from configs.schema import (
    ClassifierExperimentConfig,
    DetectorExperimentConfig,
    PipelineExperimentConfig,
)
from detector.evaluate import EvaluationResult as DetectorEvaluationResult
from detector.evaluate import evaluate_model as evaluate_detector_model
from detector.inference import load_inference_model as load_detector_model
from detector.inference import validate_inference_batch
from detector.input_validation import (
    validate_detection_batch,
    validate_detector_data_summary,
    validate_detector_label_map,
)
from detector.train import TrainingResult as DetectorTrainingResult
from detector.train import run_training as run_detector_training
from pilldata.factory import load_data_factory
from pipeline.inference import (
    EndToEndInferenceResult,
    run_end_to_end_inference,
    save_end_to_end_artifacts,
)
from utils.io import read_json, write_json
from utils.runtime import resolve_device

TrainingFactoryResult = tuple[
    Iterable[Mapping[str, Any]],
    Iterable[Mapping[str, Any]],
    Mapping[str, Any],
    Mapping[str, Any],
]


def train_classifier_experiment(
    experiment: ClassifierExperimentConfig,
) -> ClassifierTrainingResult:
    """분류기 factory 입력을 검증하고 기존 학습 함수를 실행한다."""

    device = resolve_device(experiment.device)
    factory = load_data_factory(experiment.data_factory_path)
    inputs = _call_training_factory(factory, config=experiment.training, device=device)
    _validate_classifier_inputs(
        inputs,
        expected_num_classes=experiment.expected_num_classes,
    )
    train_loader, val_loader, class_map, data_summary = _call_training_factory(
        factory,
        config=experiment.training,
        device=device,
    )
    return run_classifier_training(
        experiment.training,
        train_loader,
        val_loader,
        class_map,
        data_summary,
        expected_num_classes=experiment.expected_num_classes,
        pretrained=experiment.pretrained,
        device=device,
    )


def train_detector_experiment(
    experiment: DetectorExperimentConfig,
) -> DetectorTrainingResult:
    """탐지기 factory 입력을 검증하고 기존 학습 함수를 실행한다."""

    device = resolve_device(experiment.device)
    factory = load_data_factory(experiment.data_factory_path)
    inputs = _call_training_factory(factory, config=experiment.training, device=device)
    _validate_detector_inputs(inputs)
    train_loader, val_loader, label_map, data_summary = _call_training_factory(
        factory,
        config=experiment.training,
        device=device,
    )
    return run_detector_training(
        experiment.training,
        train_loader,
        val_loader,
        label_map,
        data_summary,
        pretrained_backbone=experiment.pretrained_backbone,
        device=device,
    )


def evaluate_classifier_experiment(
    experiment: ClassifierExperimentConfig,
    checkpoint_path: Path,
    *,
    output_dir: Path | None = None,
) -> ValidationArtifacts:
    """분류기 checkpoint를 validation loader로 평가하고 artifact를 저장한다."""

    device = resolve_device(experiment.device)
    factory = load_data_factory(experiment.data_factory_path)
    inputs = _call_training_factory(factory, config=experiment.training, device=device)
    _validate_classifier_inputs(
        inputs,
        expected_num_classes=experiment.expected_num_classes,
    )
    _, val_loader, class_map, _ = _call_training_factory(
        factory,
        config=experiment.training,
        device=device,
    )
    model, device = load_classifier_model(
        checkpoint_path,
        class_map,
        expected_num_classes=experiment.expected_num_classes,
        device=device,
    )
    artifacts = evaluate_with_predictions(
        model,
        val_loader,
        class_map,
        nn.CrossEntropyLoss(),
        device,
        show_progress=experiment.training.show_progress,
        progress_description="Classifier evaluation",
    )
    destination = (
        Path(output_dir) if output_dir is not None else Path(checkpoint_path).parent
    )
    save_validation_artifacts(artifacts, destination)
    write_json(destination / "evaluation_metrics.json", artifacts.metrics.to_dict())
    return artifacts


def evaluate_detector_experiment(
    experiment: DetectorExperimentConfig,
    checkpoint_path: Path,
    *,
    output_dir: Path | None = None,
) -> DetectorEvaluationResult:
    """탐지기 checkpoint를 validation loader로 평가하고 지표를 저장한다."""

    device = resolve_device(experiment.device)
    factory = load_data_factory(experiment.data_factory_path)
    inputs = _call_training_factory(factory, config=experiment.training, device=device)
    _validate_detector_inputs(inputs)
    _, val_loader, label_map, _ = _call_training_factory(
        factory,
        config=experiment.training,
        device=device,
    )
    model, device = load_detector_model(checkpoint_path, label_map, device=device)
    result = evaluate_detector_model(
        model,
        val_loader,
        device,
        score_threshold=experiment.training.score_threshold,
        show_progress=experiment.training.show_progress,
        progress_description="Detector evaluation",
    )
    destination = (
        Path(output_dir) if output_dir is not None else Path(checkpoint_path).parent
    )
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "evaluation_metrics.json", result.to_dict())
    return result


def run_pipeline_experiment(
    experiment: PipelineExperimentConfig,
) -> EndToEndInferenceResult:
    """두 checkpoint와 추론 factory를 불러와 통합 추론을 실행한다."""

    device = resolve_device(experiment.device)
    factory = load_data_factory(
        experiment.inference_factory_path,
        setting_name="inference_factory_path",
    )
    detector_label_map = read_json(experiment.detector_label_map_path)
    class_map = read_json(experiment.classifier_class_map_path)
    validate_detector_label_map(detector_label_map)
    if not isinstance(class_map, Mapping):
        raise TypeError("class map JSON root must be a mapping")
    num_classes = class_map.get("num_classes")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool):
        raise TypeError("class_map.num_classes must be an integer")
    validate_class_map(class_map, expected_num_classes=num_classes)

    inference_loader = factory(batch_size=experiment.batch_size)
    validate_inference_batch(_first_batch(inference_loader, name="inference_loader"))
    inference_loader = factory(batch_size=experiment.batch_size)
    detector_model, device = load_detector_model(
        experiment.detector_checkpoint_path,
        detector_label_map,
        device=device,
    )
    classifier_model, device = load_classifier_model(
        experiment.classifier_checkpoint_path,
        class_map,
        expected_num_classes=num_classes,
        device=device,
    )
    result = run_end_to_end_inference(
        detector_model,
        classifier_model,
        inference_loader,
        class_map,
        device,
        score_threshold=experiment.score_threshold,
        detector_name=experiment.detector_name,
        show_progress=experiment.show_progress,
    )
    save_end_to_end_artifacts(result, experiment.output_dir)
    return result


def _call_training_factory(
    factory: Any,
    *,
    config: Any,
    device: torch.device,
) -> TrainingFactoryResult:
    value = factory(config=config, device=device)
    if not isinstance(value, tuple) or len(value) != 4:
        raise TypeError(
            "training data factory must return "
            "(train_loader, val_loader, label_map, data_summary)"
        )
    train_loader, val_loader, label_map, data_summary = value
    if not isinstance(label_map, Mapping) or not isinstance(data_summary, Mapping):
        raise TypeError("factory label map and data summary must be mappings")
    return train_loader, val_loader, label_map, data_summary


def _validate_classifier_inputs(
    inputs: TrainingFactoryResult,
    *,
    expected_num_classes: int,
) -> None:
    train_loader, val_loader, class_map, data_summary = inputs
    validate_class_map(class_map, expected_num_classes=expected_num_classes)
    validate_data_summary(
        data_summary,
        class_map,
        expected_num_classes=expected_num_classes,
    )
    validate_classifier_batch(
        _first_batch(train_loader, name="train_loader"),
        class_map,
        expected_num_classes=expected_num_classes,
    )
    validate_classifier_batch(
        _first_batch(val_loader, name="val_loader"),
        class_map,
        expected_num_classes=expected_num_classes,
    )


def _validate_detector_inputs(inputs: TrainingFactoryResult) -> None:
    train_loader, val_loader, label_map, data_summary = inputs
    validate_detector_label_map(label_map)
    validate_detector_data_summary(data_summary, label_map)
    validate_detection_batch(
        _first_batch(train_loader, name="train_loader"),
        label_map,
    )
    validate_detection_batch(
        _first_batch(val_loader, name="val_loader"),
        label_map,
    )


def _first_batch(loader: Iterable[Any], *, name: str) -> Any:
    try:
        return next(iter(loader))
    except StopIteration as error:
        raise ValueError(f"{name} must contain at least one batch") from error
