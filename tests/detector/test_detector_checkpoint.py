"""탐지기 checkpoint, 재개와 조기 종료 테스트."""

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torchvision
from torch import nn

import detector.train as train_module
from configs.detector import DetectorConfig
from detector.checkpoint import (
    hash_label_map,
    load_checkpoint,
    restore_checkpoint_state,
    save_checkpoint,
    validate_checkpoint_compatibility,
)
from detector.evaluate import EvaluationResult
from detector.train import (
    EarlyStopping,
    TrainingEpochResult,
    build_optimizer,
    build_scheduler,
    run_training,
)
from utils.runtime import build_grad_scaler, set_seed


class TinyCheckpointDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


def make_data_summary() -> dict[str, Any]:
    return {
        "num_train_images": 2,
        "num_val_images": 2,
        "num_train_source_annotations": 3,
        "num_val_source_annotations": 3,
        "num_train_instances": 3,
        "num_val_instances": 3,
        "train_group_count": 1,
        "val_group_count": 1,
        "group_overlap_count": 0,
        "train_empty_image_count": 0,
        "val_empty_image_count": 0,
        "train_max_instances_per_image": 2,
        "val_max_instances_per_image": 2,
        "train_label_counts": {"1": 3},
        "val_label_counts": {"1": 3},
        "split_hash": "synthetic-split",
    }


def make_config(
    tmp_path: Path,
    *,
    run_name: str = "checkpoint-test",
    max_epochs: int = 3,
    patience: int = 5,
    resume_from: Path | None = None,
) -> DetectorConfig:
    return DetectorConfig(
        run_name=run_name,
        output_dir=tmp_path / "outputs",
        learning_rate=0.1,
        weight_decay=5e-4,
        score_threshold=0.5,
        nms_threshold=0.45,
        early_stopping_patience=patience,
        max_epochs=max_epochs,
        resume_from=resume_from,
    )


def make_evaluation_result(
    map_75_95: float,
    mean_average_recall: float,
) -> EvaluationResult:
    return EvaluationResult(
        map=map_75_95,
        map_50=map_75_95,
        map_75=map_75_95,
        map_75_95=map_75_95,
        mean_average_recall=mean_average_recall,
        recall_before_threshold=mean_average_recall,
        recall_after_threshold=mean_average_recall,
        mean_count_error_before_threshold=0.0,
        mean_count_error_after_threshold=0.0,
        num_images=2,
        num_ground_truths=3,
        num_predictions_before_threshold=3,
        num_predictions_after_threshold=3,
        score_threshold=0.5,
        recall_iou_threshold=0.5,
    )


def model_config() -> dict[str, Any]:
    return {
        "num_classes": 2,
        "pretrained_backbone": False,
        "nms_threshold": 0.45,
    }


def patch_training_components(
    monkeypatch: pytest.MonkeyPatch,
    evaluation_values: list[tuple[float, float]],
) -> None:
    evaluation_iterator = iter(evaluation_values)

    def build_tiny_model(*args: Any, **kwargs: Any) -> nn.Module:
        del args, kwargs
        return TinyCheckpointDetector()

    def fake_train_one_epoch(
        model: nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> TrainingEpochResult:
        del kwargs
        optimizer = args[1]
        optimizer.zero_grad(set_to_none=True)
        parameter = next(model.parameters())
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        return TrainingEpochResult(
            loss=1.0,
            loss_components={"classification": 0.6, "bbox_regression": 0.4},
            num_batches=1,
            num_images=2,
        )

    def fake_evaluate_model(*args: Any, **kwargs: Any) -> EvaluationResult:
        del args, kwargs
        return make_evaluation_result(*next(evaluation_iterator))

    monkeypatch.setattr(train_module, "build_detector_model", build_tiny_model)
    monkeypatch.setattr(train_module, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(train_module, "evaluate_model", fake_evaluate_model)


def test_hash_label_map_is_independent_of_dictionary_key_order(
    detector_label_map: dict[str, Any],
) -> None:
    first = detector_label_map
    second = {
        "classes": [{"name": "pill", "label": 1}],
        "background_label": 0,
        "num_model_classes": 2,
        "num_foreground_classes": 1,
    }

    assert hash_label_map(first) == hash_label_map(second)


def test_checkpoint_restores_model_optimizer_scheduler_and_rng(
    tmp_path: Path,
    detector_label_map: dict[str, Any],
) -> None:
    config = make_config(tmp_path)
    model = TinyCheckpointDetector()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = build_grad_scaler(torch.device("cpu"))
    checkpoint_path = tmp_path / "checkpoint.pt"
    set_seed(42)

    save_checkpoint(
        checkpoint_path,
        completed_epoch=0,
        model_name=config.model_name,
        model_config=model_config(),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        best_epoch=0,
        best_map_75_95=0.5,
        best_mean_average_recall=0.6,
        epochs_without_improvement=0,
        label_map_hash=hash_label_map(detector_label_map),
        split_hash="synthetic-split",
    )
    expected_random_values = (random.random(), np.random.rand(), torch.rand(1))
    original_weight = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.zero_()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    scheduler.step()
    random.random()
    np.random.rand()
    torch.rand(1)

    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    restore_checkpoint_state(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )

    assert torch.equal(model.weight, original_weight)
    assert scheduler.last_epoch == 0
    assert random.random() == expected_random_values[0]
    assert np.random.rand() == expected_random_values[1]
    assert torch.equal(torch.rand(1), expected_random_values[2])
    assert checkpoint["torchvision_version"] == torchvision.__version__


@pytest.mark.parametrize(
    ("field_name", "field_value", "error_text"),
    [
        ("model_name", "other", "model name"),
        ("model_config", {"num_classes": 3}, "model config"),
        ("label_map_hash", "other", "label map hash"),
        ("split_hash", "other", "split hash"),
    ],
)
def test_checkpoint_compatibility_rejects_mismatch(
    field_name: str,
    field_value: Any,
    error_text: str,
) -> None:
    checkpoint = {
        "model_name": "ssdlite320_mobilenet_v3_large",
        "model_config": model_config(),
        "label_map_hash": "expected-label-map",
        "split_hash": "synthetic-split",
        field_name: field_value,
    }

    with pytest.raises(ValueError, match=error_text):
        validate_checkpoint_compatibility(
            checkpoint,
            expected_model_name="ssdlite320_mobilenet_v3_large",
            expected_model_config=model_config(),
            expected_label_map_hash="expected-label-map",
            expected_split_hash="synthetic-split",
        )


def test_early_stopping_uses_recall_to_break_map_tie() -> None:
    early_stopping = EarlyStopping(patience=2)

    assert early_stopping.update(0, make_evaluation_result(0.5, 0.6))
    assert early_stopping.update(1, make_evaluation_result(0.5, 0.7))
    assert not early_stopping.update(2, make_evaluation_result(0.5, 0.7))
    assert not early_stopping.update(3, make_evaluation_result(0.4, 0.9))

    assert early_stopping.best_epoch == 1
    assert early_stopping.best_map_75_95 == pytest.approx(0.5)
    assert early_stopping.best_mean_average_recall == pytest.approx(0.7)
    assert early_stopping.should_stop


def test_run_training_saves_artifacts_and_best_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    detector_label_map: dict[str, Any],
) -> None:
    patch_training_components(
        monkeypatch,
        [(0.4, 0.7), (0.5, 0.6), (0.5, 0.5)],
    )

    result = run_training(
        make_config(tmp_path),
        [],
        [],
        detector_label_map,
        make_data_summary(),
        pretrained_backbone=False,
        device=torch.device("cpu"),
    )

    assert result.completed_epoch == 2
    assert result.best_epoch == 1
    assert result.best_map_75_95 == pytest.approx(0.5)
    assert result.last_checkpoint_path.exists()
    assert result.best_checkpoint_path.exists()
    assert (result.run_dir / "resolved_config.json").exists()
    assert (result.run_dir / "label_map.json").exists()
    assert (result.run_dir / "data_summary.json").exists()
    with (result.run_dir / "metrics.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [int(row["epoch"]) for row in rows] == [0, 1, 2]
    resolved_config = json.loads(
        (result.run_dir / "resolved_config.json").read_text(encoding="utf-8")
    )
    assert resolved_config["device"] == "cpu"

    best_checkpoint = load_checkpoint(
        result.best_checkpoint_path,
        torch.device("cpu"),
    )
    assert best_checkpoint["completed_epoch"] == 1


def test_run_training_rejects_existing_artifacts(
    tmp_path: Path,
    detector_label_map: dict[str, Any],
) -> None:
    config = make_config(tmp_path, run_name="existing-run")
    run_dir = config.output_dir / config.run_name
    run_dir.mkdir(parents=True)
    (run_dir / "last.pt").write_bytes(b"existing checkpoint")

    with pytest.raises(FileExistsError, match=r"last\.pt"):
        run_training(
            config,
            [],
            [],
            detector_label_map,
            make_data_summary(),
            pretrained_backbone=False,
            device=torch.device("cpu"),
        )


def test_run_training_resumes_after_completed_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    detector_label_map: dict[str, Any],
) -> None:
    config = make_config(tmp_path, run_name="resume")
    model = TinyCheckpointDetector()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    checkpoint_path = tmp_path / "resume.pt"
    save_checkpoint(
        checkpoint_path,
        completed_epoch=0,
        model_name=config.model_name,
        model_config=model_config(),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=build_grad_scaler(torch.device("cpu")),
        best_epoch=0,
        best_map_75_95=0.4,
        best_mean_average_recall=0.5,
        epochs_without_improvement=0,
        label_map_hash=hash_label_map(detector_label_map),
        split_hash="synthetic-split",
    )
    patch_training_components(monkeypatch, [(0.5, 0.5), (0.5, 0.6)])
    resume_config = make_config(
        tmp_path,
        run_name="resume",
        resume_from=checkpoint_path,
    )

    result = run_training(
        resume_config,
        [],
        [],
        detector_label_map,
        make_data_summary(),
        pretrained_backbone=False,
        device=torch.device("cpu"),
    )

    assert result.completed_epoch == 2
    assert [row["epoch"] for row in result.metrics] == [1, 2]
    assert result.best_epoch == 2


def test_run_training_rejects_incompatible_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    detector_label_map: dict[str, Any],
) -> None:
    config = make_config(tmp_path)
    model = TinyCheckpointDetector()
    optimizer = build_optimizer(model, config)
    checkpoint_path = tmp_path / "incompatible.pt"
    save_checkpoint(
        checkpoint_path,
        completed_epoch=0,
        model_name="different_model",
        model_config=model_config(),
        model=model,
        optimizer=optimizer,
        scheduler=build_scheduler(optimizer, config),
        scaler=build_grad_scaler(torch.device("cpu")),
        best_epoch=0,
        best_map_75_95=0.4,
        best_mean_average_recall=0.5,
        epochs_without_improvement=0,
        label_map_hash=hash_label_map(detector_label_map),
        split_hash="synthetic-split",
    )
    patch_training_components(monkeypatch, [(0.5, 0.5), (0.5, 0.6)])

    with pytest.raises(ValueError, match="model name"):
        run_training(
            make_config(tmp_path, resume_from=checkpoint_path),
            [],
            [],
            detector_label_map,
            make_data_summary(),
            pretrained_backbone=False,
            device=torch.device("cpu"),
        )
