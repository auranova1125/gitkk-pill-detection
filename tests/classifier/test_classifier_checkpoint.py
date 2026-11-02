"""분류기 checkpoint, 재개, 조기 종료 테스트."""

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import classifier.train as train_module
from classifier.checkpoint import (
    hash_class_map,
    load_checkpoint,
    restore_checkpoint_state,
    save_checkpoint,
    validate_checkpoint_compatibility,
)
from classifier.evaluate import EvaluationResult
from classifier.model import TrainingStage
from classifier.train import (
    EarlyStopping,
    TrainingEpochResult,
    build_optimizer,
    build_scheduler,
    resolve_training_stage,
    run_training,
)
from configs.classifier import ClassifierConfig
from utils.runtime import build_grad_scaler, set_seed

NUM_CLASSES = 2


class TinyCheckpointDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.images = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 1.0], [0.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
            ],
            dtype=torch.float32,
        )
        self.targets = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"images": self.images[index], "targets": self.targets[index]}


class TinyStageClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.layer4 = nn.Sequential(nn.Flatten(), nn.Linear(4, 4), nn.ReLU())
        self.fc = nn.Linear(4, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(self.layer4(images))


def make_class_map() -> dict[str, Any]:
    return {
        "num_classes": NUM_CLASSES,
        "classes": [
            {
                "class_index": index,
                "category_id": 1900 + index,
                "item_seq": f"item-{index}",
                "name": f"합성 품목 {index}",
            }
            for index in range(NUM_CLASSES)
        ],
    }


def make_data_summary() -> dict[str, Any]:
    return {
        "num_classes": NUM_CLASSES,
        "num_train_samples": 4,
        "num_val_samples": 4,
        "train_class_counts": {"0": 2, "1": 2},
        "val_class_counts": {"0": 2, "1": 2},
        "train_group_count": 4,
        "val_group_count": 4,
        "group_overlap_count": 0,
        "validation_class_indices": [0, 1],
        "split_hash": "synthetic-split",
    }


def make_config(
    tmp_path: Path,
    *,
    run_name: str,
    max_epochs: int = 4,
    head_only_epochs: int = 1,
    patience: int = 7,
    resume_from: Path | None = None,
) -> ClassifierConfig:
    return ClassifierConfig(
        run_name=run_name,
        output_dir=tmp_path / "outputs",
        class_map_path=tmp_path / "class_map.json",
        data_summary_path=tmp_path / "data_summary.json",
        max_epochs=max_epochs,
        head_only_epochs=head_only_epochs,
        early_stopping_patience=patience,
        resume_from=resume_from,
    )


def make_evaluation_result(
    macro_f1: float,
    top1_accuracy: float,
) -> EvaluationResult:
    return EvaluationResult(
        loss=1.0,
        top1_accuracy=top1_accuracy,
        top5_accuracy=1.0,
        macro_f1=macro_f1,
        validation_class_coverage=1.0,
        num_samples=4,
        per_class=(),
        confusion_pairs=(),
    )


def patch_tiny_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def build_tiny_model(num_classes: int, pretrained: bool = True) -> nn.Module:
        del pretrained
        return TinyStageClassifier(num_classes)

    monkeypatch.setattr(train_module, "build_classifier_model", build_tiny_model)


def test_hash_class_map_is_independent_of_dictionary_key_order() -> None:
    first = {"num_classes": 1, "classes": [{"class_index": 0, "name": "A"}]}
    second = {"classes": [{"name": "A", "class_index": 0}], "num_classes": 1}

    assert hash_class_map(first) == hash_class_map(second)


def test_checkpoint_restores_model_optimizer_and_rng(tmp_path: Path) -> None:
    config = make_config(tmp_path, run_name="checkpoint")
    model = TinyStageClassifier()
    optimizer = build_optimizer(model, config, TrainingStage.HEAD_ONLY)
    scaler = build_grad_scaler(torch.device("cpu"))
    checkpoint_path = tmp_path / "checkpoint.pt"
    class_map_hash = hash_class_map(make_class_map())
    set_seed(42)

    save_checkpoint(
        checkpoint_path,
        completed_epoch=0,
        training_stage=TrainingStage.HEAD_ONLY,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        best_epoch=0,
        best_macro_f1=0.5,
        best_top1_accuracy=0.5,
        epochs_without_improvement=0,
        class_map_hash=class_map_hash,
        split_hash="synthetic-split",
    )
    expected_random_values = (random.random(), np.random.rand(), torch.rand(1))
    original_weight = model.fc.weight.detach().clone()
    with torch.no_grad():
        model.fc.weight.zero_()
    random.random()
    np.random.rand()
    torch.rand(1)

    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    restore_checkpoint_state(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
    )

    assert torch.equal(model.fc.weight, original_weight)
    assert random.random() == expected_random_values[0]
    assert np.random.rand() == expected_random_values[1]
    assert torch.equal(torch.rand(1), expected_random_values[2])


@pytest.mark.parametrize(
    ("class_map_hash", "split_hash", "error_text"),
    [
        ("different", "synthetic-split", "class map hash"),
        ("expected", "different", "split hash"),
    ],
)
def test_checkpoint_compatibility_rejects_mismatch(
    class_map_hash: str,
    split_hash: str,
    error_text: str,
) -> None:
    checkpoint = {
        "class_map_hash": class_map_hash,
        "split_hash": split_hash,
    }

    with pytest.raises(ValueError, match=error_text):
        validate_checkpoint_compatibility(
            checkpoint,
            expected_class_map_hash="expected",
            expected_split_hash="synthetic-split",
        )


def test_early_stopping_uses_top1_to_break_macro_f1_tie() -> None:
    early_stopping = EarlyStopping(patience=2)

    assert early_stopping.update(
        0,
        make_evaluation_result(0.5, 0.6),
        count_for_stopping=False,
    )
    assert early_stopping.update(
        1,
        make_evaluation_result(0.5, 0.7),
        count_for_stopping=True,
    )
    assert early_stopping.best_epoch == 1
    assert early_stopping.epochs_without_improvement == 0


def test_run_training_saves_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_tiny_model(monkeypatch)
    config = make_config(tmp_path, run_name="full-run", max_epochs=3)
    data_loader = DataLoader(TinyCheckpointDataset(), batch_size=2, shuffle=False)

    result = run_training(
        config,
        data_loader,
        data_loader,
        make_class_map(),
        make_data_summary(),
        expected_num_classes=NUM_CLASSES,
        pretrained=False,
        device=torch.device("cpu"),
    )

    assert result.completed_epoch == 2
    assert len(result.metrics) == 3
    assert result.last_checkpoint_path.exists()
    assert result.best_checkpoint_path.exists()
    assert (result.run_dir / "resolved_config.json").exists()
    assert (result.run_dir / "class_map.json").exists()
    assert (result.run_dir / "data_summary.json").exists()
    with (result.run_dir / "metrics.csv").open(encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == 3


@pytest.mark.parametrize(
    ("completed_epoch", "head_only_epochs", "expected_stages"),
    [
        (
            0,
            3,
            [
                TrainingStage.HEAD_ONLY,
                TrainingStage.HEAD_ONLY,
                TrainingStage.LAYER4_FINE_TUNING,
            ],
        ),
        (
            1,
            2,
            [
                TrainingStage.LAYER4_FINE_TUNING,
                TrainingStage.LAYER4_FINE_TUNING,
            ],
        ),
        (
            1,
            1,
            [
                TrainingStage.LAYER4_FINE_TUNING,
                TrainingStage.LAYER4_FINE_TUNING,
            ],
        ),
    ],
)
def test_run_training_resumes_in_each_stage_and_across_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completed_epoch: int,
    head_only_epochs: int,
    expected_stages: list[TrainingStage],
) -> None:
    patch_tiny_model(monkeypatch)
    config = make_config(
        tmp_path,
        run_name=f"resume-{completed_epoch}-{head_only_epochs}",
        max_epochs=4,
        head_only_epochs=head_only_epochs,
    )
    checkpoint_path = tmp_path / f"stage-{completed_epoch}-{head_only_epochs}.pt"
    checkpoint_stage = resolve_training_stage(completed_epoch, head_only_epochs)
    model = TinyStageClassifier()
    optimizer = build_optimizer(model, config, checkpoint_stage)
    scheduler = build_scheduler(optimizer, config, checkpoint_stage)
    if scheduler is not None:
        optimizer.step()
        scheduler.step()
    save_checkpoint(
        checkpoint_path,
        completed_epoch=completed_epoch,
        training_stage=checkpoint_stage,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=build_grad_scaler(torch.device("cpu")),
        best_epoch=1,
        best_macro_f1=0.0,
        best_top1_accuracy=0.0,
        epochs_without_improvement=0,
        class_map_hash=hash_class_map(make_class_map()),
        split_hash="synthetic-split",
    )
    config.resume_from = checkpoint_path
    data_loader = DataLoader(TinyCheckpointDataset(), batch_size=2, shuffle=False)

    result = run_training(
        config,
        data_loader,
        data_loader,
        make_class_map(),
        make_data_summary(),
        expected_num_classes=NUM_CLASSES,
        pretrained=False,
        device=torch.device("cpu"),
    )

    assert result.completed_epoch == 3
    assert [metric["epoch"] for metric in result.metrics] == list(
        range(completed_epoch + 1, 4)
    )
    assert [metric["training_stage"] for metric in result.metrics] == [
        stage.value for stage in expected_stages
    ]


def test_run_training_stops_after_fine_tuning_patience(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_tiny_model(monkeypatch)
    config = make_config(
        tmp_path,
        run_name="early-stop",
        max_epochs=10,
        head_only_epochs=1,
        patience=2,
    )

    def fake_train_one_epoch(*args: Any, **kwargs: Any) -> TrainingEpochResult:
        del kwargs
        args[3].step()
        return TrainingEpochResult(1.0, 0.5, 1.0, 4)

    def fake_evaluate_model(*args: Any, **kwargs: Any) -> EvaluationResult:
        del args, kwargs
        return make_evaluation_result(0.5, 0.5)

    monkeypatch.setattr(train_module, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(train_module, "evaluate_model", fake_evaluate_model)

    result = run_training(
        config,
        [],
        [],
        make_class_map(),
        make_data_summary(),
        expected_num_classes=NUM_CLASSES,
        pretrained=False,
        device=torch.device("cpu"),
    )

    assert result.stopped_early
    assert result.completed_epoch == 2
    assert [metric["epoch"] for metric in result.metrics] == [0, 1, 2]


def test_run_training_rejects_checkpoint_stage_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_tiny_model(monkeypatch)
    config = make_config(
        tmp_path,
        run_name="stage-mismatch",
        max_epochs=4,
        head_only_epochs=3,
    )
    checkpoint_path = tmp_path / "stage-mismatch.pt"
    model = TinyStageClassifier()
    optimizer = build_optimizer(model, config, TrainingStage.LAYER4_FINE_TUNING)
    scheduler = build_scheduler(
        optimizer,
        config,
        TrainingStage.LAYER4_FINE_TUNING,
    )
    save_checkpoint(
        checkpoint_path,
        completed_epoch=0,
        training_stage=TrainingStage.LAYER4_FINE_TUNING,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=build_grad_scaler(torch.device("cpu")),
        best_epoch=0,
        best_macro_f1=0.0,
        best_top1_accuracy=0.0,
        epochs_without_improvement=0,
        class_map_hash=hash_class_map(make_class_map()),
        split_hash="synthetic-split",
    )
    config.resume_from = checkpoint_path

    with pytest.raises(ValueError, match="training stage"):
        run_training(
            config,
            [],
            [],
            make_class_map(),
            make_data_summary(),
            expected_num_classes=NUM_CLASSES,
            pretrained=False,
            device=torch.device("cpu"),
        )
