"""탐지기 학습과 평가 루프 테스트."""

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from detector import DetectorConfig
from detector.evaluate import evaluate_model, filter_predictions_by_score
from detector.train import (
    build_optimizer,
    build_scheduler,
    train_one_epoch,
)
from detector.training_data import move_detection_batch_to_device
from utils.runtime import build_grad_scaler


class TinyTrainingDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        image_term = torch.stack([image.mean() for image in images]).mean()
        target_term = torch.stack([target["boxes"].mean() for target in targets]).mean()
        return {
            "classification": (self.weight - image_term).square(),
            "bbox_regression": self.weight.square() * target_term / 100.0,
        }


class NonFiniteLossDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        del images, targets
        return {"classification": self.weight * torch.tensor(float("nan"))}


class FixedEvaluationDetector(nn.Module):
    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        assert len(images) == 2
        return [
            {
                "boxes": torch.tensor(
                    [[5.0, 5.0, 25.0, 25.0], [30.0, 30.0, 45.0, 45.0]],
                    device=images[0].device,
                ),
                "labels": torch.ones(2, dtype=torch.int64, device=images[0].device),
                "scores": torch.tensor([0.9, 0.2], device=images[0].device),
            },
            {
                "boxes": torch.tensor(
                    [[10.0, 8.0, 30.0, 38.0]],
                    device=images[1].device,
                ),
                "labels": torch.ones(1, dtype=torch.int64, device=images[1].device),
                "scores": torch.tensor([0.1], device=images[1].device),
            },
        ]


def make_config(tmp_path: Path) -> DetectorConfig:
    return DetectorConfig(
        run_name="training-test",
        output_dir=tmp_path / "outputs",
        label_map_path=tmp_path / "label_map.json",
        data_summary_path=tmp_path / "data_summary.json",
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        score_threshold=0.5,
        nms_threshold=0.45,
        early_stopping_patience=3,
        max_epochs=10,
    )


def make_batch() -> dict[str, Any]:
    first_boxes = torch.tensor(
        [[5.0, 5.0, 25.0, 25.0]],
        dtype=torch.float32,
    )
    second_boxes = torch.tensor(
        [[10.0, 8.0, 30.0, 38.0]],
        dtype=torch.float32,
    )
    return {
        "images": [
            torch.full((3, 48, 64), 0.25, dtype=torch.float32),
            torch.full((3, 40, 56), 0.5, dtype=torch.float32),
        ],
        "targets": [
            {
                "boxes": first_boxes,
                "labels": torch.ones(1, dtype=torch.int64),
                "image_id": 0,
                "area": torch.tensor([400.0], dtype=torch.float32),
                "iscrowd": torch.zeros(1, dtype=torch.int64),
            },
            {
                "boxes": second_boxes,
                "labels": torch.ones(1, dtype=torch.int64),
                "image_id": 1,
                "area": torch.tensor([600.0], dtype=torch.float32),
                "iscrowd": torch.zeros(1, dtype=torch.int64),
            },
        ],
        "meta": [{"image_id": 0}, {"image_id": 1}],
    }


def test_build_optimizer_and_scheduler(tmp_path: Path) -> None:
    model = TinyTrainingDetector()
    config = make_config(tmp_path)

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["lr"] == pytest.approx(config.learning_rate)
    assert optimizer.defaults["momentum"] == pytest.approx(config.momentum)
    assert optimizer.defaults["weight_decay"] == pytest.approx(config.weight_decay)
    assert scheduler.T_max == config.max_epochs


def test_build_optimizer_rejects_model_without_trainable_parameters(
    tmp_path: Path,
) -> None:
    model = TinyTrainingDetector()
    model.weight.requires_grad = False

    with pytest.raises(ValueError, match="trainable parameter"):
        build_optimizer(model, make_config(tmp_path))


def test_build_grad_scaler_is_disabled_without_cuda() -> None:
    assert not build_grad_scaler(torch.device("cpu")).is_enabled()
    assert not build_grad_scaler(torch.device("mps")).is_enabled()


def test_move_detection_batch_to_cpu_preserves_non_tensor_fields() -> None:
    images, targets = move_detection_batch_to_device(
        make_batch(),
        torch.device("cpu"),
    )

    assert all(image.device.type == "cpu" for image in images)
    assert all(target["boxes"].device.type == "cpu" for target in targets)
    assert targets[0]["image_id"] == 0


def test_train_one_epoch_updates_model_and_aggregates_losses(
    tmp_path: Path,
) -> None:
    model = TinyTrainingDetector()
    optimizer = build_optimizer(model, make_config(tmp_path))
    initial_weight = model.weight.detach().clone()

    result = train_one_epoch(
        model,
        [make_batch(), make_batch()],
        optimizer,
        torch.device("cpu"),
    )

    assert result.num_batches == 2
    assert result.num_images == 4
    assert result.loss > 0
    assert set(result.loss_components) == {"bbox_regression", "classification"}
    assert sum(result.loss_components.values()) == pytest.approx(result.loss)
    assert not torch.equal(initial_weight, model.weight)


def test_train_one_epoch_shows_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = TinyTrainingDetector()
    optimizer = build_optimizer(model, make_config(tmp_path))

    train_one_epoch(
        model,
        [make_batch()],
        optimizer,
        torch.device("cpu"),
        show_progress=True,
        progress_description="Detector progress test",
    )

    assert "Detector progress test" in capsys.readouterr().err


def test_train_one_epoch_rejects_empty_loader(tmp_path: Path) -> None:
    model = TinyTrainingDetector()
    optimizer = build_optimizer(model, make_config(tmp_path))

    with pytest.raises(ValueError, match="at least one batch"):
        train_one_epoch(model, [], optimizer, torch.device("cpu"))


def test_train_one_epoch_rejects_non_finite_loss(tmp_path: Path) -> None:
    model = NonFiniteLossDetector()
    optimizer = build_optimizer(model, make_config(tmp_path))

    with pytest.raises(FloatingPointError, match="finite"):
        train_one_epoch(model, [make_batch()], optimizer, torch.device("cpu"))


def test_filter_predictions_by_score_keeps_aligned_tensors() -> None:
    predictions = FixedEvaluationDetector()(make_batch()["images"])

    filtered = filter_predictions_by_score(predictions, score_threshold=0.5)

    assert filtered[0]["boxes"].shape == (1, 4)
    assert filtered[0]["labels"].tolist() == [1]
    assert filtered[0]["scores"].tolist() == pytest.approx([0.9])
    assert filtered[1]["boxes"].shape == (0, 4)


def test_evaluate_model_computes_metrics_and_threshold_analysis() -> None:
    result = evaluate_model(
        FixedEvaluationDetector(),
        [make_batch()],
        torch.device("cpu"),
        score_threshold=0.5,
    )

    assert 0.0 <= result.map <= 1.0
    assert 0.0 <= result.map_50 <= 1.0
    assert 0.0 <= result.map_75 <= 1.0
    assert 0.0 <= result.map_75_95 <= 1.0
    assert 0.0 <= result.mean_average_recall <= 1.0
    assert result.recall_before_threshold == pytest.approx(1.0)
    assert result.recall_after_threshold == pytest.approx(0.5)
    assert result.mean_count_error_before_threshold == pytest.approx(0.5)
    assert result.mean_count_error_after_threshold == pytest.approx(0.5)
    assert result.num_images == 2
    assert result.num_ground_truths == 2
    assert result.num_predictions_before_threshold == 3
    assert result.num_predictions_after_threshold == 1


def test_evaluate_model_rejects_empty_loader() -> None:
    with pytest.raises(ValueError, match="at least one batch"):
        evaluate_model(
            FixedEvaluationDetector(),
            [],
            torch.device("cpu"),
            score_threshold=0.5,
        )
