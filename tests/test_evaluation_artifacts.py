"""Validation 예측 및 오류 artifact 테스트."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from classifier.evaluate import evaluate_with_predictions, save_validation_artifacts

NUM_CLASSES = 3


class FixedValidationModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fixed_logits[: images.shape[0]]


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


def make_validation_batch() -> dict[str, Any]:
    return {
        "images": torch.zeros(2, 3, 224, 224, dtype=torch.float32),
        "targets": torch.tensor([0, 1], dtype=torch.int64),
        "meta": [
            {
                "sample_id": f"sample-{index}",
                "image_id": f"image-{index}",
                "image_path": f"/synthetic/image-{index}.png",
                "category_id": 1900 + index,
                "item_seq": f"item-{index}",
                "bbox_xywh": [10.0, 20.0, 30.0, 40.0],
                "group_id": f"group-{index}",
                "drug_dir": None,
            }
            for index in range(2)
        ],
    }


def test_evaluate_with_predictions_maps_and_records_errors() -> None:
    model = FixedValidationModel(
        torch.tensor(
            [
                [5.0, 1.0, 0.0],
                [4.0, 1.0, 0.0],
            ]
        )
    )

    artifacts = evaluate_with_predictions(
        model,
        [make_validation_batch()],
        make_class_map(),
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        expected_num_classes=NUM_CLASSES,
    )

    assert artifacts.metrics.top1_accuracy == pytest.approx(0.5)
    assert len(artifacts.predictions) == 2
    assert artifacts.predictions[0].correct
    assert artifacts.predictions[0].predicted_category_id == 1900
    assert not artifacts.predictions[1].correct
    assert artifacts.predictions[1].target_item_seq == "item-1"
    assert artifacts.predictions[1].predicted_item_seq == "item-0"
    assert len(artifacts.error_samples) == 1
    assert artifacts.error_samples[0].sample_id == "sample-1"


def test_save_validation_artifacts_writes_json(tmp_path: Path) -> None:
    model = FixedValidationModel(torch.tensor([[5.0, 1.0, 0.0], [4.0, 1.0, 0.0]]))
    artifacts = evaluate_with_predictions(
        model,
        [make_validation_batch()],
        make_class_map(),
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        expected_num_classes=NUM_CLASSES,
    )

    predictions_path, errors_path = save_validation_artifacts(artifacts, tmp_path)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    errors = json.loads(errors_path.read_text(encoding="utf-8"))

    assert len(predictions) == 2
    assert predictions[0]["bbox_xywh"] == [10.0, 20.0, 30.0, 40.0]
    assert len(errors) == 1
    assert errors[0]["sample_id"] == "sample-1"
