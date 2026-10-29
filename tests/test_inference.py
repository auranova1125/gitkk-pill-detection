"""탐지 crop 분류 및 score 결합 테스트."""

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import classifier.inference as inference_module
from classifier.checkpoint import hash_class_map, save_checkpoint
from classifier.inference import (
    load_inference_model,
    predict_detections,
    validate_classifier_crop_batch,
)
from classifier.model import TrainingStage
from classifier.train import build_grad_scaler

NUM_CLASSES = 3


class FixedLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fixed_logits[: images.shape[0]]


class TinyInferenceModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(12, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


def make_class_map(num_classes: int = NUM_CLASSES) -> dict[str, Any]:
    return {
        "num_classes": num_classes,
        "classes": [
            {
                "class_index": index,
                "category_id": 1900 + index,
                "item_seq": f"item-{index}",
                "name": f"합성 품목 {index}",
            }
            for index in range(num_classes)
        ],
    }


def make_batch(batch_size: int = 2) -> dict[str, Any]:
    return {
        "images": torch.zeros(batch_size, 3, 224, 224, dtype=torch.float32),
        "meta": [
            {
                "image_id": f"image-{index}",
                "detection_id": f"detection-{index}",
                "bbox_xywh": [10.0, 20.0, 30.0, 40.0],
                "detector_score": 1.0 if index == 0 else 0.5,
                "detector_name": "synthetic-detector",
            }
            for index in range(batch_size)
        ],
    }


def test_validate_classifier_crop_batch_normalizes_metadata() -> None:
    normalized_meta = validate_classifier_crop_batch(make_batch())

    assert len(normalized_meta) == 2
    assert normalized_meta[0]["bbox_xywh"] == (10.0, 20.0, 30.0, 40.0)
    assert normalized_meta[1]["detector_score"] == 0.5


def test_predict_detections_maps_classes_and_combines_scores() -> None:
    logits = torch.tensor([[0.0, 3.0, 1.0], [0.0, 1.0, 4.0]])
    model = FixedLogitModel(logits)
    batch = make_batch()

    result = predict_detections(
        model,
        [batch],
        make_class_map(),
        torch.device("cpu"),
    )

    expected_scores = torch.softmax(logits, dim=1).max(dim=1).values
    assert len(result.predictions) == 2
    assert not result.errors
    assert result.predictions[0].class_index == 1
    assert result.predictions[0].category_id == 1901
    assert result.predictions[0].item_seq == "item-1"
    assert result.predictions[0].classifier_score == pytest.approx(
        expected_scores[0].item()
    )
    assert result.predictions[0].final_score == pytest.approx(expected_scores[0].item())
    assert result.predictions[1].class_index == 2
    assert result.predictions[1].final_score == pytest.approx(
        0.5 * expected_scores[1].item()
    )


def test_predict_detections_keeps_prediction_when_mapping_is_missing() -> None:
    class_map = make_class_map()
    class_map["classes"] = class_map["classes"][:2]
    model = FixedLogitModel(torch.tensor([[0.0, 1.0, 4.0]]))

    result = predict_detections(
        model,
        [make_batch(batch_size=1)],
        class_map,
        torch.device("cpu"),
    )

    assert len(result.predictions) == 1
    assert result.predictions[0].class_index == 2
    assert result.predictions[0].category_id is None
    assert result.predictions[0].item_seq is None
    assert len(result.errors) == 1
    assert result.errors[0].detection_id == "detection-0"


@pytest.mark.parametrize("detector_score", [-0.1, 1.1, float("nan")])
def test_validate_classifier_crop_batch_rejects_invalid_detector_score(
    detector_score: float,
) -> None:
    batch = make_batch(batch_size=1)
    batch["meta"][0]["detector_score"] = detector_score

    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_classifier_crop_batch(batch)


def test_predict_detections_rejects_wrong_logit_shape() -> None:
    model = FixedLogitModel(torch.zeros(1, 4))

    with pytest.raises(ValueError, match="logits must have shape"):
        predict_detections(
            model,
            [make_batch(batch_size=1)],
            make_class_map(),
            torch.device("cpu"),
        )


def test_predict_detections_accepts_empty_loader() -> None:
    result = predict_detections(
        FixedLogitModel(torch.zeros(1, NUM_CLASSES)),
        [],
        make_class_map(),
        torch.device("cpu"),
    )

    assert not result.predictions
    assert not result.errors


def test_load_inference_model_restores_checkpoint_without_pretrained_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class_map = make_class_map()
    original_model = TinyInferenceModel(NUM_CLASSES)
    checkpoint_path = tmp_path / "best.pt"
    optimizer = torch.optim.AdamW(original_model.parameters())
    save_checkpoint(
        checkpoint_path,
        completed_epoch=0,
        training_stage=TrainingStage.HEAD_ONLY,
        model=original_model,
        optimizer=optimizer,
        scheduler=None,
        scaler=build_grad_scaler(torch.device("cpu")),
        best_epoch=0,
        best_macro_f1=0.5,
        best_top1_accuracy=0.5,
        epochs_without_improvement=0,
        class_map_hash=hash_class_map(class_map),
        split_hash="synthetic-split",
    )
    pretrained_values: list[bool] = []

    def build_tiny_model(num_classes: int, pretrained: bool = True) -> nn.Module:
        pretrained_values.append(pretrained)
        return TinyInferenceModel(num_classes)

    monkeypatch.setattr(inference_module, "build_model", build_tiny_model)

    restored_model, device = load_inference_model(
        checkpoint_path,
        class_map,
        expected_num_classes=NUM_CLASSES,
        device=torch.device("cpu"),
    )

    assert device == torch.device("cpu")
    assert pretrained_values == [False]
    assert all(
        torch.equal(original_parameter, restored_parameter)
        for original_parameter, restored_parameter in zip(
            original_model.parameters(),
            restored_model.parameters(),
            strict=True,
        )
    )
