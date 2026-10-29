"""탐지기와 분류기 end-to-end 추론 pipeline 테스트."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from pipeline import (
    plot_end_to_end_predictions,
    run_end_to_end_inference,
    save_end_to_end_artifacts,
)


class FixedDetector(nn.Module):
    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "boxes": torch.tensor([[2.0, 3.0, 18.0, 20.0]]),
                "labels": torch.tensor([1], dtype=torch.int64),
                "scores": torch.tensor([0.8]),
            }
            for _ in images
        ]


class FixedClassifier(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((images.shape[0], 3), device=images.device)
        logits[:, 1] = 3.0
        return logits


def make_batch(*, image_id: int = 7) -> dict[str, Any]:
    return {
        "images": [torch.full((3, 24, 32), 0.5, dtype=torch.float32)],
        "meta": [
            {
                "image_id": image_id,
                "file_name": f"image-{image_id}.jpg",
                "image_path": f"images/image-{image_id}.jpg",
                "original_size": [24, 32],
            }
        ],
    }


def test_run_end_to_end_inference_combines_predictions(
    classifier_class_map: dict[str, Any],
) -> None:
    result = run_end_to_end_inference(
        FixedDetector(),
        FixedClassifier(),
        [make_batch()],
        classifier_class_map,
        torch.device("cpu"),
        score_threshold=0.5,
        detector_name="fixed-detector",
        show_progress=False,
    )

    expected_classifier_score = torch.softmax(
        torch.tensor([0.0, 3.0, 0.0]),
        dim=0,
    )[1].item()
    assert len(result.predictions) == 1
    assert not result.detector_errors
    assert not result.classifier_errors
    prediction = result.predictions[0]
    assert prediction.image_id == "7"
    assert prediction.detection_id == "7-0001"
    assert prediction.bbox_xywh == (2.0, 3.0, 16.0, 17.0)
    assert prediction.class_index == 1
    assert prediction.category_id == 1901
    assert prediction.item_seq == "item-1"
    assert prediction.class_name == "drug-1"
    assert prediction.classifier_score == pytest.approx(expected_classifier_score)
    assert prediction.final_score == pytest.approx(
        0.8 * expected_classifier_score,
    )


def test_run_end_to_end_inference_rejects_duplicate_image_ids(
    classifier_class_map: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="duplicate image_id"):
        run_end_to_end_inference(
            FixedDetector(),
            FixedClassifier(),
            [make_batch(), make_batch()],
            classifier_class_map,
            torch.device("cpu"),
            score_threshold=0.5,
            detector_name="fixed-detector",
            show_progress=False,
        )


def test_save_and_plot_end_to_end_artifacts(
    tmp_path: Path,
    classifier_class_map: dict[str, Any],
) -> None:
    result = run_end_to_end_inference(
        FixedDetector(),
        FixedClassifier(),
        [make_batch()],
        classifier_class_map,
        torch.device("cpu"),
        score_threshold=0.5,
        detector_name="fixed-detector",
        show_progress=False,
    )

    predictions_path, errors_path = save_end_to_end_artifacts(result, tmp_path)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    errors = json.loads(errors_path.read_text(encoding="utf-8"))
    figure = plot_end_to_end_predictions(
        make_batch()["images"][0],
        result.predictions,
        title="End-to-end result",
    )

    assert predictions[0]["class_name"] == "drug-1"
    assert errors == {"detector_errors": [], "classifier_errors": []}
    assert len(figure.axes[0].patches) == 1
