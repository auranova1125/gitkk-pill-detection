"""탐지기와 분류기 pill recognition 추론 pipeline 테스트."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import pipeline.pill_recognition as pill_recognition_module
from configs.pill_recognition import PillRecognitionConfig
from pipeline import (
    plot_pill_predictions,
    recognize_pills,
    save_pill_recognition_artifacts,
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


def make_config(tmp_path: Path) -> PillRecognitionConfig:
    return PillRecognitionConfig(
        run_name="inference-test",
        inference_factory_path=None,
        input_dir=tmp_path / "images",
        detector_checkpoint_path=tmp_path / "detector.pt",
        classifier_checkpoint_path=tmp_path / "classifier.pt",
        detector_label_map_path=tmp_path / "label_map.json",
        classifier_class_map_path=tmp_path / "class_map.json",
        output_dir=tmp_path / "outputs" / "inference",
        detector_name="detector-test",
        device="cpu",
    )


def test_recognize_pills_combines_predictions(
    classifier_class_map: dict[str, Any],
) -> None:
    result = recognize_pills(
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


def test_recognize_pills_rejects_duplicate_image_ids(
    classifier_class_map: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="duplicate image_id"):
        recognize_pills(
            FixedDetector(),
            FixedClassifier(),
            [make_batch(), make_batch()],
            classifier_class_map,
            torch.device("cpu"),
            score_threshold=0.5,
            detector_name="fixed-detector",
            show_progress=False,
        )


def test_prepare_pill_recognition_run_records_config_and_rejects_overwrite(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    run_dir = pill_recognition_module._prepare_pill_recognition_run(config)
    resolved_config = json.loads(
        (run_dir / "resolved_config.json").read_text(encoding="utf-8")
    )

    assert run_dir == config.run_dir
    assert resolved_config["run_name"] == "inference-test"
    assert resolved_config["output_dir"] == str(tmp_path / "outputs" / "inference")

    (run_dir / "pill_predictions.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="inference artifacts"):
        pill_recognition_module._prepare_pill_recognition_run(config)


def test_save_and_plot_pill_recognition_artifacts(
    tmp_path: Path,
    classifier_class_map: dict[str, Any],
) -> None:
    result = recognize_pills(
        FixedDetector(),
        FixedClassifier(),
        [make_batch()],
        classifier_class_map,
        torch.device("cpu"),
        score_threshold=0.5,
        detector_name="fixed-detector",
        show_progress=False,
    )

    predictions_path, errors_path = save_pill_recognition_artifacts(result, tmp_path)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    errors = json.loads(errors_path.read_text(encoding="utf-8"))
    figure = plot_pill_predictions(
        make_batch()["images"][0],
        result.predictions,
        title="Pill recognition result",
    )

    assert predictions[0]["class_name"] == "drug-1"
    assert errors == {"detector_errors": [], "classifier_errors": []}
    assert len(figure.axes[0].patches) == 1
