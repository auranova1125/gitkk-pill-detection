"""탐지기와 분류기 pill recognition 추론 pipeline 테스트."""

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import pipeline.pill_recognition as pill_recognition_module
from configs.pill_recognition import PillRecognitionConfig
from pipeline import (
    PillRecognitionData,
    plot_pill_predictions,
    prepare_pill_recognition_data,
    recognize_pills,
    recognize_pills_from_config,
    save_pill_recognition_artifacts,
    save_submission_csv,
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


def test_prepare_pill_recognition_data_reuses_validated_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    loader = [make_batch()]

    def factory(*, config: PillRecognitionConfig, batch_size: int):
        assert config.run_name == "inference-test"
        assert batch_size == 4
        return loader

    monkeypatch.setattr(
        pill_recognition_module,
        "load_data_factory",
        lambda factory_path, *, setting_name: factory,
    )

    prepared_data = prepare_pill_recognition_data(config)

    assert prepared_data.data_loader is loader
    assert prepared_data.preview_batch is loader[0]
    assert prepared_data.device == torch.device("cpu")


def test_recognize_pills_from_config_saves_submission_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    classifier_class_map: dict[str, Any],
    detector_label_map: dict[str, Any],
) -> None:
    config = make_config(tmp_path)
    config.detector_label_map_path.write_text(
        json.dumps(detector_label_map),
        encoding="utf-8",
    )
    config.classifier_class_map_path.write_text(
        json.dumps(classifier_class_map),
        encoding="utf-8",
    )
    batch = make_batch()
    prepared_data = PillRecognitionData(
        data_loader=[batch],
        preview_batch=batch,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(
        pill_recognition_module,
        "load_detector_inference_model",
        lambda *args, **kwargs: (FixedDetector(), torch.device("cpu")),
    )
    monkeypatch.setattr(
        pill_recognition_module,
        "load_classifier_inference_model",
        lambda *args, **kwargs: (FixedClassifier(), torch.device("cpu")),
    )

    result = recognize_pills_from_config(config, data=prepared_data)

    with (config.run_dir / "submission.csv").open(
        encoding="utf-8", newline=""
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(result.predictions) == 1
    assert rows[0]["image_id"] == "7"
    assert rows[0]["category_id"] == "1901"


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


def test_save_submission_csv_uses_competition_columns(
    tmp_path: Path,
    classifier_class_map: dict[str, Any],
) -> None:
    result = recognize_pills(
        FixedDetector(),
        FixedClassifier(),
        [make_batch(image_id=10), make_batch(image_id=2)],
        classifier_class_map,
        torch.device("cpu"),
        score_threshold=0.5,
        detector_name="fixed-detector",
        show_progress=False,
    )

    submission_path = save_submission_csv(result, tmp_path)
    with submission_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert list(rows[0]) == [
        "annotation_id",
        "image_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "score",
    ]
    assert [row["annotation_id"] for row in rows] == ["1", "2"]
    assert [row["image_id"] for row in rows] == ["2", "10"]
    assert all(row["category_id"] == "1901" for row in rows)
    assert rows[0]["bbox_x"] == "2.0"
    assert rows[0]["bbox_y"] == "3.0"
    assert rows[0]["bbox_w"] == "16.0"
    assert rows[0]["bbox_h"] == "17.0"
    assert float(rows[0]["score"]) == pytest.approx(result.predictions[0].final_score)


def test_save_submission_csv_rejects_non_numeric_image_id(
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
    invalid_result = replace(
        result,
        predictions=(replace(result.predictions[0], image_id="image-7"),),
    )

    with pytest.raises(ValueError, match="ASCII integer string"):
        save_submission_csv(invalid_result, tmp_path)
