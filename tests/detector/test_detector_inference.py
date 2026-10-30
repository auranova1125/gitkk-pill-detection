"""탐지기 추론과 알약 인식 연결 테스트."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import detector.inference as inference_module
from classifier.inference import combine_prediction_scores
from detector.checkpoint import hash_label_map, save_checkpoint
from detector.inference import (
    DetectorPrediction,
    load_detector_inference_model,
    plot_detection_comparison,
    predict_detector,
    save_inference_artifacts,
    validate_inference_batch,
)
from pilldata.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PADDING_COLOR_RGB,
)
from pipeline.pill_recognition import build_classifier_batch


class FixedDetector(nn.Module):
    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "boxes": torch.tensor(
                    [
                        [2.0, 3.0, 12.0, 13.0],
                        [0.0, 0.0, 5.0, 5.0],
                        [1.0, 1.0, 2.0, 2.0],
                        [-5.0, -4.0, 40.0, 25.0],
                        [31.0, 0.0, 40.0, 10.0],
                    ],
                    dtype=torch.float32,
                    device=images[0].device,
                ),
                "labels": torch.tensor(
                    [1, 1, 2, 1, 1],
                    dtype=torch.int64,
                    device=images[0].device,
                ),
                "scores": torch.tensor(
                    [0.9, 0.2, 0.8, 0.7, 0.6],
                    dtype=torch.float32,
                    device=images[0].device,
                ),
            }
            for _ in images
        ]


class EmptyDetector(nn.Module):
    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty(0, dtype=torch.int64),
                "scores": torch.empty(0, dtype=torch.float32),
            }
            for _ in images
        ]


class TinyInferenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


def make_inference_batch(image_id: int = 7) -> dict[str, Any]:
    return {
        "images": [torch.full((3, 20, 30), 0.5, dtype=torch.float32)],
        "meta": [
            {
                "image_id": image_id,
                "file_name": f"image-{image_id}.png",
                "image_path": f"/synthetic/image-{image_id}.png",
                "original_size": [20, 30],
            }
        ],
    }


def make_prediction(
    *,
    bbox_xyxy: tuple[float, float, float, float] = (2.0, 2.0, 10.0, 6.0),
    detection_id: str = "7-0001",
    score: float = 0.8,
) -> DetectorPrediction:
    x1, y1, x2, y2 = bbox_xyxy
    return DetectorPrediction(
        image_id=7,
        detection_id=detection_id,
        bbox_xyxy=bbox_xyxy,
        bbox_xywh=(x1, y1, x2 - x1, y2 - y1),
        label=1,
        label_name="pill",
        detector_score=score,
        detector_name="synthetic-detector",
    )


def test_validate_inference_batch_accepts_contract() -> None:
    images, metadata = validate_inference_batch(make_inference_batch())

    assert len(images) == 1
    assert metadata[0]["image_id"] == 7
    assert metadata[0]["original_size"] == [20, 30]


def test_validate_inference_batch_rejects_size_mismatch() -> None:
    batch = make_inference_batch()
    batch["meta"][0]["original_size"] = [10, 10]

    with pytest.raises(ValueError, match="must match"):
        validate_inference_batch(batch)


def test_predict_detector_filters_and_records_errors() -> None:
    result = predict_detector(
        FixedDetector(),
        [make_inference_batch()],
        torch.device("cpu"),
        score_threshold=0.5,
        detector_name="ssdlite320_mobilenet_v3_large",
    )

    assert len(result.predictions) == 2
    assert len(result.errors) == 2
    assert [item.detection_id for item in result.predictions] == [
        "7-0001",
        "7-0002",
    ]
    assert result.predictions[0].bbox_xyxy == pytest.approx((2.0, 3.0, 12.0, 13.0))
    assert result.predictions[0].bbox_xywh == pytest.approx((2.0, 3.0, 10.0, 10.0))
    assert result.predictions[1].bbox_xyxy == pytest.approx((0.0, 0.0, 30.0, 20.0))
    assert "label 1" in result.errors[0].message
    assert "zero area" in result.errors[1].message


def test_predict_detector_accepts_empty_predictions() -> None:
    result = predict_detector(
        EmptyDetector(),
        [make_inference_batch()],
        torch.device("cpu"),
        score_threshold=0.5,
        detector_name="empty-detector",
    )

    assert result.predictions == ()
    assert result.errors == ()


def test_predict_detector_rejects_duplicate_image_ids() -> None:
    batch = make_inference_batch()

    with pytest.raises(ValueError, match="duplicate image_id"):
        predict_detector(
            EmptyDetector(),
            [batch, batch],
            torch.device("cpu"),
            score_threshold=0.5,
            detector_name="empty-detector",
        )


def test_save_inference_artifacts_writes_json(tmp_path: Path) -> None:
    result = predict_detector(
        FixedDetector(),
        [make_inference_batch()],
        torch.device("cpu"),
        score_threshold=0.5,
        detector_name="synthetic-detector",
    )

    predictions_path, errors_path = save_inference_artifacts(result, tmp_path)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    errors = json.loads(errors_path.read_text(encoding="utf-8"))

    assert predictions[0]["bbox_xywh"] == [2.0, 3.0, 10.0, 10.0]
    assert predictions[0]["label_name"] == "pill"
    assert len(errors) == 2


def test_plot_detection_comparison_draws_ground_truth_and_prediction() -> None:
    from matplotlib import pyplot as plt

    image = torch.zeros(3, 20, 30, dtype=torch.float32)
    target = {"boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0]], dtype=torch.float32)}

    figure = plot_detection_comparison(
        image,
        [make_prediction()],
        target=target,
        title="synthetic",
    )

    assert len(figure.axes[0].patches) == 2
    assert figure.axes[0].get_title() == "synthetic"
    plt.close(figure)


def test_load_inference_model_restores_matching_backbone_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    detector_label_map: dict[str, Any],
) -> None:
    label_map = detector_label_map
    model = TinyInferenceModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "detector.pt"
    save_checkpoint(
        checkpoint_path,
        completed_epoch=0,
        model_name="fake_detector",
        model_config={
            "num_classes": 2,
            "pretrained_backbone": True,
            "nms_threshold": 0.45,
        },
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        best_epoch=0,
        best_map_75_95=0.5,
        best_mean_average_recall=0.6,
        epochs_without_improvement=0,
        label_map_hash=hash_label_map(label_map),
        split_hash="synthetic-split",
    )
    captured: dict[str, Any] = {}

    def build_tiny_model(*args: Any, **kwargs: Any) -> nn.Module:
        captured.update(kwargs)
        return TinyInferenceModel()

    monkeypatch.setattr(inference_module, "build_detector_model", build_tiny_model)

    loaded_model, device = load_detector_inference_model(
        checkpoint_path,
        label_map,
        device=torch.device("cpu"),
    )

    assert device.type == "cpu"
    assert isinstance(loaded_model, TinyInferenceModel)
    assert loaded_model.weight.item() == pytest.approx(1.0)
    assert not loaded_model.training
    assert captured == {
        "num_classes": 2,
        "pretrained_backbone": True,
        "nms_threshold": 0.45,
    }


def test_build_classifier_batch_matches_classifier_contract() -> None:
    image = torch.full((3, 8, 12), 0.25, dtype=torch.float32)

    batch = build_classifier_batch(image, [make_prediction()])

    assert batch["images"].dtype == torch.float32
    assert batch["images"].shape == (1, 3, 224, 224)
    assert torch.isfinite(batch["images"]).all()
    assert batch["meta"] == [
        {
            "image_id": "7",
            "detection_id": "7-0001",
            "bbox_xywh": [2.0, 2.0, 8.0, 4.0],
            "detector_score": 0.8,
            "detector_name": "synthetic-detector",
        }
    ]

    expected_padding = torch.tensor(
        [
            (PADDING_COLOR_RGB[channel] / 255.0 - IMAGENET_MEAN[channel])
            / IMAGENET_STD[channel]
            for channel in range(3)
        ]
    )
    torch.testing.assert_close(
        batch["images"][0, :, 0, 0],
        expected_padding,
        rtol=1e-5,
        atol=1e-6,
    )


def test_build_classifier_batch_clips_bbox_and_handles_empty_predictions() -> None:
    image = torch.zeros(3, 8, 12, dtype=torch.float32)
    clipped_batch = build_classifier_batch(
        image,
        [make_prediction(bbox_xyxy=(-2.0, -1.0, 6.0, 4.0))],
    )
    empty_batch = build_classifier_batch(image, [])

    assert clipped_batch["meta"][0]["bbox_xywh"] == [0.0, 0.0, 6.0, 4.0]
    assert empty_batch["images"].shape == (0, 3, 224, 224)
    assert empty_batch["meta"] == []


def test_build_classifier_batch_rejects_duplicate_detection_ids() -> None:
    image = torch.zeros(3, 8, 12, dtype=torch.float32)

    with pytest.raises(ValueError, match="duplicate detection_id"):
        build_classifier_batch(
            image,
            [make_prediction(), make_prediction()],
        )


@pytest.mark.parametrize(
    ("detector_score", "classifier_score", "expected"),
    [
        (0.8, 0.5, 0.4),
        (1.0, 1.0, 1.0),
        (0.0, 0.7, 0.0),
    ],
)
def test_combine_scores(
    detector_score: float,
    classifier_score: float,
    expected: float,
) -> None:
    assert combine_prediction_scores(detector_score, classifier_score) == pytest.approx(
        expected
    )


def test_combine_scores_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="between"):
        combine_prediction_scores(1.1, 0.5)
