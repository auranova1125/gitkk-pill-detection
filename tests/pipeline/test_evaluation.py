"""알약 탐지·분류 통합 검증 지표 테스트."""

import csv
import json
from pathlib import Path

import pytest

from pipeline import (
    PillPrediction,
    PillRecognitionResult,
    evaluate_pill_recognition,
    save_pill_recognition_evaluation,
)


def make_prediction(
    detection_id: str,
    bbox_xywh: tuple[float, float, float, float],
    category_id: int,
    *,
    score: float = 0.9,
) -> PillPrediction:
    return PillPrediction(
        image_id="1",
        detection_id=detection_id,
        bbox_xywh=bbox_xywh,
        detector_score=score,
        detector_name="fixed-detector",
        class_index=0,
        category_id=category_id,
        item_seq=f"item-{category_id}",
        class_name=f"drug-{category_id}",
        classifier_score=score,
        final_score=score,
    )


def make_validation_loader() -> list[dict[str, object]]:
    return [
        {
            "images": [],
            "targets": [],
            "meta": [
                {
                    "image_id": 1,
                    "transformed_boxes_xyxy": [
                        [0.0, 0.0, 10.0, 10.0],
                        [20.0, 20.0, 30.0, 30.0],
                        [40.0, 40.0, 50.0, 50.0],
                    ],
                    "category_ids": [10, 20, 30],
                }
            ],
        }
    ]


def test_evaluate_pill_recognition_separates_error_types() -> None:
    result = PillRecognitionResult(
        predictions=(
            make_prediction("correct", (0.0, 0.0, 10.0, 10.0), 10),
            make_prediction("wrong-class", (20.0, 20.0, 10.0, 10.0), 99),
            make_prediction("false-positive", (60.0, 60.0, 10.0, 10.0), 40),
        ),
        detector_errors=(),
        classifier_errors=(),
    )

    evaluation, details = evaluate_pill_recognition(
        result,
        make_validation_loader(),
    )

    assert evaluation.num_images == 1
    assert evaluation.num_ground_truths == 3
    assert evaluation.num_predictions == 3
    assert evaluation.num_detection_matches == 2
    assert evaluation.num_correct_predictions == 1
    assert evaluation.num_misclassified == 1
    assert evaluation.num_false_positives == 1
    assert evaluation.num_false_negatives == 1
    assert evaluation.detection_precision == pytest.approx(2 / 3)
    assert evaluation.detection_recall == pytest.approx(2 / 3)
    assert evaluation.matched_classification_accuracy == pytest.approx(1 / 2)
    assert evaluation.end_to_end_precision == pytest.approx(1 / 3)
    assert evaluation.end_to_end_recall == pytest.approx(1 / 3)
    assert {detail.status for detail in details} == {
        "correct",
        "misclassified",
        "false_positive",
        "false_negative",
    }


def test_evaluate_pill_recognition_reports_each_high_iou_map() -> None:
    result = PillRecognitionResult(
        predictions=(make_prediction("near-perfect", (0.0, 0.0, 9.2, 10.0), 10),),
        detector_errors=(),
        classifier_errors=(),
    )
    validation_loader = [
        {
            "images": [],
            "targets": [],
            "meta": [
                {
                    "image_id": 1,
                    "transformed_boxes_xyxy": [[0.0, 0.0, 10.0, 10.0]],
                    "category_ids": [10],
                }
            ],
        }
    ]

    evaluation, _ = evaluate_pill_recognition(result, validation_loader)

    assert evaluation.map_75 == pytest.approx(1.0)
    assert evaluation.map_80 == pytest.approx(1.0)
    assert evaluation.map_85 == pytest.approx(1.0)
    assert evaluation.map_90 == pytest.approx(1.0)
    assert evaluation.map_95 == pytest.approx(0.0)
    assert evaluation.map_75_95 == pytest.approx(0.8)


def test_save_pill_recognition_evaluation_writes_artifacts(
    tmp_path: Path,
) -> None:
    result = PillRecognitionResult(
        predictions=(make_prediction("correct", (0.0, 0.0, 10.0, 10.0), 10),),
        detector_errors=(),
        classifier_errors=(),
    )
    evaluation, details = evaluate_pill_recognition(
        result,
        make_validation_loader(),
    )

    metrics_path, details_path = save_pill_recognition_evaluation(
        evaluation,
        details,
        tmp_path,
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with details_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert metrics["num_images"] == 1
    assert len(rows) == 3
    assert {row["status"] for row in rows} == {
        "correct",
        "false_negative",
    }
