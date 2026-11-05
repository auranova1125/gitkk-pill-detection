"""오분류·미탐·GT 클래스 평균·threshold를 검증한다."""

import pytest

from pipeline.yolo_e2e import score_rows


def record(image_id=1, label=250, dataset="select"):
    return {
        "image_id": image_id,
        "boxes": [[0, 0, 10, 10]],
        "category_ids": [label],
        "dataset": dataset,
    }


def pred(label=250, score=0.9, final=0.81):
    return {
        "bbox_xyxy": [0, 0, 10, 10],
        "category_id": label,
        "detector_score": score,
        "classifier_score": final / score,
        "final_score": final,
    }


def test_perfect_boxes_with_correct_classes_score_one():
    result = score_rows([record()], {1: [pred()]}, threshold_on="detector")
    assert result["map_75_95"] == pytest.approx(1)
    assert result["ap_95"] == pytest.approx(1)


def test_wrong_class_fails_e2e_but_not_detector_only():
    rows, predictions = [record()], {1: [pred(label=573)]}
    result = score_rows(rows, predictions, threshold_on="detector")
    assert result["map_75_95"] == 0
    assert result["num_gt_classes"] == 1
    assert result["per_class"][0]["category_id"] == 250
    detection = score_rows(
        rows, predictions, threshold_on="detector", ignore_classes=True
    )
    assert detection["map_75_95"] == pytest.approx(1)


def test_missing_image_predictions_remain_false_negative():
    rows = [record(1, 250), record(2, 573)]
    result = score_rows(rows, {1: [pred()], 2: []}, threshold_on="detector")
    assert result["num_images"] == 2
    assert result["num_gt_classes"] == 2
    assert result["map_75_95"] == pytest.approx(0.5)
    assert result["ap_95"] == pytest.approx(0.5)


def test_detector_and_product_threshold_are_different():
    rows, predictions = [record()], {1: [pred(final=0.09)]}
    assert (
        score_rows(rows, predictions, threshold_on="detector")["num_predictions"] == 1
    )
    assert score_rows(rows, predictions, threshold_on="final")["num_predictions"] == 0
    assert score_rows(rows, predictions, threshold_on="final")["map_75_95"] == 0


def test_missing_cache_is_error_not_silently_skipped():
    with pytest.raises(ValueError, match="기록이 없는"):
        score_rows([record()], {}, threshold_on="detector")


def test_false_positive_counts_on_empty_gt_image():
    rows = [record(), {"image_id": 2, "boxes": [], "category_ids": []}]
    result = score_rows(
        rows, {1: [pred(final=0.5)], 2: [pred(final=0.9)]}, threshold_on="detector"
    )
    assert result["map_75_95"] < 1
    assert result["num_predictions"] == 2
