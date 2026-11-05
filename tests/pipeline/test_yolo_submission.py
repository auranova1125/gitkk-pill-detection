"""제출 파일 컬럼, 좌표, 템플릿 대조를 검증한다."""

import csv

import pytest

from pipeline.pill_recognition import SUBMISSION_FIELD_NAMES
from pipeline.yolo_submission import validate_submission


def make_csv(path, x=1):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_FIELD_NAMES)
        writer.writeheader()
        writer.writerow(
            dict(
                annotation_id=1,
                image_id=42,
                category_id=250,
                bbox_x=x,
                bbox_y=2,
                bbox_w=5,
                bbox_h=6,
                score=0.9,
            )
        )


def test_valid_submission_and_zero_prediction_image(tmp_path):
    path = tmp_path / "submission.csv"
    make_csv(path)
    result = validate_submission(path, {42: (20, 20), 43: (20, 20)}, {250})
    assert result["num_predictions"] == 1
    assert result["num_images_without_predictions"] == 1
    assert result["sample_columns_verified"] is False


def test_sample_columns_must_match(tmp_path):
    path = tmp_path / "submission.csv"
    make_csv(path)
    sample = tmp_path / "sample.csv"
    sample.write_text("image_id,prediction\n")
    with pytest.raises(ValueError, match="sample_submission"):
        validate_submission(path, {42: (20, 20)}, {250}, sample)


def test_bbox_must_be_inside_image(tmp_path):
    path = tmp_path / "submission.csv"
    make_csv(path, x=19)
    with pytest.raises(ValueError, match="경계"):
        validate_submission(path, {42: (20, 20)}, {250})
