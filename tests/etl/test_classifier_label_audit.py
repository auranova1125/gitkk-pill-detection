"""분류 라벨 2차 검수의 사람 판정 보존과 조치 병합을 검사한다."""

from pathlib import Path

import pandas as pd
import pytest

from etl.classifier.label_audit.merge_review_actions import build_label_actions
from etl.classifier.label_audit.review_records import preserve_human_reviews


def _samples() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "a_0",
                "image_id": "a",
                "code": "K-000001",
                "name": "가정",
                "split": "train",
                "source": "aihub",
            },
            {
                "sample_id": "a_1",
                "image_id": "a",
                "code": "K-000002",
                "name": "나정",
                "split": "train",
                "source": "aihub",
            },
            {
                "sample_id": "b_0",
                "image_id": "b",
                "code": "K-000001",
                "name": "가정",
                "split": "val",
                "source": "kaggle",
            },
        ]
    )


def _position_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "a_0",
                "image_id": "a",
                "code": "K-000001",
                "판정": "오류",
                "조치": "이미지 제외",
                "메모": "위치 오류",
                "검수": "사람 확인 완료",
            }
        ]
    )


def _class_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "b_0",
                "image_id": "b",
                "code": "K-000001",
                "suspect_code": "K-000002",
                "판정": "오류",
                "조치": "라벨 교환 (수정)",
                "메모": "이름표 오류",
                "검수": "사람 확인 완료",
            }
        ]
    )


def test_preserves_reviews_and_keeps_reviewed_previous_candidate(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "reviews.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "a_0",
                "score": "0.2",
                "그림번호": "1",
                "그림파일": "review.jpg",
                "판정": "오류",
                "조치": "bbox 제외",
                "메모": "확인",
                "검수": "사람 확인 완료",
            },
            {
                "sample_id": "old_0",
                "score": "0.3",
                "그림번호": "2",
                "그림파일": "old.jpg",
                "판정": "정상",
                "조치": "",
                "메모": "이전 후보",
                "검수": "사람 확인 완료",
            },
        ]
    ).to_csv(review_path, index=False)
    candidates = pd.DataFrame(
        [
            {"sample_id": "a_0", "score": 0.1, "그림번호": 3, "그림파일": "new.jpg"},
            {"sample_id": "new_0", "score": 0.4, "그림번호": 4, "그림파일": "new2.jpg"},
        ]
    )

    result = preserve_human_reviews(candidates, review_path).set_index("sample_id")

    assert result.loc["a_0", "판정"] == "오류"
    assert result.loc["new_0", "판정"] == ""
    assert result.loc["old_0", "판정"] == "정상"
    assert result.loc["old_0", "그림파일"] == ""


def test_builds_image_exclusion_and_label_swap_actions() -> None:
    actions = build_label_actions(
        _samples(),
        _position_candidates(),
        _class_candidates(),
    ).set_index("sample_id")

    assert set(actions.index) == {"a_0", "a_1", "b_0"}
    assert actions.loc["a_0", "action"] == "이미지 제외"
    assert actions.loc["a_1", "action"] == "이미지 제외"
    assert actions.loc["b_0", "action"] == "라벨 교환 (수정)"
    assert actions.loc["b_0", "new_code"] == "K-000002"


def test_rejects_incomplete_review() -> None:
    candidates = _position_candidates()
    candidates.loc[0, "검수"] = ""

    with pytest.raises(ValueError, match="incomplete reviews"):
        build_label_actions(_samples(), candidates, _class_candidates())
