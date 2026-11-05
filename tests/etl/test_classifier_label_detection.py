"""분류 라벨 오류 후보의 위치·embedding 점수 계산을 검사한다."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from etl.classifier.label_audit.detect_bbox_errors import (
    bbox_iou,
    position_priority,
)
from etl.classifier.label_audit.detect_class_errors import (
    build_class_scores,
    compare_group_embeddings,
)


def test_bbox_iou_and_position_priority() -> None:
    assert bbox_iou([0, 0, 10, 10], [5, 0, 10, 10]) == 1 / 3
    assert (
        position_priority(SimpleNamespace(fg_in=0.1, match_iou=0.0, unmatched_blobs=1))
        == 1
    )
    assert (
        position_priority(SimpleNamespace(fg_in=0.8, match_iou=0.0, unmatched_blobs=0))
        == 2
    )
    assert (
        position_priority(SimpleNamespace(fg_in=0.8, match_iou=0.55, unmatched_blobs=0))
        == 3
    )


def _embedding_samples() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "a0",
                "image_id": "a",
                "group_id": "g",
                "code": "K-000001",
                "name": "가정",
                "split": "train",
                "source": "aihub",
                "class_index": 0,
                "crop_path": "a0.png",
            },
            {
                "sample_id": "a1",
                "image_id": "a",
                "group_id": "g",
                "code": "K-000002",
                "name": "나정",
                "split": "train",
                "source": "aihub",
                "class_index": 1,
                "crop_path": "a1.png",
            },
            {
                "sample_id": "b0",
                "image_id": "b",
                "group_id": "g",
                "code": "K-000001",
                "name": "가정",
                "split": "train",
                "source": "aihub",
                "class_index": 0,
                "crop_path": "b0.png",
            },
            {
                "sample_id": "b1",
                "image_id": "b",
                "group_id": "g",
                "code": "K-000002",
                "name": "나정",
                "split": "train",
                "source": "aihub",
                "class_index": 1,
                "crop_path": "b1.png",
            },
        ]
    )


def test_group_embedding_comparison_uses_other_images() -> None:
    samples = _embedding_samples()
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]],
        dtype=np.float32,
    )

    own, other, other_class = compare_group_embeddings(samples, embeddings)

    assert np.all(own > other)
    assert other_class.tolist() == [1, 0, 1, 0]


def test_class_scores_detect_reciprocal_label_swap() -> None:
    samples = _embedding_samples()
    embeddings = np.array(
        [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    model_probabilities = np.array(
        [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]],
        dtype=np.float32,
    )
    oof_probabilities = np.array(
        [[0.1, 0.9], [0.9, 0.1], [0.9, 0.1], [0.1, 0.9]],
        dtype=np.float32,
    )

    scores = build_class_scores(
        samples,
        embeddings,
        model_probabilities,
        oof_probabilities,
    )

    assert scores.loc[0, "suspect_class"] == 1
    assert scores.loc[1, "suspect_class"] == 0
    assert scores.loc[0, "swap"]
    assert scores.loc[1, "swap"]
    assert scores.loc[0, "priority"] == 1
    assert scores.loc[1, "priority"] == 1
