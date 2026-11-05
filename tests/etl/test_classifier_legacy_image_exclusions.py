"""확정 이미지 제외 목록의 분류 ETL 적용 테스트."""

from pathlib import Path

import pandas as pd

from etl.classifier.legacy_image_exclusions import (
    exclude_classifier_samples,
    load_excluded_image_ids,
    normalize_source_image_id,
)


def test_normalize_source_image_id_removes_source_metadata() -> None:
    expected = "K-000001-000002_0_2_0_2_70_000_200"

    assert normalize_source_image_id(f"source_3_{expected}.png") == expected
    assert normalize_source_image_id(f"images/codeit_{expected}.png") == expected
    assert normalize_source_image_id(expected) == expected


def test_load_excluded_image_ids_deduplicates_annotation_rows(
    tmp_path: Path,
) -> None:
    exclusions_path = tmp_path / "excluded.csv"
    pd.DataFrame(
        {
            "이미지 파일명": [
                "source_1_image-a.png",
                "source_1_image-a.png",
                "codeit_image-b.png",
            ]
        }
    ).to_csv(exclusions_path, index=False)

    assert load_excluded_image_ids(exclusions_path) == {"image-a", "image-b"}


def test_exclude_classifier_samples_removes_entire_images() -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["a-0", "a-1", "b-0"],
            "image_id": ["image-a", "image-a", "image-b"],
        }
    )

    remaining, excluded = exclude_classifier_samples(samples, {"image-a"})

    assert remaining["sample_id"].tolist() == ["b-0"]
    assert excluded["sample_id"].tolist() == ["a-0", "a-1"]
