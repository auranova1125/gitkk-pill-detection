"""이전 분류 패키지의 1차 이미지 제외 목록을 적용한다."""

import argparse
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePath

import pandas as pd

from data_loading.classifier_package import (
    read_classifier_samples,
    summarize_classifier_samples,
)
from utils.io import read_json, write_json

DEFAULT_EXCLUSIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "excluded_data"
    / "archive"
    / "legacy_excluded_image_details.csv"
)
SOURCE_PREFIX_PATTERN = re.compile(r"^(?:source_\d+_|codeit_)")
EXCLUDED_FILE_NAME_COLUMN = "이미지 파일명"


@dataclass(frozen=True, slots=True)
class ImageExclusionResult:
    """이미지 제외 적용 전후의 분류 샘플 수."""

    excluded_image_count: int
    excluded_sample_count: int
    train_sample_count: int
    validation_sample_count: int


def normalize_source_image_id(value: object) -> str:
    """경로·확장자·출처 접두사를 제거해 공통 원본 이미지 ID를 만든다."""

    image_id = PurePath(str(value).replace("\\", "/")).name
    image_id = Path(image_id).stem
    return SOURCE_PREFIX_PATTERN.sub("", image_id)


def load_excluded_image_ids(path: Path = DEFAULT_EXCLUSIONS_PATH) -> set[str]:
    """확정 제외 상세표에서 중복 없는 원본 이미지 ID를 읽는다."""

    exclusions = pd.read_csv(path, dtype=str, keep_default_na=False)
    if EXCLUDED_FILE_NAME_COLUMN not in exclusions:
        raise ValueError(f"제외 목록에 필수 열이 없습니다: {EXCLUDED_FILE_NAME_COLUMN}")
    image_ids = {
        normalize_source_image_id(file_name)
        for file_name in exclusions[EXCLUDED_FILE_NAME_COLUMN]
        if str(file_name).strip()
    }
    if not image_ids:
        raise ValueError("제외 목록에 이미지가 없습니다.")
    return image_ids


def exclude_classifier_samples(
    samples: pd.DataFrame,
    excluded_image_ids: set[str],
    *,
    image_id_column: str = "image_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """제외 이미지에 연결된 모든 crop 행을 제거한다."""

    if image_id_column not in samples:
        raise ValueError(f"samples에 필수 열이 없습니다: {image_id_column}")
    normalized_ids = samples[image_id_column].map(normalize_source_image_id)
    excluded_mask = normalized_ids.isin(excluded_image_ids)
    excluded_samples = samples.loc[excluded_mask].copy()
    remaining_samples = samples.loc[~excluded_mask].copy().reset_index(drop=True)
    return remaining_samples, excluded_samples.reset_index(drop=True)


def apply_exclusions_to_classifier_package(
    package_dir: Path,
    *,
    exclusions_path: Path = DEFAULT_EXCLUSIONS_PATH,
) -> ImageExclusionResult:
    """기존 분류 패키지의 samples와 summary를 제외 목록 기준으로 갱신한다."""

    package_dir = Path(package_dir).expanduser().resolve()
    samples_path = package_dir / "samples.csv"
    summary_path = package_dir / "data_summary.json"
    class_map = read_json(package_dir / "class_map.json")
    samples = pd.read_csv(samples_path, dtype=str, keep_default_na=False)
    excluded_image_ids = load_excluded_image_ids(exclusions_path)
    remaining, excluded = exclude_classifier_samples(samples, excluded_image_ids)
    if excluded.empty:
        raise ValueError("현재 분류 패키지에 제외 대상 이미지가 없습니다.")
    if remaining["class_index"].nunique() != class_map["num_classes"]:
        raise ValueError("이미지 제외 후 샘플이 없는 클래스가 생깁니다.")

    with tempfile.TemporaryDirectory(
        prefix="classifier-exclusions-",
        dir=package_dir.parent,
    ) as temporary_directory:
        temporary_package = Path(temporary_directory)
        temporary_samples_path = temporary_package / "samples.csv"
        temporary_summary_path = temporary_package / "data_summary.json"
        remaining.to_csv(temporary_samples_path, index=False)
        runtime_samples = read_classifier_samples(temporary_package)
        summary = summarize_classifier_samples(
            runtime_samples,
            class_map["num_classes"],
        )
        write_json(temporary_summary_path, summary)
        temporary_samples_path.replace(samples_path)
        temporary_summary_path.replace(summary_path)

    return ImageExclusionResult(
        excluded_image_count=excluded["image_id"]
        .map(normalize_source_image_id)
        .nunique(),
        excluded_sample_count=len(excluded),
        train_sample_count=int((remaining["split"] == "train").sum()),
        validation_sample_count=int((remaining["split"] == "val").sum()),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_dir", type=Path)
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=DEFAULT_EXCLUSIONS_PATH,
    )
    args = parser.parse_args(argv)
    result = apply_exclusions_to_classifier_package(
        args.package_dir,
        exclusions_path=args.exclusions,
    )
    print(
        f"제외 완료: 이미지 {result.excluded_image_count}, "
        f"crop {result.excluded_sample_count}, "
        f"train {result.train_sample_count}, "
        f"validation {result.validation_sample_count}"
    )


if __name__ == "__main__":
    main()
