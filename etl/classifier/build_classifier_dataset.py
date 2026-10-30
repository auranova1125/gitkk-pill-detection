"""분류 학습 패키지를 생성하는 ETL 단계를 순서대로 실행한다."""

import argparse
from pathlib import Path

from etl.classifier.assign_data_splits import assign_data_splits
from etl.classifier.create_pill_crops import create_pill_crops
from etl.classifier.export_training_package import export_training_package
from etl.classifier.paths import DETECTOR_PACKAGE
from etl.classifier.select_source_images import select_source_images
from etl.classifier.summarize_crops import summarize_crops
from etl.classifier.validate_source_labels import validate_source_labels


def build_classifier_dataset(
    *,
    detector_package: Path = DETECTOR_PACKAGE,
    target_samples_per_class: int = 108,
    skip_png_validation: bool = False,
    overwrite: bool = False,
) -> None:
    """원천 라벨 검증부터 학습 패키지 export까지 전체 ETL을 실행한다."""

    if target_samples_per_class <= 0:
        raise ValueError("target_samples_per_class must be greater than zero")

    validate_source_labels(skip_png=skip_png_validation)
    select_source_images(target=target_samples_per_class)
    create_pill_crops(overwrite=overwrite)
    assign_data_splits(detector_package)
    summarize_crops(minimum=target_samples_per_class)
    export_training_package(overwrite=overwrite)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="분류 학습 패키지 ETL 전체 단계를 실행합니다."
    )
    parser.add_argument(
        "--detector-package",
        type=Path,
        default=DETECTOR_PACKAGE,
    )
    parser.add_argument("--target", type=int, default=108)
    parser.add_argument("--skip-png-validation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    build_classifier_dataset(
        detector_package=args.detector_package,
        target_samples_per_class=args.target,
        skip_png_validation=args.skip_png_validation,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
