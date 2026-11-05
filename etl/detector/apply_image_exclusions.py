"""확정된 detector 이미지 제외 목록을 기존 학습 패키지에 적용한다."""

import argparse
import csv
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePath

import pandas as pd

from etl.detector.training_package import write_training_package_metadata
from utils.io import write_json

DEFAULT_EXCLUSIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "excluded_data"
    / "final"
    / "detector_excluded_images.csv"
)
SOURCE_PREFIX_PATTERN = re.compile(r"^(?:source_\d+_|codeit_)")


@dataclass(frozen=True, slots=True)
class DetectorExclusionResult:
    """Detector 제외 적용 결과."""

    excluded_images: int
    excluded_annotations: int
    train_images: int
    validation_images: int


def normalize_detector_image_id(value: object) -> str:
    """경로·확장자·출처 접두사를 제거한 이미지 ID를 반환한다."""

    image_id = PurePath(str(value).replace("\\", "/")).name
    return SOURCE_PREFIX_PATTERN.sub("", Path(image_id).stem)


def load_detector_exclusions(path: Path = DEFAULT_EXCLUSIONS_PATH) -> pd.DataFrame:
    """확정 detector 제외표를 읽고 필수 필드와 중복을 검증한다."""

    exclusions = pd.read_csv(path, dtype=str, keep_default_na=False)
    required_columns = {"file_name", "split", "annotation_count", "applied_action"}
    missing_columns = required_columns - set(exclusions.columns)
    if missing_columns:
        raise ValueError(f"detector 제외표 필수 열 누락: {sorted(missing_columns)}")
    normalized_ids = exclusions["file_name"].map(normalize_detector_image_id)
    if normalized_ids.duplicated().any():
        raise ValueError("detector 제외표에 중복 이미지가 있습니다.")
    if set(exclusions["applied_action"]) != {"이미지 및 bbox 전체 제외"}:
        raise ValueError("detector 제외표에 확정되지 않은 조치가 있습니다.")
    return exclusions.assign(normalized_image_id=normalized_ids)


def archive_excluded_detector_images(
    package_dir: Path,
    archive_dir: Path,
    *,
    exclusions_path: Path = DEFAULT_EXCLUSIONS_PATH,
) -> int:
    """활성 images 폴더의 제외 파일을 복구 가능한 별도 폴더로 옮긴다."""

    package_dir = Path(package_dir).expanduser().resolve()
    archive_dir = Path(archive_dir).expanduser().resolve()
    exclusions = load_detector_exclusions(exclusions_path)
    exclusion_ids = set(exclusions["normalized_image_id"])
    source_by_id = {
        normalize_detector_image_id(path.name): path
        for path in (package_dir / "images").iterdir()
        if path.is_file()
    }
    missing_ids = exclusion_ids - source_by_id.keys()
    if missing_ids:
        raise ValueError(
            "활성 images 폴더에서 제외 파일을 찾을 수 없습니다: "
            f"{sorted(missing_ids)[:3]}"
        )
    destinations = {
        image_id: archive_dir / source_by_id[image_id].name
        for image_id in exclusion_ids
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(f"제외 이미지 보관 경로에 파일이 있습니다: {existing[0]}")

    archive_dir.mkdir(parents=True, exist_ok=True)
    for image_id, destination in destinations.items():
        shutil.move(source_by_id[image_id], destination)
    return len(destinations)


def apply_detector_exclusions(
    package_dir: Path,
    *,
    exclusions_path: Path = DEFAULT_EXCLUSIONS_PATH,
) -> DetectorExclusionResult:
    """annotations와 manifest에서 확정 이미지를 제거하고 summary를 재생성한다."""

    package_dir = Path(package_dir).expanduser().resolve()
    annotations_path = package_dir / "annotations.json"
    manifest_path = package_dir / "split_manifest.csv"
    summary_path = package_dir / "detector_data_summary.json"
    exclusions = load_detector_exclusions(exclusions_path)
    exclusion_ids = set(exclusions["normalized_image_id"])
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)

    excluded_images = [
        image
        for image in annotations["images"]
        if normalize_detector_image_id(image["file_name"]) in exclusion_ids
    ]
    found_ids = {
        normalize_detector_image_id(image["file_name"]) for image in excluded_images
    }
    missing_ids = exclusion_ids - found_ids
    if missing_ids:
        raise ValueError(
            "detector 패키지에서 제외 이미지를 찾을 수 없습니다: "
            f"{sorted(missing_ids)[:3]}"
        )
    excluded_image_ids = {image["id"] for image in excluded_images}
    excluded_annotations = [
        annotation
        for annotation in annotations["annotations"]
        if annotation["image_id"] in excluded_image_ids
    ]
    annotations["images"] = [
        image
        for image in annotations["images"]
        if image["id"] not in excluded_image_ids
    ]
    annotations["annotations"] = [
        annotation
        for annotation in annotations["annotations"]
        if annotation["image_id"] not in excluded_image_ids
    ]
    remaining_manifest = manifest[
        ~manifest["image_id"].astype(int).isin(excluded_image_ids)
    ].copy()

    original_annotations = annotations_path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    original_summary = summary_path.read_bytes()
    try:
        with tempfile.TemporaryDirectory(
            prefix="detector-exclusions-",
            dir=package_dir.parent,
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            temporary_annotations = temporary_root / "annotations.json"
            temporary_manifest = temporary_root / "split_manifest.csv"
            write_json(temporary_annotations, annotations)
            remaining_manifest.to_csv(
                temporary_manifest,
                index=False,
                encoding="utf-8-sig",
                quoting=csv.QUOTE_MINIMAL,
            )
            temporary_annotations.replace(annotations_path)
            temporary_manifest.replace(manifest_path)
        write_training_package_metadata(package_dir, overwrite=True)
    except Exception:
        annotations_path.write_bytes(original_annotations)
        manifest_path.write_bytes(original_manifest)
        summary_path.write_bytes(original_summary)
        raise

    return DetectorExclusionResult(
        excluded_images=len(excluded_images),
        excluded_annotations=len(excluded_annotations),
        train_images=int((remaining_manifest["split"] == "train").sum()),
        validation_images=int(
            (remaining_manifest["split"] == "validation").sum()
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS_PATH)
    parser.add_argument("--archive-images-dir", type=Path)
    args = parser.parse_args(argv)
    result = apply_detector_exclusions(
        args.package_dir,
        exclusions_path=args.exclusions,
    )
    if args.archive_images_dir is not None:
        archived_count = archive_excluded_detector_images(
            args.package_dir,
            args.archive_images_dir,
            exclusions_path=args.exclusions,
        )
        print(f"제외 이미지 이동: {archived_count}")
    print(
        f"제외 완료: 이미지 {result.excluded_images}, "
        f"annotation {result.excluded_annotations}, "
        f"train {result.train_images}, validation {result.validation_images}"
    )


if __name__ == "__main__":
    main()
