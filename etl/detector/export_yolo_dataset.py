"""검증된 탐지 학습 패키지를 Ultralytics YOLO 형식으로 내보낸다."""

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from data_loading.detector_package import load_detector_package
from utils.io import read_json, write_json

YOLO_DATASET_FILE = "yolo_dataset.yaml"
YOLO_TRAIN_LIST_FILE = "yolo_train.txt"
YOLO_VALIDATION_LIST_FILE = "yolo_validation.txt"
YOLO_EXPORT_SUMMARY_FILE = "yolo_export_summary.json"
YOLO_LABEL_ROOT = "labels"
YOLO_EXPORT_FORMAT = "ultralytics-yolo-detect-v1"


@dataclass(frozen=True, slots=True)
class YoloDetectorDatasetExport:
    """생성된 YOLO 데이터 view의 주요 경로."""

    dataset_yaml_path: Path
    train_list_path: Path
    validation_list_path: Path
    label_dir: Path
    summary_path: Path


def export_yolo_detector_dataset(
    data_dir: str | Path,
    *,
    overwrite: bool = False,
) -> YoloDetectorDatasetExport:
    """기존 이미지와 split을 보존한 채 YOLO label view를 생성한다."""

    package_dir = Path(data_dir).expanduser().resolve()
    records_by_split, label_map, data_summary = load_detector_package(package_dir)
    _validate_single_pill_label_map(label_map)

    paths = _build_export_paths(package_dir)
    _prepare_export_paths(paths, overwrite=overwrite)

    paths.label_dir.mkdir(parents=True)
    image_paths_by_split: dict[str, list[Path]] = {
        "train": [],
        "validation": [],
    }
    label_paths: set[Path] = set()
    for split in ("train", "validation"):
        for record in records_by_split[split]:
            label_path = package_dir / _yolo_label_relative_path(record)
            if label_path in label_paths:
                raise ValueError(f"YOLO label 경로가 중복됩니다: {label_path}")
            label_paths.add(label_path)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(
                _format_yolo_labels(record),
                encoding="utf-8",
            )
            image_paths_by_split[split].append(Path(record["image_path"]))

    _write_image_list(paths.train_list_path, image_paths_by_split["train"])
    _write_image_list(
        paths.validation_list_path,
        image_paths_by_split["validation"],
    )
    dataset_document = {
        "path": str(package_dir),
        "train": paths.train_list_path.name,
        "val": paths.validation_list_path.name,
        "names": {0: "pill"},
    }
    paths.dataset_yaml_path.write_text(
        yaml.safe_dump(
            dataset_document,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    export_summary = {
        "format": YOLO_EXPORT_FORMAT,
        "source_split_hash": data_summary["split_hash"],
        "dataset_yaml": paths.dataset_yaml_path.name,
        "train_list": paths.train_list_path.name,
        "validation_list": paths.validation_list_path.name,
        "label_root": paths.label_dir.name,
        "num_train_images": len(image_paths_by_split["train"]),
        "num_validation_images": len(image_paths_by_split["validation"]),
        "num_train_instances": data_summary["num_train_instances"],
        "num_validation_instances": data_summary["num_val_instances"],
    }
    write_json(paths.summary_path, export_summary, sort_keys=True)

    _validate_exported_yolo_dataset(
        paths,
        records_by_split=records_by_split,
        expected_dataset_document=dataset_document,
    )
    return paths


def _build_export_paths(package_dir: Path) -> YoloDetectorDatasetExport:
    return YoloDetectorDatasetExport(
        dataset_yaml_path=package_dir / YOLO_DATASET_FILE,
        train_list_path=package_dir / YOLO_TRAIN_LIST_FILE,
        validation_list_path=package_dir / YOLO_VALIDATION_LIST_FILE,
        label_dir=package_dir / YOLO_LABEL_ROOT,
        summary_path=package_dir / YOLO_EXPORT_SUMMARY_FILE,
    )


def _prepare_export_paths(
    paths: YoloDetectorDatasetExport,
    *,
    overwrite: bool,
) -> None:
    owned_files = (
        paths.dataset_yaml_path,
        paths.train_list_path,
        paths.validation_list_path,
        paths.summary_path,
    )
    existing = [path for path in (*owned_files, paths.label_dir) if path.exists()]
    if not existing:
        return
    if not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"YOLO export가 이미 존재합니다: {names}")
    if not paths.summary_path.is_file():
        raise ValueError(
            "기존 labels 또는 YOLO 파일의 소유권을 확인할 summary가 없습니다."
        )
    summary = read_json(paths.summary_path)
    if (
        not isinstance(summary, dict)
        or summary.get("format") != YOLO_EXPORT_FORMAT
        or summary.get("label_root") != paths.label_dir.name
    ):
        raise ValueError("기존 YOLO export summary 형식이 올바르지 않습니다.")

    if paths.label_dir.exists():
        if not paths.label_dir.is_dir():
            raise ValueError("YOLO label 경로가 폴더가 아닙니다.")
        shutil.rmtree(paths.label_dir)
    for path in owned_files:
        if path.exists():
            if not path.is_file():
                raise ValueError(f"YOLO export 파일 경로가 파일이 아닙니다: {path}")
            path.unlink()


def _validate_single_pill_label_map(label_map: dict[str, Any]) -> None:
    expected_classes = [{"label": 1, "name": "pill"}]
    if (
        label_map.get("num_foreground_classes") != 1
        or label_map.get("classes") != expected_classes
    ):
        raise ValueError("YOLO export는 단일 pill foreground label만 지원합니다.")


def _yolo_label_relative_path(record: dict[str, Any]) -> Path:
    relative_path = PurePosixPath(record["relative_path"])
    if len(relative_path.parts) < 2 or relative_path.parts[0] != "images":
        raise ValueError(
            f"YOLO export 이미지 경로는 images/ 아래에 있어야 합니다: {relative_path}"
        )
    image_relative_path = PurePosixPath(*relative_path.parts[1:])
    if not image_relative_path.suffix:
        raise ValueError(f"이미지 확장자가 없습니다: {relative_path}")
    label_relative_path = image_relative_path.with_suffix(".txt")
    return Path(YOLO_LABEL_ROOT).joinpath(*label_relative_path.parts)


def _format_yolo_labels(record: dict[str, Any]) -> str:
    width = float(record["width"])
    height = float(record["height"])
    lines: list[str] = []
    for x, y, box_width, box_height in record["bbox_xywh"]:
        center_x = (x + box_width / 2.0) / width
        center_y = (y + box_height / 2.0) / height
        normalized_width = box_width / width
        normalized_height = box_height / height
        values = (center_x, center_y, normalized_width, normalized_height)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(
                f"YOLO 정규화 bbox가 범위를 벗어났습니다: {record['image_id']}"
            )
        lines.append("0 " + " ".join(f"{value:.10f}" for value in values))
    return "\n".join(lines) + "\n"


def _write_image_list(path: Path, image_paths: list[Path]) -> None:
    path.write_text(
        "\n".join(str(image_path.resolve()) for image_path in image_paths) + "\n",
        encoding="utf-8",
    )


def _validate_exported_yolo_dataset(
    paths: YoloDetectorDatasetExport,
    *,
    records_by_split: dict[str, list[dict[str, Any]]],
    expected_dataset_document: dict[str, Any],
) -> None:
    dataset_document = yaml.safe_load(
        paths.dataset_yaml_path.read_text(encoding="utf-8")
    )
    if dataset_document != expected_dataset_document:
        raise ValueError("생성된 YOLO dataset YAML이 실행 설정과 다릅니다.")

    for split, list_path in (
        ("train", paths.train_list_path),
        ("validation", paths.validation_list_path),
    ):
        records = records_by_split[split]
        image_lines = list_path.read_text(encoding="utf-8").splitlines()
        expected_image_lines = [record["image_path"] for record in records]
        if image_lines != expected_image_lines:
            raise ValueError(f"YOLO {split} 이미지 목록이 source split과 다릅니다.")
        for record in records:
            label_path = paths.dataset_yaml_path.parent / _yolo_label_relative_path(
                record
            )
            label_lines = label_path.read_text(encoding="utf-8").splitlines()
            if len(label_lines) != len(record["bbox_xywh"]):
                raise ValueError(
                    f"YOLO bbox 개수가 source와 다릅니다: {record['image_id']}"
                )
            for line in label_lines:
                fields = line.split()
                if len(fields) != 5 or fields[0] != "0":
                    raise ValueError(f"YOLO label 행 형식 오류: {label_path}")
                coordinates = [float(value) for value in fields[1:]]
                if any(value < 0.0 or value > 1.0 for value in coordinates):
                    raise ValueError(f"YOLO label 좌표 범위 오류: {label_path}")
                if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
                    raise ValueError(f"YOLO label bbox 크기 오류: {label_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="검증된 detector package를 Ultralytics YOLO 형식으로 내보냅니다."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/detector"),
        help="detector 학습 패키지 경로",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이 exporter가 만든 기존 YOLO view를 교체합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = export_yolo_detector_dataset(
        args.data_dir,
        overwrite=args.overwrite,
    )
    print(f"YOLO dataset YAML: {result.dataset_yaml_path}")


if __name__ == "__main__":
    main()
