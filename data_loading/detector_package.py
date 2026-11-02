"""고정 분할된 탐지 학습 패키지를 읽고 전체 구성을 검증한다.

이미지 픽셀은 Dataset에서 읽는다. 이 모듈은 annotation, manifest,
이미지 경로, bbox와 분할 무결성만 확인한다.
"""

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from data_loading.detector_data_paths import (
    resolve_detector_data_paths,
    resolve_detector_image_path,
)
from utils.io import read_json

DetectorRecord = dict[str, Any]
DetectorRecordsBySplit = dict[str, list[DetectorRecord]]


def load_detector_package(
    settings: Mapping[str, Any],
) -> tuple[DetectorRecordsBySplit, dict[str, Any], dict[str, Any]]:
    """탐지 패키지를 읽고 라벨·경로·고정 분할의 일관성을 확인한다."""

    package_dir, image_roots = resolve_detector_data_paths(settings)
    manifest = read_detector_manifest(package_dir / "split_manifest.csv")
    all_image_ids: set[int] = set()
    all_annotation_ids: set[int] = set()
    records_by_split: DetectorRecordsBySplit = {}

    for split in ("train", "validation"):
        records_by_split[split] = _build_detector_split_records(
            split=split,
            labels=read_json(package_dir / "annotations" / f"{split}.json"),
            manifest=manifest,
            image_roots=image_roots,
            all_image_ids=all_image_ids,
            all_annotation_ids=all_annotation_ids,
        )

    if set(manifest) != all_image_ids:
        raise ValueError("분할 목록과 라벨의 전체 이미지 구성이 다릅니다.")
    _validate_detector_group_split(records_by_split)

    label_map = {
        "num_foreground_classes": 1,
        "num_model_classes": 2,
        "background_label": 0,
        "classes": [{"label": 1, "name": "pill"}],
    }
    return (
        records_by_split,
        label_map,
        summarize_detector_records(records_by_split),
    )


def read_detector_manifest(path: Path) -> dict[int, dict[str, str]]:
    """분할 manifest를 이미지 ID로 조회 가능한 mapping으로 읽는다."""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    manifest: dict[int, dict[str, str]] = {}
    for row in rows:
        image_id = int(row["image_id"])
        if image_id in manifest:
            raise ValueError(f"분할 목록의 이미지 ID 중복: {image_id}")
        manifest[image_id] = row
    return manifest


def calculate_detector_split_hash(records_by_split: DetectorRecordsBySplit) -> str:
    """검증된 split과 annotation 구성을 재현 가능한 해시로 만든다."""

    hash_records = [
        {
            "split": split,
            "image_id": record["image_id"],
            "group_id": record["group_id"],
            "file_name": record["relative_path"],
            "annotation_ids": record["annotation_ids"],
            "bbox_xywh": record["bbox_xywh"],
            "category_ids": record["category_ids"],
        }
        for split in ("train", "validation")
        for record in records_by_split[split]
    ]
    return hashlib.sha256(
        json.dumps(
            {"version": "fixed-pill-data-v1", "records": hash_records},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def summarize_detector_records(
    records_by_split: DetectorRecordsBySplit,
) -> dict[str, Any]:
    """검증된 이미지별 기록에서 탐지 ``data_summary``를 계산한다."""

    summary: dict[str, Any] = {
        "group_overlap_count": 0,
        "split_hash": calculate_detector_split_hash(records_by_split),
        "clipped_box_count": 0,
        "clipped_annotation_ids": [],
    }
    for split, summary_name in (("train", "train"), ("validation", "val")):
        records = records_by_split[split]
        instance_counts = [len(record["boxes"]) for record in records]
        if not instance_counts:
            raise ValueError(f"이미지가 없는 분할: {split}")
        instance_count = sum(instance_counts)
        summary.update(
            {
                f"num_{summary_name}_images": len(records),
                f"num_{summary_name}_source_annotations": instance_count,
                f"num_{summary_name}_instances": instance_count,
                f"{summary_name}_group_count": len(
                    {record["group_id"] for record in records}
                ),
                f"{summary_name}_empty_image_count": 0,
                f"{summary_name}_max_instances_per_image": max(instance_counts),
                f"{summary_name}_label_counts": {"1": instance_count},
            }
        )
    return summary


def _build_detector_split_records(
    *,
    split: str,
    labels: Mapping[str, Any],
    manifest: Mapping[int, Mapping[str, str]],
    image_roots: Mapping[str, Path],
    all_image_ids: set[int],
    all_annotation_ids: set[int],
) -> list[DetectorRecord]:
    images = labels["images"]
    annotations = labels["annotations"]
    split_image_ids: set[int] = set()
    annotations_by_image: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)

    for image in images:
        image_id = image["id"]
        if type(image_id) is not int or image_id < 0 or image_id in all_image_ids:
            raise ValueError(f"이미지 ID 오류 또는 중복: {image_id}")
        all_image_ids.add(image_id)
        split_image_ids.add(image_id)

    for annotation in annotations:
        annotation_id = annotation["id"]
        if (
            type(annotation_id) is not int
            or annotation_id < 0
            or annotation_id in all_annotation_ids
        ):
            raise ValueError(f"annotation ID 오류 또는 중복: {annotation_id}")
        image_id = annotation["image_id"]
        if type(image_id) is not int or image_id not in split_image_ids:
            raise ValueError(f"이미지에 연결되지 않은 annotation: {annotation_id}")
        all_annotation_ids.add(annotation_id)
        annotations_by_image[image_id].append(annotation)

    records = [
        _build_detector_image_record(
            split=split,
            image=image,
            annotations=annotations_by_image[image["id"]],
            manifest=manifest,
            image_roots=image_roots,
        )
        for image in images
    ]
    return sorted(records, key=lambda record: record["image_id"])


def _build_detector_image_record(
    *,
    split: str,
    image: Mapping[str, Any],
    annotations: list[Mapping[str, Any]],
    manifest: Mapping[int, Mapping[str, str]],
    image_roots: Mapping[str, Path],
) -> DetectorRecord:
    image_id = image["id"]
    if image_id not in manifest:
        raise ValueError(f"분할 목록에 없는 이미지: {image_id}")
    manifest_row = manifest[image_id]
    if manifest_row["split"] != split:
        raise ValueError(f"분할 배정 불일치: {image_id}")

    width = image["width"]
    height = image["height"]
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise ValueError(f"이미지 크기 오류: {image_id}")

    relative_path = str(image["file_name"]).replace("\\", "/")
    dataset = manifest_row["dataset"]
    if relative_path != manifest_row["image_path"]:
        raise ValueError(f"이미지 경로 기록 불일치: {image_id}")
    group_id = manifest_row["group_id"]
    if not isinstance(group_id, str) or not group_id.strip():
        raise ValueError(f"비어 있는 조합 그룹: {image_id}")

    image_path = resolve_detector_image_path(
        relative_path,
        dataset=dataset,
        roots=image_roots,
    )
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not annotations:
        raise ValueError(f"박스 없는 이미지: {image_id}")
    if len(annotations) != int(manifest_row["annotation_count"]):
        raise ValueError(f"annotation 개수 불일치: {image_id}")

    boxes: list[list[float]] = []
    source_boxes: list[list[float]] = []
    category_ids: list[int] = []
    drug_codes: list[str] = []
    for annotation in annotations:
        box, source_box, category_id, drug_code = _validate_detector_annotation(
            annotation,
            image_width=width,
            image_height=height,
        )
        boxes.append(box)
        source_boxes.append(source_box)
        category_ids.append(category_id)
        drug_codes.append(drug_code)

    return {
        "image_id": image_id,
        "source_image_id": str(manifest_row["original_image_id"]),
        "original_file_name": PurePosixPath(relative_path).name,
        "relative_path": relative_path,
        "image_path": str(image_path),
        "dataset": dataset,
        "group_id": group_id,
        "width": width,
        "height": height,
        "boxes": boxes,
        "bbox_xywh": source_boxes,
        "annotation_ids": [annotation["id"] for annotation in annotations],
        "category_ids": category_ids,
        "drug_codes": drug_codes,
    }


def _validate_detector_annotation(
    annotation: Mapping[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> tuple[list[float], list[float], int, str]:
    annotation_id = annotation["id"]
    category_id = annotation["category_id"]
    drug_code = annotation.get("drug_code")
    if type(category_id) is not int or category_id < 0:
        raise ValueError(f"category_id 오류: {annotation_id}")
    if not isinstance(drug_code, str) or re.fullmatch(r"K-\d{6}", drug_code) is None:
        raise ValueError(f"drug_code 오류: {annotation_id}")

    raw_box = annotation["bbox"]
    if not isinstance(raw_box, list) or len(raw_box) != 4:
        raise ValueError(f"bbox 원소 수 오류: {annotation_id}")
    if any(
        isinstance(value, bool) or not isinstance(value, int | float)
        for value in raw_box
    ):
        raise ValueError(f"bbox 자료형 오류: {annotation_id}")
    x, y, width, height = map(float, raw_box)
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError(f"bbox 숫자 오류: {annotation_id}")
    if width <= 0 or height <= 0:
        raise ValueError(f"bbox 크기 오류: {annotation_id}")
    if annotation.get("iscrowd", 0) != 0:
        raise ValueError(f"iscrowd가 0이 아닙니다: {annotation_id}")
    if annotation.get("ignore", 0) != 0:
        raise ValueError(f"ignore가 0이 아닙니다: {annotation_id}")
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError(f"bbox 이미지 경계 오류: {annotation_id}")

    area = annotation.get("area")
    if (
        isinstance(area, bool)
        or not isinstance(area, int | float)
        or not math.isfinite(area)
        or not math.isclose(area, width * height, rel_tol=1e-6, abs_tol=1e-4)
    ):
        raise ValueError(f"bbox 면적 불일치: {annotation_id}")
    return (
        [x, y, x + width, y + height],
        [x, y, width, height],
        category_id,
        drug_code,
    )


def _validate_detector_group_split(
    records_by_split: DetectorRecordsBySplit,
) -> None:
    train_groups = {record["group_id"] for record in records_by_split["train"]}
    validation_groups = {
        record["group_id"] for record in records_by_split["validation"]
    }
    overlap = train_groups & validation_groups
    if overlap:
        raise ValueError(f"Train과 Validation의 조합 그룹 중복: {len(overlap)}개")
