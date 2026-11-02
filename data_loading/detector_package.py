"""라벨과 분할 목록을 읽고, 검증된 이미지별 기록을 만든다.

전체 검사와 DataLoader 생성이 이 함수를 함께 사용한다.
이미지 픽셀과 실제 객체 위치는 여기서 검사하지 않는다.
"""

import copy
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from utils.io import read_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = {
    "package_dir": "data/detector/training_package_detector",
    "image_roots": {
        "default": "data/detector/training_package_detector/images",
    },
    "augmentation": {
        "horizontal_flip": 0.0,
        "vertical_flip": 0.0,
        "brightness": 0.0,
        "contrast": 0.0,
    },
}


def repository_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def load_detector_settings(path: str | Path | None = None):
    """기본 설정 또는 명시한 JSON 설정 파일을 읽는다."""

    if path is None:
        settings = copy.deepcopy(DEFAULT_SETTINGS)
        _validate_detector_settings(settings)
        return settings
    path = repository_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"지정한 데이터 설정 파일이 없습니다: {path}")
    settings = read_json(path)
    _validate_detector_settings(settings)
    return settings


def _validate_detector_settings(settings: Any) -> None:
    if not isinstance(settings, Mapping):
        raise TypeError("detector data settings must be a mapping")
    for key in DEFAULT_SETTINGS:
        if key not in settings:
            raise ValueError(f"데이터 설정 항목이 없습니다: {key}")
    package_dir = settings["package_dir"]
    if not isinstance(package_dir, str | Path) or not str(package_dir).strip():
        raise ValueError("package_dir must be a non-empty path")
    image_roots = settings["image_roots"]
    if not isinstance(image_roots, Mapping) or not image_roots:
        raise ValueError("image_roots must be a non-empty mapping")
    for dataset, root in image_roots.items():
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("image_roots dataset names must be non-empty strings")
        if not isinstance(root, str | Path) or not str(root).strip():
            raise ValueError(f"image root must be a non-empty path: {dataset}")
    if not isinstance(settings["augmentation"], Mapping):
        raise TypeError("augmentation must be a mapping")


def resolve_data_paths(settings):
    _validate_detector_settings(settings)
    return repository_path(settings["package_dir"]), {
        name: repository_path(value) for name, value in settings["image_roots"].items()
    }


def resolve_image_path(relative_name, *, dataset, roots):
    """dataset root와 상대 경로를 실제 이미지 경로로 변환한다."""

    if dataset not in roots:
        raise ValueError(f"이미지 출처 설정이 없습니다: {dataset}")
    relative = PurePosixPath(str(relative_name).replace("\\", "/"))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or str(relative) in {"", "."}
    ):
        raise ValueError(
            f"이미지 경로는 데이터 루트 기준 상대 경로여야 합니다: {relative}"
        )
    root = Path(roots[dataset]).resolve()
    result = root.joinpath(*relative.parts).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"이미지 기준 폴더를 벗어난 경로입니다: {relative}")
    return result


def original_file_name(file_name):
    name = Path(file_name).name
    match = re.search(r"(K-\d{6}.*)$", name)

    if match is None:
        raise ValueError(f"원본 파일명을 확인할 수 없습니다: {name}")

    return match.group(1)


def canonical_group(file_name):
    prefix = original_file_name(file_name).split("_", 1)[0]
    codes = prefix.removeprefix("K-").split("-")

    if not codes or not all(re.fullmatch(r"\d{6}", code) for code in codes):
        raise ValueError(f"알약 조합 형식 오류: {file_name}")

    return "K-" + "-".join(sorted(codes))


def read_records(settings):
    """고정 분할을 읽는다. 재분할하거나 잘못된 박스를 보정하지 않는다."""
    package, roots = resolve_data_paths(settings)

    with (package / "split_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        manifest_rows = list(csv.DictReader(file))

    manifest = {}

    for row in manifest_rows:
        image_id = int(row["image_id"])

        if image_id in manifest:
            raise ValueError(f"분할 목록의 이미지 ID 중복: {image_id}")

        manifest[image_id] = row

    all_image_ids = set()
    all_annotation_ids = set()
    records_by_split = {}

    for split in ["train", "validation"]:
        labels = read_json(package / "annotations" / f"{split}.json")
        split_ids = set()
        annotation_groups = defaultdict(list)

        for image in labels["images"]:
            image_id = image["id"]

            if type(image_id) is not int or image_id < 0 or image_id in all_image_ids:
                raise ValueError(f"이미지 ID 오류 또는 중복: {image_id}")

            all_image_ids.add(image_id)
            split_ids.add(image_id)

        for annotation in labels["annotations"]:
            annotation_id = annotation["id"]

            if (
                type(annotation_id) is not int
                or annotation_id < 0
                or annotation_id in all_annotation_ids
            ):
                raise ValueError(f"annotation ID 오류 또는 중복: {annotation_id}")

            if (
                type(annotation["image_id"]) is not int
                or annotation["image_id"] not in split_ids
            ):
                raise ValueError(f"이미지에 연결되지 않은 annotation: {annotation_id}")

            all_annotation_ids.add(annotation_id)
            annotation_groups[annotation["image_id"]].append(annotation)

        records = []

        for image in labels["images"]:
            image_id = image["id"]

            if image_id not in manifest:
                raise ValueError(f"분할 목록에 없는 이미지: {image_id}")

            row = manifest[image_id]

            if row["split"] != split:
                raise ValueError(f"분할 배정 불일치: {image_id}")

            width = image["width"]
            height = image["height"]

            if any(type(v) is not int or v <= 0 for v in (width, height)):
                raise ValueError(f"이미지 크기 오류: {image_id}")

            relative_path = str(image["file_name"]).replace("\\", "/")
            dataset = row["dataset"]
            if relative_path != row["image_path"]:
                raise ValueError(f"이미지 경로 기록 불일치: {image_id}")

            group = canonical_group(relative_path)

            if group != row["group_id"]:
                raise ValueError(f"조합 그룹 기록 불일치: {image_id}")

            actual_path = resolve_image_path(
                relative_path,
                dataset=dataset,
                roots=roots,
            )

            if not actual_path.is_file():
                raise FileNotFoundError(actual_path)

            annotations = annotation_groups[image_id]

            if not annotations:
                raise ValueError(f"박스 없는 이미지: {image_id}")

            if len(annotations) != int(row["annotation_count"]):
                raise ValueError(f"annotation 개수 불일치: {image_id}")

            boxes = []
            source_boxes = []
            category_ids = []
            drug_codes = []

            for annotation in annotations:
                annotation_id = annotation["id"]
                raw_box = annotation["bbox"]
                category_id = annotation["category_id"]
                drug_code = annotation.get("drug_code")

                if type(category_id) is not int or category_id < 0:
                    raise ValueError(f"category_id 오류: {annotation_id}")

                if (
                    not isinstance(drug_code, str)
                    or re.fullmatch(r"K-\d{6}", drug_code) is None
                ):
                    raise ValueError(f"drug_code 오류: {annotation_id}")

                if len(raw_box) != 4:
                    raise ValueError(f"bbox 원소 수 오류: {annotation_id}")

                if any(
                    isinstance(v, bool) or not isinstance(v, (int, float))
                    for v in raw_box
                ):
                    raise ValueError(f"bbox 자료형 오류: {annotation_id}")
                x, y, w, h = map(float, raw_box)

                if not all(math.isfinite(v) for v in [x, y, w, h]):
                    raise ValueError(f"bbox 숫자 오류: {annotation_id}")

                if w <= 0 or h <= 0:
                    raise ValueError(f"bbox 크기 오류: {annotation_id}")

                if annotation.get("iscrowd", 0) != 0:
                    raise ValueError(f"iscrowd가 0이 아닙니다: {annotation_id}")

                if annotation.get("ignore", 0) != 0:
                    raise ValueError(f"ignore가 0이 아닙니다: {annotation_id}")

                if x < 0 or y < 0 or x + w > width or y + h > height:
                    raise ValueError(f"bbox 이미지 경계 오류: {annotation_id}")
                area = annotation.get("area")
                if (
                    isinstance(area, bool)
                    or not isinstance(area, (int, float))
                    or not math.isfinite(area)
                    or not math.isclose(area, w * h, rel_tol=1e-6, abs_tol=1e-4)
                ):
                    raise ValueError(f"bbox 면적 불일치: {annotation_id}")
                box = [x, y, x + w, y + h]

                boxes.append(box)
                source_boxes.append([x, y, w, h])
                category_ids.append(category_id)
                drug_codes.append(drug_code)

            records.append(
                {
                    "image_id": image_id,
                    "source_image_id": str(row["original_image_id"]),
                    "original_file_name": original_file_name(relative_path),
                    "relative_path": relative_path,
                    "image_path": str(actual_path.resolve()),
                    "dataset": dataset,
                    "group_id": group,
                    "width": width,
                    "height": height,
                    "boxes": boxes,
                    "bbox_xywh": source_boxes,
                    "annotation_ids": [a["id"] for a in annotations],
                    "category_ids": category_ids,
                    "drug_codes": drug_codes,
                }
            )

        records_by_split[split] = sorted(records, key=lambda record: record["image_id"])

    if set(manifest) != all_image_ids:
        raise ValueError("분할 목록과 라벨의 전체 이미지 구성이 다릅니다.")

    groups = {
        split: {record["group_id"] for record in records}
        for split, records in records_by_split.items()
    }

    overlap = groups["train"] & groups["validation"]

    if overlap:
        raise ValueError(f"Train과 Validation의 조합 그룹 중복: {len(overlap)}개")

    label_map = {
        "num_foreground_classes": 1,
        "num_model_classes": 2,
        "background_label": 0,
        "classes": [{"label": 1, "name": "pill"}],
    }

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
        for split in ["train", "validation"]
        for record in records_by_split[split]
    ]

    split_hash = hashlib.sha256(
        json.dumps(
            {"version": "fixed-pill-data-v1", "records": hash_records},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    summary = {
        "group_overlap_count": 0,
        "split_hash": split_hash,
        "clipped_box_count": 0,
        "clipped_annotation_ids": [],
    }
    for split, name in (("train", "train"), ("validation", "val")):
        counts = [len(record["boxes"]) for record in records_by_split[split]]
        if not counts:
            raise ValueError(f"이미지가 없는 분할: {split}")
        count = sum(counts)
        summary.update(
            {
                f"num_{name}_images": len(counts),
                f"num_{name}_source_annotations": count,
                f"num_{name}_instances": count,
                f"{name}_group_count": len(groups[split]),
                f"{name}_empty_image_count": 0,
                f"{name}_max_instances_per_image": max(counts),
                f"{name}_label_counts": {"1": count},
            }
        )
    return records_by_split, label_map, summary
