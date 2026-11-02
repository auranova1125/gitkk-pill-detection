"""Codeit 이미지 단위 annotation을 검증된 공통 표본으로 읽는다."""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from data_loading.detector_package import canonical_group
from utils.io import read_json


@dataclass(frozen=True, slots=True)
class DetectorSourceAnnotation:
    """이미지 하나와 그 이미지의 모든 알약 annotation."""

    dataset: str
    annotation_path: Path
    image_path: Path
    group_id: str
    document: Mapping[str, Any]


def load_codeit_annotations(source_dir: Path) -> tuple[DetectorSourceAnnotation, ...]:
    """Archive의 `codeit_train_validated` 형식을 읽고 전체 표본을 검증한다."""

    source_dir = Path(source_dir).expanduser().resolve()
    annotations_dir = source_dir / "annotations"
    images_dir = source_dir / "images"
    if not annotations_dir.is_dir() or not images_dir.is_dir():
        raise FileNotFoundError(
            "codeit source must contain annotations/ and images/ directories"
        )

    annotation_paths = sorted(
        path
        for path in annotations_dir.glob("*.json")
        if not path.name.startswith("._")
    )
    if not annotation_paths:
        raise ValueError("codeit source contains no annotation JSON files")

    samples: list[DetectorSourceAnnotation] = []
    image_ids: set[int] = set()
    annotation_ids: set[int] = set()
    for annotation_path in annotation_paths:
        sample = _validate_codeit_annotation(
            read_json(annotation_path),
            annotation_path=annotation_path,
            images_dir=images_dir,
        )
        image_id = sample.document["images"][0]["id"]
        if image_id in image_ids:
            raise ValueError(f"duplicate source image ID: {image_id}")
        image_ids.add(image_id)
        for annotation in sample.document["annotations"]:
            annotation_id = annotation["id"]
            if annotation_id in annotation_ids:
                raise ValueError(f"duplicate source annotation ID: {annotation_id}")
            annotation_ids.add(annotation_id)
        samples.append(sample)
    return tuple(samples)


def _validate_codeit_annotation(
    value: Any,
    *,
    annotation_path: Path,
    images_dir: Path,
) -> DetectorSourceAnnotation:
    """Codeit 이미지 하나와 완전한 bbox 목록을 포함한 JSON을 검증한다."""

    if not isinstance(value, Mapping):
        raise TypeError("source annotation root must be a mapping")
    images = value.get("images")
    annotations = value.get("annotations")
    categories = value.get("categories")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("source annotation must contain exactly one image")
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("source annotation must contain at least one bbox")
    if not isinstance(categories, list) or not categories:
        raise ValueError("source annotation must contain categories")

    image = images[0]
    if not isinstance(image, Mapping):
        raise TypeError("source image entry must be a mapping")
    image_id = _require_non_negative_integer(image.get("id"), name="image.id")
    width = _require_positive_integer(image.get("width"), name="image.width")
    height = _require_positive_integer(image.get("height"), name="image.height")
    file_name = image.get("file_name")
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("image.file_name must be a non-empty string")
    if Path(file_name).name != file_name:
        raise ValueError("source image.file_name must be a plain file name")

    image_path = (Path(images_dir) / file_name).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as source_image:
        if source_image.size != (width, height):
            raise ValueError(f"source image size mismatch: {file_name}")

    category_ids: set[int] = set()
    for category in categories:
        if not isinstance(category, Mapping):
            raise TypeError("source category entry must be a mapping")
        category_id = _require_non_negative_integer(
            category.get("id"),
            name="category.id",
        )
        if category_id in category_ids:
            raise ValueError(f"duplicate source category ID: {category_id}")
        if category.get("name") != f"K-{category_id:06d}":
            raise ValueError(f"source category name does not match ID: {category_id}")
        category_ids.add(category_id)
    if len(category_ids) != len(categories):
        raise ValueError("source categories must be unique mappings")

    local_annotation_ids: set[int] = set()
    drug_codes: list[str] = []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            raise TypeError(f"annotations[{index}] must be a mapping")
        annotation_id = _require_non_negative_integer(
            annotation.get("id"),
            name=f"annotations[{index}].id",
        )
        if annotation_id in local_annotation_ids:
            raise ValueError(f"duplicate annotation ID in image: {annotation_id}")
        local_annotation_ids.add(annotation_id)
        if annotation.get("image_id") != image_id:
            raise ValueError(f"annotation {annotation_id} references another image")

        category_id = _require_non_negative_integer(
            annotation.get("category_id"),
            name=f"annotations[{index}].category_id",
        )
        if category_id not in category_ids:
            raise ValueError(f"annotation category is missing: {category_id}")
        drug_code = annotation.get("drug_code")
        if (
            not isinstance(drug_code, str)
            or re.fullmatch(r"K-\d{6}", drug_code) is None
        ):
            raise ValueError(f"annotation {annotation_id} has an invalid drug_code")
        if int(drug_code.removeprefix("K-")) != category_id:
            raise ValueError(
                f"annotation {annotation_id} category and drug_code differ"
            )
        drug_codes.append(drug_code)

        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"annotation {annotation_id} bbox must be XYWH")
        if any(
            isinstance(number, bool) or not isinstance(number, int | float)
            for number in bbox
        ):
            raise TypeError(f"annotation {annotation_id} bbox must be numeric")
        x, y, box_width, box_height = map(float, bbox)
        if not all(math.isfinite(number) for number in (x, y, box_width, box_height)):
            raise ValueError(f"annotation {annotation_id} bbox must be finite")
        if (
            x < 0
            or y < 0
            or box_width <= 0
            or box_height <= 0
            or x + box_width > width
            or y + box_height > height
        ):
            raise ValueError(f"annotation {annotation_id} bbox is outside the image")
        area = annotation.get("area")
        if (
            isinstance(area, bool)
            or not isinstance(area, int | float)
            or not math.isclose(float(area), box_width * box_height)
        ):
            raise ValueError(f"annotation {annotation_id} area does not match bbox")
        if annotation.get("iscrowd", 0) != 0:
            raise ValueError(f"annotation {annotation_id} iscrowd must be zero")

    group_id = canonical_group(file_name)
    expected_codes = set(group_id.removeprefix("K-").split("-"))
    actual_codes = {code.removeprefix("K-") for code in drug_codes}
    if len(drug_codes) != len(actual_codes):
        raise ValueError("source annotation contains a duplicate drug code")
    if actual_codes != expected_codes or len(drug_codes) != len(expected_codes):
        raise ValueError(
            "annotation drug codes must cover every pill code in the image file name"
        )

    return DetectorSourceAnnotation(
        dataset="codeit",
        annotation_path=Path(annotation_path).resolve(),
        image_path=image_path,
        group_id=group_id,
        document=value,
    )


def _require_non_negative_integer(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_positive_integer(value: Any, *, name: str) -> int:
    value = _require_non_negative_integer(value, name=name)
    if value == 0:
        raise ValueError(f"{name} must be greater than zero")
    return value
