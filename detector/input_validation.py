"""단일 클래스 탐지기 입력 검증."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from pilldata.validation import (
    normalize_bbox_xywh,
    require_fields,
    require_mapping,
    require_non_empty_string,
    validate_rgb_image_tensor,
)


class InputValidationError(ValueError):
    """탐지기 입력이 명세를 만족하지 않을 때 발생하는 예외."""


_REQUIRED_LABEL_MAP_FIELDS = {
    "num_foreground_classes",
    "num_model_classes",
    "background_label",
    "classes",
}
_REQUIRED_CLASS_FIELDS = {"label", "name"}
_REQUIRED_SUMMARY_FIELDS = {
    "num_train_images",
    "num_val_images",
    "num_train_source_annotations",
    "num_val_source_annotations",
    "num_train_instances",
    "num_val_instances",
    "train_group_count",
    "val_group_count",
    "group_overlap_count",
    "train_empty_image_count",
    "val_empty_image_count",
    "train_max_instances_per_image",
    "val_max_instances_per_image",
    "train_label_counts",
    "val_label_counts",
    "split_hash",
}
_REQUIRED_TARGET_FIELDS = {"boxes", "labels", "image_id", "area", "iscrowd"}
_REQUIRED_META_FIELDS = {
    "image_id",
    "source_image_id",
    "file_name",
    "image_path",
    "group_id",
    "original_size",
    "source_annotation_count",
    "annotation_ids",
    "bbox_xywh",
}


def _fail(message: str) -> None:
    raise InputValidationError(message)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    return require_mapping(value, name, error_type=InputValidationError)


def _require_fields(
    value: Mapping[str, Any],
    required_fields: set[str],
    name: str,
) -> None:
    require_fields(
        value,
        required_fields,
        name,
        error_type=InputValidationError,
    )


def _require_non_empty_string(value: Any, name: str) -> str:
    return require_non_empty_string(
        value,
        name,
        type_error=InputValidationError,
        value_error=InputValidationError,
    )


def validate_detector_label_map(label_map: Mapping[str, Any]) -> None:
    """단일 foreground 클래스의 label map을 검증한다."""

    label_map = _require_mapping(label_map, "label_map")
    _require_fields(label_map, _REQUIRED_LABEL_MAP_FIELDS, "label_map")

    expected_values = {
        "num_foreground_classes": 1,
        "num_model_classes": 2,
        "background_label": 0,
    }
    for field_name, expected_value in expected_values.items():
        value = label_map[field_name]
        if not _is_integer(value) or value != expected_value:
            _fail(f"label_map.{field_name} must equal {expected_value}, got {value!r}")

    classes = label_map["classes"]
    if not isinstance(classes, list):
        _fail("label_map.classes must be a list")
    if len(classes) != 1:
        _fail("label_map.classes must contain exactly one foreground class")

    pill_class = _require_mapping(classes[0], "label_map.classes[0]")
    _require_fields(pill_class, _REQUIRED_CLASS_FIELDS, "label_map.classes[0]")
    if not _is_integer(pill_class["label"]) or pill_class["label"] != 1:
        _fail("label_map.classes[0].label must equal 1")
    if pill_class["name"] != "pill":
        _fail("label_map.classes[0].name must equal 'pill'")


def _validate_count_mapping(value: Any, name: str) -> int:
    counts = _require_mapping(value, name)
    if set(counts) != {"1"}:
        _fail(f"{name} must contain exactly the pill label key '1'")
    count = counts["1"]
    if not _is_integer(count) or count < 0:
        _fail(f"{name}['1'] must be a non-negative integer")
    return count


def validate_detector_data_summary(
    data_summary: Mapping[str, Any],
    label_map: Mapping[str, Any],
) -> None:
    """데이터 분할과 annotation 집계가 입력 명세와 일치하는지 검증한다."""

    validate_detector_label_map(label_map)
    data_summary = _require_mapping(data_summary, "data_summary")
    _require_fields(data_summary, _REQUIRED_SUMMARY_FIELDS, "data_summary")

    integer_fields = (
        "num_train_images",
        "num_val_images",
        "num_train_source_annotations",
        "num_val_source_annotations",
        "num_train_instances",
        "num_val_instances",
        "train_group_count",
        "val_group_count",
        "group_overlap_count",
        "train_empty_image_count",
        "val_empty_image_count",
        "train_max_instances_per_image",
        "val_max_instances_per_image",
    )
    for field_name in integer_fields:
        value = data_summary[field_name]
        if not _is_integer(value) or value < 0:
            _fail(f"data_summary.{field_name} must be a non-negative integer")

    for field_name in ("num_train_images", "num_val_images"):
        if data_summary[field_name] == 0:
            _fail(f"data_summary.{field_name} must be greater than zero")

    if data_summary["group_overlap_count"] != 0:
        _fail("data_summary.group_overlap_count must be zero")
    for field_name in ("train_empty_image_count", "val_empty_image_count"):
        if data_summary[field_name] != 0:
            _fail(f"data_summary.{field_name} must be zero")

    _validate_split_summary(data_summary, split="train")
    _validate_split_summary(data_summary, split="val")
    _require_non_empty_string(data_summary["split_hash"], "data_summary.split_hash")


def _validate_split_summary(data_summary: Mapping[str, Any], *, split: str) -> None:
    image_count = data_summary[f"num_{split}_images"]
    source_count = data_summary[f"num_{split}_source_annotations"]
    instance_count = data_summary[f"num_{split}_instances"]
    group_count = data_summary[f"{split}_group_count"]
    max_instances = data_summary[f"{split}_max_instances_per_image"]
    label_count = _validate_count_mapping(
        data_summary[f"{split}_label_counts"],
        f"data_summary.{split}_label_counts",
    )

    if source_count != instance_count:
        _fail(
            f"data_summary.num_{split}_source_annotations must equal "
            f"num_{split}_instances"
        )
    if label_count != instance_count:
        _fail(
            f"data_summary.num_{split}_instances must equal the sum of "
            f"{split}_label_counts"
        )
    if instance_count < image_count:
        _fail(
            f"data_summary.num_{split}_instances must be at least "
            f"num_{split}_images when empty images are not allowed"
        )
    if not 1 <= group_count <= image_count:
        _fail(
            f"data_summary.{split}_group_count must be between 1 and num_{split}_images"
        )
    if not 1 <= max_instances <= instance_count:
        _fail(
            f"data_summary.{split}_max_instances_per_image must be between 1 and "
            f"num_{split}_instances"
        )


def validate_detector_training_batch(
    batch: Mapping[str, Any],
    label_map: Mapping[str, Any],
) -> None:
    """DataLoader가 반환한 하나의 detection batch를 검증한다."""

    validate_detector_label_map(label_map)
    batch = _require_mapping(batch, "batch")
    _require_fields(batch, {"images", "targets", "meta"}, "batch")

    images = batch["images"]
    targets = batch["targets"]
    metadata = batch["meta"]
    for value, name in (
        (images, "batch.images"),
        (targets, "batch.targets"),
        (metadata, "batch.meta"),
    ):
        if not isinstance(value, list):
            _fail(f"{name} must be a list")

    batch_size = len(images)
    if batch_size == 0:
        _fail("batch.images must contain at least one image")
    if len(targets) != batch_size or len(metadata) != batch_size:
        _fail("batch.images, batch.targets, and batch.meta must have equal lengths")

    image_ids: list[int] = []
    for index, (image, raw_target, raw_meta) in enumerate(
        zip(images, targets, metadata, strict=True)
    ):
        height, width = _validate_image(image, index)
        target = _validate_target(raw_target, index, height=height, width=width)
        meta = _validate_meta(raw_meta, index, target=target)
        image_ids.append(meta["image_id"])

    if len(set(image_ids)) != len(image_ids):
        _fail("batch image_id values must be unique")


def _validate_image(image: Any, index: int) -> tuple[int, int]:
    name = f"batch.images[{index}]"
    return validate_rgb_image_tensor(
        image,
        name=name,
        type_error=InputValidationError,
        value_error=InputValidationError,
    )


def _validate_target(
    raw_target: Any,
    index: int,
    *,
    height: int,
    width: int,
) -> Mapping[str, Any]:
    name = f"batch.targets[{index}]"
    target = _require_mapping(raw_target, name)
    _require_fields(target, _REQUIRED_TARGET_FIELDS, name)

    boxes = _require_tensor(
        target["boxes"],
        f"{name}.boxes",
        dtype=torch.float32,
        shape_tail=(4,),
    )
    if boxes.ndim != 2:
        _fail(f"{name}.boxes must have shape [K, 4]")
    object_count = boxes.shape[0]
    if object_count == 0:
        _fail(f"{name}.boxes must contain at least one box")
    if not torch.isfinite(boxes).all():
        _fail(f"{name}.boxes must contain only finite values")

    labels = _require_vector(
        target["labels"], f"{name}.labels", torch.int64, object_count
    )
    area = _require_vector(target["area"], f"{name}.area", torch.float32, object_count)
    iscrowd = _require_vector(
        target["iscrowd"], f"{name}.iscrowd", torch.int64, object_count
    )
    _validate_image_id(target["image_id"], f"{name}.image_id")

    if torch.any(labels != 1):
        _fail(f"{name}.labels must contain only the pill label 1")
    if torch.any(iscrowd != 0):
        _fail(f"{name}.iscrowd must contain only 0")
    if not torch.isfinite(area).all() or torch.any(area <= 0):
        _fail(f"{name}.area must contain only positive finite values")

    x1, y1, x2, y2 = boxes.unbind(dim=1)
    if torch.any(x1 < 0) or torch.any(y1 < 0):
        _fail(f"{name}.boxes minimum coordinates must be inside the image")
    if torch.any(x2 > width) or torch.any(y2 > height):
        _fail(f"{name}.boxes maximum coordinates must be inside the image")
    if torch.any(x2 <= x1) or torch.any(y2 <= y1):
        _fail(f"{name}.boxes must have positive width and height")

    calculated_area = (x2 - x1) * (y2 - y1)
    if not torch.allclose(area, calculated_area, rtol=1e-5, atol=1e-5):
        _fail(f"{name}.area must match the area calculated from boxes")
    return target


def _require_tensor(
    value: Any,
    name: str,
    *,
    dtype: torch.dtype,
    shape_tail: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        _fail(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        _fail(f"{name} must be on CPU")
    if value.dtype != dtype:
        _fail(f"{name} must use {dtype}, got {value.dtype}")
    if tuple(value.shape[-len(shape_tail) :]) != shape_tail:
        _fail(f"{name} must end with shape {shape_tail}, got {tuple(value.shape)}")
    return value


def _require_vector(
    value: Any,
    name: str,
    dtype: torch.dtype,
    expected_length: int,
) -> torch.Tensor:
    tensor = _require_tensor(value, name, dtype=dtype, shape_tail=(expected_length,))
    if tensor.ndim != 1:
        _fail(f"{name} must have shape [{expected_length}]")
    return tensor


def _validate_image_id(value: Any, name: str) -> int:
    if not _is_integer(value) or value < 0:
        _fail(f"{name} must be a non-negative integer")
    return value


def _validate_meta(
    raw_meta: Any,
    index: int,
    *,
    target: Mapping[str, Any],
) -> Mapping[str, Any]:
    name = f"batch.meta[{index}]"
    meta = _require_mapping(raw_meta, name)
    _require_fields(meta, _REQUIRED_META_FIELDS, name)

    image_id = _validate_image_id(meta["image_id"], f"{name}.image_id")
    if image_id != target["image_id"]:
        _fail(f"{name}.image_id must match batch.targets[{index}].image_id")

    for field_name in ("source_image_id", "file_name", "image_path", "group_id"):
        _require_non_empty_string(meta[field_name], f"{name}.{field_name}")
    _validate_original_size(meta["original_size"], f"{name}.original_size")

    source_count = meta["source_annotation_count"]
    if not _is_integer(source_count) or source_count <= 0:
        _fail(f"{name}.source_annotation_count must be a positive integer")

    annotation_ids = meta["annotation_ids"]
    if not isinstance(annotation_ids, list):
        _fail(f"{name}.annotation_ids must be a list")
    if any(not _is_integer(annotation_id) for annotation_id in annotation_ids):
        _fail(f"{name}.annotation_ids must contain only integers")
    if len(set(annotation_ids)) != len(annotation_ids):
        _fail(f"{name}.annotation_ids must not contain duplicates")

    bbox_xywh = meta["bbox_xywh"]
    if not isinstance(bbox_xywh, list):
        _fail(f"{name}.bbox_xywh must be a list")
    for box_index, box in enumerate(bbox_xywh):
        _validate_xywh(box, f"{name}.bbox_xywh[{box_index}]")

    target_count = target["boxes"].shape[0]
    lengths = {
        "source_annotation_count": source_count,
        "annotation_ids": len(annotation_ids),
        "bbox_xywh": len(bbox_xywh),
        "target_boxes": target_count,
        "target_labels": target["labels"].shape[0],
        "target_area": target["area"].shape[0],
        "target_iscrowd": target["iscrowd"].shape[0],
    }
    if len(set(lengths.values())) != 1:
        _fail(f"{name} annotation and target counts must match: {lengths}")
    return meta


def _validate_original_size(value: Any, name: str) -> None:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        _fail(f"{name} must contain [height, width]")
    if any(not _is_integer(dimension) or dimension <= 0 for dimension in value):
        _fail(f"{name} must contain two positive integers")


def _validate_xywh(value: Any, name: str) -> None:
    normalize_bbox_xywh(
        value,
        name=name,
        type_error=InputValidationError,
        value_error=InputValidationError,
    )
