"""분류기 입력 검증."""

from collections.abc import Mapping
from typing import Any

import torch

from pilldata.validation import (
    normalize_bbox_xywh,
    require_fields,
    require_mapping,
    require_non_empty_string,
)


class InputValidationError(ValueError):
    """분류기 입력이 요구사항을 만족하지 않을 때 발생하는 예외."""


_REQUIRED_CLASS_FIELDS = {
    "class_index",
    "category_id",
    "item_seq",
    "name",
}
_REQUIRED_META_FIELDS = {
    "sample_id",
    "image_id",
    "image_path",
    "category_id",
    "item_seq",
    "bbox_xywh",
    "group_id",
    "drug_dir",
}
_REQUIRED_SUMMARY_FIELDS = {
    "num_classes",
    "num_train_samples",
    "num_val_samples",
    "train_class_counts",
    "val_class_counts",
    "train_group_count",
    "val_group_count",
    "group_overlap_count",
    "validation_class_indices",
    "split_hash",
}


def _fail(message: str) -> None:
    raise InputValidationError(message)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_num_classes(expected_num_classes: int) -> None:
    if not _is_integer(expected_num_classes) or expected_num_classes <= 0:
        raise ValueError("expected_num_classes must be a positive integer")


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


def _class_entries_by_index(
    class_map: Mapping[str, Any],
    *,
    expected_num_classes: int,
) -> dict[int, Mapping[str, Any]]:
    validate_class_map(class_map, expected_num_classes=expected_num_classes)
    return {entry["class_index"]: entry for entry in class_map["classes"]}


def validate_class_map(
    class_map: Mapping[str, Any],
    *,
    expected_num_classes: int = 118,
) -> None:
    """클래스 index와 실제 품목 매핑을 검증한다."""

    _validate_num_classes(expected_num_classes)
    class_map = _require_mapping(class_map, "class_map")
    _require_fields(class_map, {"num_classes", "classes"}, "class_map")

    if class_map["num_classes"] != expected_num_classes:
        _fail(
            "class_map.num_classes must equal "
            f"{expected_num_classes}, got {class_map['num_classes']!r}"
        )

    classes = class_map["classes"]
    if not isinstance(classes, list):
        _fail("class_map.classes must be a list")
    if len(classes) != expected_num_classes:
        _fail(
            "class_map.classes length must equal "
            f"{expected_num_classes}, got {len(classes)}"
        )

    class_indices: list[int] = []
    category_ids: list[int] = []
    item_sequences: list[str] = []
    for position, raw_entry in enumerate(classes):
        entry = _require_mapping(raw_entry, f"class_map.classes[{position}]")
        _require_fields(
            entry,
            _REQUIRED_CLASS_FIELDS,
            f"class_map.classes[{position}]",
        )

        class_index = entry["class_index"]
        category_id = entry["category_id"]
        item_seq = entry["item_seq"]
        name = entry["name"]

        if not _is_integer(class_index):
            _fail(f"class_map.classes[{position}].class_index must be an integer")
        if not _is_integer(category_id):
            _fail(f"class_map.classes[{position}].category_id must be an integer")
        if not isinstance(item_seq, str) or not item_seq:
            _fail(f"class_map.classes[{position}].item_seq must be a non-empty string")
        if not isinstance(name, str) or not name:
            _fail(f"class_map.classes[{position}].name must be a non-empty string")

        class_indices.append(class_index)
        category_ids.append(category_id)
        item_sequences.append(item_seq)

    expected_indices = set(range(expected_num_classes))
    if set(class_indices) != expected_indices or len(set(class_indices)) != len(
        class_indices
    ):
        _fail(
            "class_map class_index values must contain each integer from "
            f"0 to {expected_num_classes - 1} exactly once"
        )
    if len(set(category_ids)) != len(category_ids):
        _fail("class_map category_id values must be unique")
    if len(set(item_sequences)) != len(item_sequences):
        _fail("class_map item_seq values must be unique")


def _validate_count_mapping(
    counts: Any,
    *,
    name: str,
    expected_num_classes: int,
    require_positive: bool,
) -> dict[int, int]:
    counts = _require_mapping(counts, name)
    parsed_counts: dict[int, int] = {}

    for raw_index, count in counts.items():
        try:
            class_index = int(raw_index)
        except (TypeError, ValueError):
            _fail(f"{name} keys must be integer strings")
        if str(class_index) != str(raw_index):
            _fail(f"{name} keys must use canonical integer strings")
        if class_index in parsed_counts:
            _fail(f"{name} contains duplicate class index {class_index}")
        if not _is_integer(count) or count < 0:
            _fail(f"{name}[{raw_index!r}] must be a non-negative integer")
        if require_positive and count == 0:
            _fail(f"{name}[{raw_index!r}] must be greater than zero")
        parsed_counts[class_index] = count

    expected_indices = set(range(expected_num_classes))
    if set(parsed_counts) != expected_indices:
        _fail(
            f"{name} must contain exactly the class indices "
            f"0 to {expected_num_classes - 1}"
        )
    return parsed_counts


def validate_data_summary(
    data_summary: Mapping[str, Any],
    class_map: Mapping[str, Any],
    *,
    expected_num_classes: int = 118,
) -> None:
    """분할 요약과 클래스 coverage 정보를 검증한다."""

    _validate_num_classes(expected_num_classes)
    validate_class_map(class_map, expected_num_classes=expected_num_classes)
    data_summary = _require_mapping(data_summary, "data_summary")
    _require_fields(data_summary, _REQUIRED_SUMMARY_FIELDS, "data_summary")

    if data_summary["num_classes"] != expected_num_classes:
        _fail(
            "data_summary.num_classes must equal "
            f"{expected_num_classes}, got {data_summary['num_classes']!r}"
        )

    integer_fields = (
        "num_train_samples",
        "num_val_samples",
        "train_group_count",
        "val_group_count",
        "group_overlap_count",
    )
    for field_name in integer_fields:
        value = data_summary[field_name]
        if not _is_integer(value) or value < 0:
            _fail(f"data_summary.{field_name} must be a non-negative integer")

    if data_summary["group_overlap_count"] != 0:
        _fail("data_summary.group_overlap_count must be zero")

    train_counts = _validate_count_mapping(
        data_summary["train_class_counts"],
        name="data_summary.train_class_counts",
        expected_num_classes=expected_num_classes,
        require_positive=True,
    )
    val_counts = _validate_count_mapping(
        data_summary["val_class_counts"],
        name="data_summary.val_class_counts",
        expected_num_classes=expected_num_classes,
        require_positive=False,
    )

    if sum(train_counts.values()) != data_summary["num_train_samples"]:
        _fail("data_summary.num_train_samples must equal the sum of train_class_counts")
    if sum(val_counts.values()) != data_summary["num_val_samples"]:
        _fail("data_summary.num_val_samples must equal the sum of val_class_counts")

    validation_class_indices = data_summary["validation_class_indices"]
    if not isinstance(validation_class_indices, list):
        _fail("data_summary.validation_class_indices must be a list")
    if any(not _is_integer(index) for index in validation_class_indices):
        _fail("data_summary.validation_class_indices must contain only integers")
    if len(set(validation_class_indices)) != len(validation_class_indices):
        _fail("data_summary.validation_class_indices must not contain duplicates")

    expected_validation_indices = {
        class_index for class_index, count in val_counts.items() if count > 0
    }
    if set(validation_class_indices) != expected_validation_indices:
        _fail(
            "data_summary.validation_class_indices must match classes with positive "
            "validation counts"
        )

    split_hash = data_summary["split_hash"]
    if not isinstance(split_hash, str) or not split_hash:
        _fail("data_summary.split_hash must be a non-empty string")


def _validate_string_field(
    meta: Mapping[str, Any], field_name: str, index: int
) -> None:
    require_non_empty_string(
        meta[field_name],
        f"batch.meta[{index}].{field_name}",
        strip=False,
        type_error=InputValidationError,
        value_error=InputValidationError,
    )


def _validate_bbox(value: Any, index: int) -> None:
    normalize_bbox_xywh(
        value,
        name=f"batch.meta[{index}].bbox_xywh",
        type_error=InputValidationError,
        value_error=InputValidationError,
    )


def validate_batch(
    batch: Mapping[str, Any],
    class_map: Mapping[str, Any],
    *,
    expected_num_classes: int = 118,
) -> None:
    """DataLoader가 반환한 하나의 batch와 품목 매핑을 검증한다."""

    class_entries = _class_entries_by_index(
        class_map,
        expected_num_classes=expected_num_classes,
    )
    batch = _require_mapping(batch, "batch")
    _require_fields(batch, {"images", "targets", "meta"}, "batch")

    images = batch["images"]
    targets = batch["targets"]
    meta = batch["meta"]

    if not isinstance(images, torch.Tensor):
        _fail("batch.images must be a torch.Tensor")
    if images.dtype != torch.float32:
        _fail(f"batch.images must use torch.float32, got {images.dtype}")
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
        _fail(
            f"batch.images must have shape [B, 3, 224, 224], got {tuple(images.shape)}"
        )
    if images.shape[0] == 0:
        _fail("batch.images must contain at least one sample")
    if not torch.isfinite(images).all():
        _fail("batch.images must not contain NaN or Inf")

    batch_size = images.shape[0]
    if not isinstance(targets, torch.Tensor):
        _fail("batch.targets must be a torch.Tensor")
    if targets.dtype != torch.int64:
        _fail(f"batch.targets must use torch.int64, got {targets.dtype}")
    if targets.ndim != 1 or targets.shape[0] != batch_size:
        _fail(
            f"batch.targets must have shape [{batch_size}], got {tuple(targets.shape)}"
        )
    if torch.any((targets < 0) | (targets >= expected_num_classes)):
        _fail(f"batch.targets values must be between 0 and {expected_num_classes - 1}")

    if not isinstance(meta, list):
        _fail("batch.meta must be a list")
    if len(meta) != batch_size:
        _fail(f"batch.meta length must equal batch size {batch_size}, got {len(meta)}")

    for index, (raw_meta, target) in enumerate(
        zip(meta, targets.tolist(), strict=True)
    ):
        sample_meta = _require_mapping(raw_meta, f"batch.meta[{index}]")
        _require_fields(sample_meta, _REQUIRED_META_FIELDS, f"batch.meta[{index}]")

        for field_name in (
            "sample_id",
            "image_id",
            "image_path",
            "item_seq",
            "group_id",
        ):
            _validate_string_field(sample_meta, field_name, index)

        category_id = sample_meta["category_id"]
        if not _is_integer(category_id):
            _fail(f"batch.meta[{index}].category_id must be an integer")
        drug_dir = sample_meta["drug_dir"]
        if drug_dir is not None and not isinstance(drug_dir, str):
            _fail(f"batch.meta[{index}].drug_dir must be a string or None")
        _validate_bbox(sample_meta["bbox_xywh"], index)

        class_entry = class_entries[target]
        if category_id != class_entry["category_id"]:
            _fail(
                f"batch.meta[{index}].category_id does not match target class {target}"
            )
        if sample_meta["item_seq"] != class_entry["item_seq"]:
            _fail(f"batch.meta[{index}].item_seq does not match target class {target}")
