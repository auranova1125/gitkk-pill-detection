"""탐지기 설정과 입력 검증 테스트."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import data_loading.factory as data_loading_factory_module
from configs.detector import DetectorConfig
from detector.input_validation import (
    InputValidationError,
    validate_detector_data_summary,
    validate_detector_label_map,
    validate_detector_training_batch,
)
from detector.training_data import prepare_detector_data


def make_data_summary() -> dict[str, Any]:
    return {
        "num_train_images": 4,
        "num_val_images": 2,
        "num_train_instances": 7,
        "num_val_instances": 3,
        "train_group_count": 2,
        "val_group_count": 1,
        "group_overlap_count": 0,
        "train_empty_image_count": 0,
        "val_empty_image_count": 0,
        "train_max_instances_per_image": 2,
        "val_max_instances_per_image": 2,
        "train_label_counts": {"1": 7},
        "val_label_counts": {"1": 3},
        "split_hash": "synthetic-split-hash",
    }


class SyntheticDetectionDataset(Dataset[dict[str, Any]]):
    def __init__(self) -> None:
        self._sizes = [(64, 80), (48, 72)]
        self._boxes = [
            [[5.0, 7.0, 25.0, 27.0], [40.0, 30.0, 70.0, 55.0]],
            [[10.0, 8.0, 30.0, 38.0]],
        ]

    def __len__(self) -> int:
        return len(self._sizes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        height, width = self._sizes[index]
        boxes = torch.tensor(self._boxes[index], dtype=torch.float32)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        object_count = boxes.shape[0]
        annotation_ids = list(range(index * 10, index * 10 + object_count))
        bbox_xywh = [
            [
                float(box[0]),
                float(box[1]),
                float(box[2] - box[0]),
                float(box[3] - box[1]),
            ]
            for box in boxes
        ]
        return {
            "image": torch.full(
                (3, height, width),
                fill_value=0.25 * (index + 1),
                dtype=torch.float32,
            ),
            "target": {
                "boxes": boxes,
                "labels": torch.ones(object_count, dtype=torch.int64),
                "image_id": index,
                "area": area,
                "iscrowd": torch.zeros(object_count, dtype=torch.int64),
            },
            "meta": {
                "image_id": index,
                "file_name": f"synthetic-{index}.png",
                "image_path": f"/synthetic/synthetic-{index}.png",
                "group_id": f"group-{index}",
                "original_size": [height, width],
                "annotation_count": object_count,
                "annotation_ids": annotation_ids,
                "bbox_xywh": bbox_xywh,
            },
        }


def collate_detector_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [sample["image"] for sample in samples],
        "targets": [sample["target"] for sample in samples],
        "meta": [sample["meta"] for sample in samples],
    }


@pytest.fixture
def label_map(detector_label_map: dict[str, Any]) -> dict[str, Any]:
    return detector_label_map


@pytest.fixture
def valid_batch() -> dict[str, Any]:
    loader = DataLoader(
        SyntheticDetectionDataset(),
        batch_size=2,
        shuffle=False,
        collate_fn=collate_detector_samples,
    )
    return next(iter(loader))


def make_config(tmp_path: Path, **overrides: Any) -> DetectorConfig:
    values = {
        "run_name": "ssdlite-baseline",
        "output_dir": tmp_path / "outputs",
        "learning_rate": 1e-3,
        "weight_decay": 5e-4,
        "score_threshold": 0.25,
        "nms_threshold": 0.45,
        "early_stopping_patience": 7,
    }
    values.update(overrides)
    return DetectorConfig(**values)


def test_detector_config_defaults_and_serialization(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    assert config.model_name == "fasterrcnn_resnet50_fpn_v2"
    assert config.seed == 42
    assert config.batch_size == 1
    assert config.image_min_size is None
    assert config.image_max_size is None
    assert config.max_epochs == 50
    assert config.momentum == 0.9
    assert config.horizontal_flip == 0.0
    assert config.vertical_flip == 0.0
    assert config.brightness == 0.0
    assert config.contrast == 0.0
    assert config.to_dict()["output_dir"] == str(tmp_path / "outputs")
    assert config.to_dict()["resume_from"] is None


def test_prepare_detector_data_returns_fresh_loaders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_batch: dict[str, Any],
    label_map: dict[str, Any],
) -> None:
    summary = make_data_summary()
    loaders = [([valid_batch], [valid_batch]), ([valid_batch], [valid_batch])]
    factory_calls = 0

    def factory(*, config: DetectorConfig, device: torch.device) -> tuple[Any, ...]:
        nonlocal factory_calls
        assert config.device == "cpu"
        assert device == torch.device("cpu")
        train_loader, val_loader = loaders[factory_calls]
        factory_calls += 1
        return train_loader, val_loader, label_map, summary

    monkeypatch.setattr(
        data_loading_factory_module,
        "load_data_factory",
        lambda _: factory,
    )
    config = make_config(
        tmp_path,
        data_factory_path="synthetic.module:factory",
        device="cpu",
    )

    prepared = prepare_detector_data(config)

    assert factory_calls == 2
    assert prepared.train_loader is loaders[1][0]
    assert prepared.val_loader is loaders[1][1]


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("run_name", ""),
        ("model_name", ""),
        ("seed", -1),
        ("batch_size", 0),
        ("image_min_size", 0),
        ("image_max_size", 0),
        ("max_epochs", 0),
        ("learning_rate", 0.0),
        ("learning_rate", float("nan")),
        ("momentum", 1.0),
        ("weight_decay", -1.0),
        ("horizontal_flip", -0.1),
        ("vertical_flip", 1.1),
        ("brightness", 0.31),
        ("contrast", -0.1),
        ("score_threshold", 1.1),
        ("nms_threshold", -0.1),
        ("early_stopping_patience", 0),
    ],
)
def test_detector_config_rejects_invalid_values(
    tmp_path: Path,
    field_name: str,
    field_value: Any,
) -> None:
    overrides = {field_name: field_value}
    if field_name == "image_min_size":
        overrides["image_max_size"] = 640
    elif field_name == "image_max_size":
        overrides["image_min_size"] = 384
    with pytest.raises(ValueError):
        make_config(tmp_path, **overrides)


def test_detector_config_requires_image_sizes_together(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="configured together"):
        make_config(tmp_path, image_min_size=384)


def test_detector_config_rejects_reversed_image_sizes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="less than or equal"):
        make_config(tmp_path, image_min_size=640, image_max_size=384)


def test_validate_detector_label_map_accepts_pill_mapping(
    detector_label_map: dict[str, Any],
) -> None:
    validate_detector_label_map(detector_label_map)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("num_foreground_classes", 2),
        ("num_model_classes", 1),
        ("background_label", 1),
    ],
)
def test_validate_detector_label_map_rejects_invalid_counts(
    detector_label_map: dict[str, Any],
    field_name: str,
    field_value: int,
) -> None:
    label_map = detector_label_map
    label_map[field_name] = field_value

    with pytest.raises(InputValidationError, match=field_name):
        validate_detector_label_map(label_map)


def test_validate_detector_label_map_rejects_non_pill_class(
    detector_label_map: dict[str, Any],
) -> None:
    label_map = detector_label_map
    label_map["classes"][0] = {"label": 2, "name": "tablet"}

    with pytest.raises(InputValidationError):
        validate_detector_label_map(label_map)


def test_validate_detector_data_summary_accepts_consistent_counts(
    label_map: dict[str, Any],
) -> None:
    validate_detector_data_summary(make_data_summary(), label_map)


@pytest.mark.parametrize(
    ("field_name", "field_value", "error_text"),
    [
        ("group_overlap_count", 1, "group_overlap_count"),
        ("train_empty_image_count", 1, "train_empty_image_count"),
        ("num_train_instances", 6, "num_train_instances"),
        ("train_label_counts", {"1": 6}, "train_label_counts"),
        ("train_label_counts", {"0": 7}, "pill label"),
    ],
)
def test_validate_detector_data_summary_rejects_inconsistent_values(
    label_map: dict[str, Any],
    field_name: str,
    field_value: Any,
    error_text: str,
) -> None:
    summary = make_data_summary()
    summary[field_name] = field_value

    with pytest.raises(InputValidationError, match=error_text):
        validate_detector_data_summary(summary, label_map)


def test_validate_detector_training_batch_accepts_multiple_images_and_boxes(
    valid_batch: dict[str, Any],
    label_map: dict[str, Any],
) -> None:
    validate_detector_training_batch(valid_batch, label_map)


def test_validate_detector_training_batch_requires_lists(
    valid_batch: dict[str, Any],
    label_map: dict[str, Any],
) -> None:
    invalid_batch = {
        **valid_batch,
        "images": torch.zeros(2, 3, 64, 80, dtype=torch.float32),
    }

    with pytest.raises(InputValidationError, match=r"batch\.images must be a list"):
        validate_detector_training_batch(invalid_batch, label_map)


@pytest.mark.parametrize(
    ("replacement", "error_text"),
    [
        (torch.zeros(1, 64, 80), "shape"),
        (torch.zeros(3, 64, 80, dtype=torch.float64), "float32"),
        (torch.full((3, 64, 80), 1.1), "between 0.0 and 1.0"),
    ],
)
def test_validate_detector_training_batch_rejects_invalid_images(
    valid_batch: dict[str, Any],
    label_map: dict[str, Any],
    replacement: torch.Tensor,
    error_text: str,
) -> None:
    invalid_batch = deepcopy(valid_batch)
    invalid_batch["images"][0] = replacement

    with pytest.raises(InputValidationError, match=error_text):
        validate_detector_training_batch(invalid_batch, label_map)


def test_validate_detector_training_batch_rejects_box_outside_image(
    valid_batch: dict[str, Any],
    label_map: dict[str, Any],
) -> None:
    invalid_batch = deepcopy(valid_batch)
    invalid_batch["targets"][0]["boxes"][0, 2] = 81.0

    with pytest.raises(InputValidationError, match="inside the image"):
        validate_detector_training_batch(invalid_batch, label_map)


def test_validate_detector_training_batch_rejects_incorrect_area(
    valid_batch: dict[str, Any],
    label_map: dict[str, Any],
) -> None:
    invalid_batch = deepcopy(valid_batch)
    invalid_batch["targets"][0]["area"][0] = 1.0

    with pytest.raises(InputValidationError, match="calculated from boxes"):
        validate_detector_training_batch(invalid_batch, label_map)


def test_validate_detector_training_batch_rejects_non_pill_label(
    valid_batch: dict[str, Any],
    label_map: dict[str, Any],
) -> None:
    invalid_batch = deepcopy(valid_batch)
    invalid_batch["targets"][0]["labels"][0] = 0

    with pytest.raises(InputValidationError, match="pill label 1"):
        validate_detector_training_batch(invalid_batch, label_map)


def test_validate_detector_training_batch_rejects_annotation_count_mismatch(
    valid_batch: dict[str, Any],
    label_map: dict[str, Any],
) -> None:
    invalid_batch = deepcopy(valid_batch)
    invalid_batch["meta"][0]["annotation_count"] = 1

    with pytest.raises(InputValidationError, match="counts must match"):
        validate_detector_training_batch(invalid_batch, label_map)


def test_validate_detector_training_batch_rejects_duplicate_annotation_ids(
    valid_batch: dict[str, Any],
    label_map: dict[str, Any],
) -> None:
    invalid_batch = deepcopy(valid_batch)
    invalid_batch["meta"][0]["annotation_ids"] = [10, 10]

    with pytest.raises(InputValidationError, match="must not contain duplicates"):
        validate_detector_training_batch(invalid_batch, label_map)


def test_validate_detector_training_batch_rejects_target_meta_image_id_mismatch(
    valid_batch: dict[str, Any],
    label_map: dict[str, Any],
) -> None:
    invalid_batch = deepcopy(valid_batch)
    invalid_batch["meta"][0]["image_id"] = 99

    with pytest.raises(InputValidationError, match="must match"):
        validate_detector_training_batch(invalid_batch, label_map)


def test_validate_detector_training_batch_rejects_duplicate_image_ids(
    valid_batch: dict[str, Any],
    label_map: dict[str, Any],
) -> None:
    invalid_batch = deepcopy(valid_batch)
    invalid_batch["targets"][1]["image_id"] = 0
    invalid_batch["meta"][1]["image_id"] = 0

    with pytest.raises(InputValidationError, match="image_id values must be unique"):
        validate_detector_training_batch(invalid_batch, label_map)
