"""분류기 설정과 입력 검증 테스트."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights

import classifier.model as model_module
from classifier.config import TrainConfig
from classifier.input_validation import (
    InputValidationError,
    validate_batch,
    validate_class_map,
    validate_data_summary,
)
from classifier.model import (
    TrainingStage,
    build_model,
    configure_model_for_stage,
    set_model_train_mode,
)
from pilldata.classifier_dataset import (
    build_classifier_inputs,
    read_samples,
)
from pilldata.classifier_dataset import (
    make_data_summary as make_runtime_data_summary,
)

NUM_CLASSES = 8


def make_class_map(num_classes: int = NUM_CLASSES) -> dict[str, Any]:
    return {
        "num_classes": num_classes,
        "classes": [
            {
                "class_index": index,
                "category_id": 1900 + index,
                "item_seq": f"item-{index:03d}",
                "name": f"합성 품목 {index}",
            }
            for index in range(num_classes)
        ],
    }


def make_data_summary(num_classes: int = NUM_CLASSES) -> dict[str, Any]:
    train_counts = {str(index): 2 for index in range(num_classes)}
    val_counts = {
        str(index): 1 if index < num_classes - 1 else 0 for index in range(num_classes)
    }
    return {
        "num_classes": num_classes,
        "num_train_samples": sum(train_counts.values()),
        "num_val_samples": sum(val_counts.values()),
        "train_class_counts": train_counts,
        "val_class_counts": val_counts,
        "train_group_count": num_classes,
        "val_group_count": num_classes - 1,
        "group_overlap_count": 0,
        "validation_class_indices": list(range(num_classes - 1)),
        "split_hash": "synthetic-split",
    }


class SyntheticPillDataset(Dataset[dict[str, Any]]):
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_classes

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": torch.full((3, 224, 224), float(index), dtype=torch.float32),
            "target": index,
            "meta": {
                "sample_id": f"sample-{index}",
                "image_id": f"image-{index}",
                "image_path": f"/synthetic/image-{index}.png",
                "category_id": 1900 + index,
                "item_seq": f"item-{index:03d}",
                "bbox_xywh": [10.0, 20.0, 30.0, 40.0],
                "group_id": f"group-{index}",
                "drug_dir": None,
            },
        }


def collate_pill_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([sample["image"] for sample in samples]),
        "targets": torch.tensor(
            [sample["target"] for sample in samples],
            dtype=torch.int64,
        ),
        "meta": [sample["meta"] for sample in samples],
    }


@pytest.fixture
def class_map() -> dict[str, Any]:
    return make_class_map()


@pytest.fixture
def valid_batch() -> dict[str, Any]:
    loader = DataLoader(
        SyntheticPillDataset(),
        batch_size=4,
        shuffle=False,
        collate_fn=collate_pill_samples,
    )
    return next(iter(loader))


def test_train_config_defaults_and_serialization(tmp_path: Path) -> None:
    config = TrainConfig(
        run_name="baseline",
        output_dir=tmp_path / "outputs",
        class_map_path=tmp_path / "class_map.json",
        data_summary_path=tmp_path / "data_summary.json",
    )

    assert config.seed == 42
    assert config.batch_size == 32
    assert config.max_epochs == 30
    assert config.head_only_epochs == 3
    assert config.to_dict()["output_dir"] == str(tmp_path / "outputs")
    assert config.to_dict()["resume_from"] is None


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("run_name", ""),
        ("batch_size", 0),
        ("max_epochs", 0),
        ("head_only_epochs", 31),
        ("lr_head_only", 0.0),
        ("weight_decay", -1.0),
        ("early_stopping_patience", 0),
    ],
)
def test_train_config_rejects_invalid_values(
    tmp_path: Path,
    field_name: str,
    field_value: Any,
) -> None:
    values = {
        "run_name": "baseline",
        "output_dir": tmp_path / "outputs",
        "class_map_path": tmp_path / "class_map.json",
        "data_summary_path": tmp_path / "data_summary.json",
        field_name: field_value,
    }

    with pytest.raises(ValueError):
        TrainConfig(**values)


def test_validate_class_map_accepts_synthetic_mapping(
    class_map: dict[str, Any],
) -> None:
    validate_class_map(class_map, expected_num_classes=NUM_CLASSES)


def test_validate_class_map_uses_118_classes_by_default() -> None:
    validate_class_map(make_class_map(num_classes=118))


def test_validate_class_map_rejects_synthetic_mapping_without_override(
    class_map: dict[str, Any],
) -> None:
    with pytest.raises(InputValidationError, match="118"):
        validate_class_map(class_map)


@pytest.mark.parametrize("duplicate_field", ["class_index", "category_id", "item_seq"])
def test_validate_class_map_rejects_duplicates(
    class_map: dict[str, Any],
    duplicate_field: str,
) -> None:
    class_map["classes"][1][duplicate_field] = class_map["classes"][0][duplicate_field]

    with pytest.raises(InputValidationError):
        validate_class_map(class_map, expected_num_classes=NUM_CLASSES)


def test_validate_data_summary_accepts_missing_validation_class(
    class_map: dict[str, Any],
) -> None:
    validate_data_summary(
        make_data_summary(),
        class_map,
        expected_num_classes=NUM_CLASSES,
    )


def test_validate_data_summary_rejects_group_overlap(
    class_map: dict[str, Any],
) -> None:
    data_summary = make_data_summary()
    data_summary["group_overlap_count"] = 1

    with pytest.raises(InputValidationError, match="group_overlap_count"):
        validate_data_summary(
            data_summary,
            class_map,
            expected_num_classes=NUM_CLASSES,
        )


def test_validate_data_summary_requires_every_training_class(
    class_map: dict[str, Any],
) -> None:
    data_summary = make_data_summary()
    data_summary["train_class_counts"][str(NUM_CLASSES - 1)] = 0
    data_summary["num_train_samples"] -= 2

    with pytest.raises(InputValidationError, match="must be greater than zero"):
        validate_data_summary(
            data_summary,
            class_map,
            expected_num_classes=NUM_CLASSES,
        )


def test_validate_data_summary_rejects_validation_index_mismatch(
    class_map: dict[str, Any],
) -> None:
    data_summary = make_data_summary()
    data_summary["validation_class_indices"].append(NUM_CLASSES - 1)

    with pytest.raises(InputValidationError, match="validation_class_indices"):
        validate_data_summary(
            data_summary,
            class_map,
            expected_num_classes=NUM_CLASSES,
        )


def test_validate_batch_accepts_synthetic_loader_batch(
    valid_batch: dict[str, Any],
    class_map: dict[str, Any],
) -> None:
    validate_batch(valid_batch, class_map, expected_num_classes=NUM_CLASSES)


def test_runtime_classifier_factory_reads_training_package(tmp_path: Path) -> None:
    num_classes = 2
    class_map = make_class_map(num_classes)
    (tmp_path / "class_map.json").write_text(
        json.dumps(class_map, ensure_ascii=False),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    for split_index, split in enumerate(("train", "val")):
        for class_index in range(num_classes):
            sample_id = f"{split}-{class_index}"
            crop_path = Path("crops") / f"item-{class_index:03d}" / f"{sample_id}.png"
            absolute_crop_path = tmp_path / crop_path
            absolute_crop_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB",
                (12 + class_index, 10 + split_index),
                color=(80 + class_index, 100, 120),
            ).save(absolute_crop_path)
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_id": f"image-{sample_id}",
                    "image_path": f"/synthetic/{sample_id}.png",
                    "category_id": 1900 + class_index,
                    "item_seq": f"item-{class_index:03d}",
                    "bbox_xywh": json.dumps([1.0, 2.0, 8.0, 6.0]),
                    "group_id": f"group-{sample_id}",
                    "drug_dir": "",
                    "class_index": class_index,
                    "crop_path": str(crop_path),
                    "split": split,
                }
            )

    pd.DataFrame(rows).to_csv(tmp_path / "samples.csv", index=False)
    data_summary = make_runtime_data_summary(read_samples(tmp_path), num_classes)
    (tmp_path / "data_summary.json").write_text(
        json.dumps(data_summary, ensure_ascii=False),
        encoding="utf-8",
    )
    config = TrainConfig(
        run_name="runtime-loader",
        output_dir=tmp_path / "outputs",
        class_map_path=tmp_path / "class_map.json",
        data_summary_path=tmp_path / "data_summary.json",
        batch_size=2,
    )

    train_loader, val_loader, loaded_class_map, loaded_summary = (
        build_classifier_inputs(config=config, device=torch.device("cpu"))
    )

    assert loaded_class_map == class_map
    assert loaded_summary == data_summary
    validate_batch(
        next(iter(train_loader)),
        loaded_class_map,
        expected_num_classes=num_classes,
    )
    validate_batch(
        next(iter(val_loader)),
        loaded_class_map,
        expected_num_classes=num_classes,
    )


@pytest.mark.parametrize(
    ("field_name", "field_value", "error_text"),
    [
        ("images", torch.zeros(4, 3, 32, 32), "shape"),
        ("images", torch.zeros(4, 3, 224, 224, dtype=torch.float64), "float32"),
        ("targets", torch.tensor([0, 1, 2, 8]), "between"),
        ("targets", torch.tensor([0, 1, 2, 3], dtype=torch.int32), "int64"),
    ],
)
def test_validate_batch_rejects_invalid_tensors(
    valid_batch: dict[str, Any],
    class_map: dict[str, Any],
    field_name: str,
    field_value: torch.Tensor,
    error_text: str,
) -> None:
    invalid_batch = {**valid_batch, field_name: field_value}

    with pytest.raises(InputValidationError, match=error_text):
        validate_batch(
            invalid_batch,
            class_map,
            expected_num_classes=NUM_CLASSES,
        )


def test_validate_batch_rejects_non_finite_image(
    valid_batch: dict[str, Any],
    class_map: dict[str, Any],
) -> None:
    invalid_batch = deepcopy(valid_batch)
    invalid_batch["images"][0, 0, 0, 0] = torch.nan

    with pytest.raises(InputValidationError, match="NaN or Inf"):
        validate_batch(
            invalid_batch,
            class_map,
            expected_num_classes=NUM_CLASSES,
        )


def test_validate_batch_rejects_meta_target_mismatch(
    valid_batch: dict[str, Any],
    class_map: dict[str, Any],
) -> None:
    invalid_batch = deepcopy(valid_batch)
    invalid_batch["meta"][0]["category_id"] = 9999

    with pytest.raises(InputValidationError, match="does not match target"):
        validate_batch(
            invalid_batch,
            class_map,
            expected_num_classes=NUM_CLASSES,
        )


def test_validate_batch_requires_meta_list(
    valid_batch: dict[str, Any],
    class_map: dict[str, Any],
) -> None:
    invalid_batch = {**valid_batch, "meta": {"sample_id": ["sample-0"]}}

    with pytest.raises(InputValidationError, match="must be a list"):
        validate_batch(
            invalid_batch,
            class_map,
            expected_num_classes=NUM_CLASSES,
        )


@pytest.mark.parametrize("num_classes", [0, -1])
def test_build_model_rejects_non_positive_class_count(num_classes: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        build_model(num_classes, pretrained=False)


def test_build_model_rejects_boolean_class_count() -> None:
    with pytest.raises(TypeError, match="integer"):
        build_model(True, pretrained=False)


def test_build_model_uses_imagenet_v2_weights_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_resnet50 = model_module.resnet50
    captured_weights: list[ResNet50_Weights | None] = []

    def build_without_download(*, weights: ResNet50_Weights | None) -> nn.Module:
        captured_weights.append(weights)
        return original_resnet50(weights=None)

    monkeypatch.setattr(model_module, "resnet50", build_without_download)

    model = build_model(NUM_CLASSES)

    assert captured_weights == [ResNet50_Weights.IMAGENET1K_V2]
    assert model.fc.out_features == NUM_CLASSES


def test_build_model_forward_and_backward() -> None:
    model = build_model(NUM_CLASSES, pretrained=False)
    configure_model_for_stage(model, TrainingStage.HEAD_ONLY)
    set_model_train_mode(model)

    images = torch.randn(2, 3, 224, 224)
    logits = model(images)
    logits.sum().backward()

    assert logits.shape == (2, NUM_CLASSES)
    assert model.fc.weight.grad is not None
    assert model.conv1.weight.grad is None


def test_head_only_stage_trains_only_fc() -> None:
    model = build_model(NUM_CLASSES, pretrained=False)

    configure_model_for_stage(model, TrainingStage.HEAD_ONLY)

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names == {"fc.weight", "fc.bias"}


def test_layer4_stage_trains_layer4_without_batch_norm_and_fc() -> None:
    model = build_model(NUM_CLASSES, pretrained=False)

    configure_model_for_stage(model, TrainingStage.LAYER4_FINE_TUNING)

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all(name.startswith(("layer4.", "fc.")) for name in trainable_names)
    assert model.layer4[0].conv1.weight.requires_grad
    assert not model.layer4[0].bn1.weight.requires_grad
    assert model.fc.weight.requires_grad
    assert not model.layer3[-1].conv3.weight.requires_grad


@pytest.mark.parametrize(
    "stage",
    [TrainingStage.HEAD_ONLY, TrainingStage.LAYER4_FINE_TUNING],
)
def test_frozen_batch_norm_stays_in_eval_mode(stage: TrainingStage) -> None:
    model = build_model(NUM_CLASSES, pretrained=False)
    configure_model_for_stage(model, stage)

    set_model_train_mode(model)

    batch_norm_modules = [
        module for module in model.modules() if isinstance(module, nn.BatchNorm2d)
    ]
    assert model.training
    assert batch_norm_modules
    assert all(not module.training for module in batch_norm_modules)
    assert all(
        not parameter.requires_grad
        for module in batch_norm_modules
        for parameter in module.parameters()
    )


def test_configure_model_rejects_unknown_stage() -> None:
    model = build_model(NUM_CLASSES, pretrained=False)

    with pytest.raises(ValueError, match="stage must be one of"):
        configure_model_for_stage(model, "unknown")
