"""탐지 모델 registry와 SSDLite smoke test."""

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection.ssd import SSD

import detector.model as model_module
from detector import (
    MODEL_BUILDERS,
    available_detector_models,
    build_detector_model,
    build_ssdlite320,
    select_device,
)

NUM_MODEL_CLASSES = 2


def make_images_and_targets() -> tuple[
    list[torch.Tensor], list[dict[str, torch.Tensor]]
]:
    images = [
        torch.rand(3, 96, 128, dtype=torch.float32),
        torch.rand(3, 80, 112, dtype=torch.float32),
    ]
    targets = [
        {
            "boxes": torch.tensor(
                [[10.0, 12.0, 45.0, 52.0], [60.0, 30.0, 105.0, 82.0]],
                dtype=torch.float32,
            ),
            "labels": torch.ones(2, dtype=torch.int64),
        },
        {
            "boxes": torch.tensor(
                [[20.0, 18.0, 70.0, 65.0]],
                dtype=torch.float32,
            ),
            "labels": torch.ones(1, dtype=torch.int64),
        },
    ]
    return images, targets


def test_available_detector_models_contains_ssdlite() -> None:
    assert available_detector_models() == ("ssdlite320_mobilenet_v3_large",)


def test_build_detector_model_dispatches_registered_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_builder(*, num_classes: int, pretrained_backbone: bool) -> nn.Module:
        captured["num_classes"] = num_classes
        captured["pretrained_backbone"] = pretrained_backbone
        return nn.Identity()

    monkeypatch.setitem(MODEL_BUILDERS, "fake_detector", fake_builder)

    model = build_detector_model(
        "fake_detector",
        num_classes=NUM_MODEL_CLASSES,
        pretrained_backbone=False,
    )

    assert isinstance(model, nn.Identity)
    assert captured == {
        "num_classes": NUM_MODEL_CLASSES,
        "pretrained_backbone": False,
    }


def test_build_detector_model_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown detector model"):
        build_detector_model("unknown", num_classes=NUM_MODEL_CLASSES)


@pytest.mark.parametrize("num_classes", [0, 1, -1])
def test_build_ssdlite_rejects_class_count_without_foreground(
    num_classes: int,
) -> None:
    with pytest.raises(ValueError, match="background"):
        build_ssdlite320(num_classes=num_classes, pretrained_backbone=False)


def test_build_ssdlite_rejects_boolean_class_count() -> None:
    with pytest.raises(TypeError, match="integer"):
        build_ssdlite320(num_classes=True, pretrained_backbone=False)


@pytest.mark.parametrize(
    ("pretrained_backbone", "expected_weights"),
    [
        (True, MobileNet_V3_Large_Weights.IMAGENET1K_V1),
        (False, None),
    ],
)
def test_build_ssdlite_selects_backbone_weights(
    monkeypatch: pytest.MonkeyPatch,
    pretrained_backbone: bool,
    expected_weights: MobileNet_V3_Large_Weights | None,
) -> None:
    captured: dict[str, Any] = {}

    def fake_ssdlite_builder(**kwargs: Any) -> nn.Module:
        captured.update(kwargs)
        return nn.Identity()

    monkeypatch.setattr(
        model_module,
        "ssdlite320_mobilenet_v3_large",
        fake_ssdlite_builder,
    )

    model = build_ssdlite320(
        num_classes=NUM_MODEL_CLASSES,
        pretrained_backbone=pretrained_backbone,
    )

    assert isinstance(model, nn.Identity)
    assert captured == {
        "weights": None,
        "weights_backbone": expected_weights,
        "num_classes": NUM_MODEL_CLASSES,
    }


def test_build_ssdlite_configures_two_model_classes() -> None:
    model = build_ssdlite320(
        num_classes=NUM_MODEL_CLASSES,
        pretrained_backbone=False,
    )

    assert isinstance(model, SSD)
    assert model.head.classification_head.num_columns == NUM_MODEL_CLASSES
    assert model.transform.min_size == (320,)
    assert model.transform.max_size == 320


def test_ssdlite_training_forward_and_backward() -> None:
    model = build_ssdlite320(
        num_classes=NUM_MODEL_CLASSES,
        pretrained_backbone=False,
    )
    images, targets = make_images_and_targets()
    model.train()

    losses = model(images, targets)
    total_loss = sum(losses.values())
    total_loss.backward()

    assert set(losses) == {"bbox_regression", "classification"}
    assert all(loss.ndim == 0 for loss in losses.values())
    assert torch.isfinite(total_loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_ssdlite_evaluation_forward_runs_nms() -> None:
    model = build_ssdlite320(
        num_classes=NUM_MODEL_CLASSES,
        pretrained_backbone=False,
    )
    images, _ = make_images_and_targets()
    model.eval()

    with torch.inference_mode():
        predictions = model(images)

    assert len(predictions) == len(images)
    for prediction in predictions:
        assert set(prediction) >= {"boxes", "labels", "scores"}
        assert prediction["boxes"].ndim == 2
        assert prediction["boxes"].shape[1] == 4
        assert prediction["labels"].dtype == torch.int64
        assert prediction["scores"].dtype == torch.float32
        assert prediction["labels"].shape == prediction["scores"].shape
        assert prediction["boxes"].shape[0] == prediction["scores"].shape[0]


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected_device"),
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_select_device_priority(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    expected_device: str,
) -> None:
    cuda_probe: Callable[[], bool] = lambda: cuda_available
    mps_probe: Callable[[], bool] = lambda: mps_available
    monkeypatch.setattr(torch.cuda, "is_available", cuda_probe)
    monkeypatch.setattr(torch.backends.mps, "is_available", mps_probe)

    assert select_device().type == expected_device
