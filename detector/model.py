"""torchvision 기반 탐지 모델 생성과 장치 선택."""

from collections.abc import Callable

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSD

DetectorBuilder = Callable[..., nn.Module]


def build_ssdlite320(
    *,
    num_classes: int,
    pretrained_backbone: bool = True,
    nms_threshold: float = 0.55,
) -> SSD:
    """MobileNetV3-Large backbone을 사용하는 SSDLite320을 생성한다."""

    _validate_num_classes(num_classes)
    if not isinstance(pretrained_backbone, bool):
        raise TypeError("pretrained_backbone must be a boolean")
    _validate_probability(nms_threshold, name="nms_threshold")

    backbone_weights = (
        MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    )
    return ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=num_classes,
        nms_thresh=nms_threshold,
    )


MODEL_BUILDERS: dict[str, DetectorBuilder] = {
    "ssdlite320_mobilenet_v3_large": build_ssdlite320,
}


def available_detector_models() -> tuple[str, ...]:
    """등록된 탐지 모델 이름을 정렬해 반환한다."""

    return tuple(sorted(MODEL_BUILDERS))


def build_detector_model(
    model_name: str,
    *,
    num_classes: int,
    pretrained_backbone: bool = True,
    nms_threshold: float = 0.55,
) -> nn.Module:
    """registry에서 이름으로 탐지 모델 builder를 찾아 실행한다."""

    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string")
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    _validate_num_classes(num_classes)
    if not isinstance(pretrained_backbone, bool):
        raise TypeError("pretrained_backbone must be a boolean")
    _validate_probability(nms_threshold, name="nms_threshold")

    try:
        builder = MODEL_BUILDERS[model_name]
    except KeyError as error:
        available_names = ", ".join(available_detector_models())
        raise ValueError(
            f"unknown detector model {model_name!r}; available models: {available_names}"
        ) from error
    return builder(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        nms_threshold=nms_threshold,
    )


def select_device() -> torch.device:
    """CUDA, MPS, CPU 순서로 사용 가능한 장치를 선택한다."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _validate_num_classes(num_classes: int) -> None:
    if not isinstance(num_classes, int) or isinstance(num_classes, bool):
        raise TypeError("num_classes must be an integer")
    if num_classes < 2:
        raise ValueError(
            "num_classes must include background and at least one foreground class"
        )


def _validate_probability(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
