"""torchvision 기반 탐지 모델 생성."""

from collections.abc import Callable

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.ssd import SSD

from data_loading.validation import normalize_probability

DetectorBuilder = Callable[..., nn.Module]


def build_ssdlite320(
    *,
    num_classes: int,
    pretrained_backbone: bool = True,
    nms_threshold: float = 0.55,
    image_min_size: int | None = None,
    image_max_size: int | None = None,
) -> SSD:
    """MobileNetV3-Large backbone을 사용하는 SSDLite320을 생성한다."""

    _validate_num_classes(num_classes)
    if not isinstance(pretrained_backbone, bool):
        raise TypeError("pretrained_backbone must be a boolean")
    nms_threshold = normalize_probability(nms_threshold, name="nms_threshold")
    _validate_image_size_pair(image_min_size, image_max_size)
    if (image_min_size, image_max_size) not in {(None, None), (320, 320)}:
        raise ValueError("SSDLite320 only supports image size 320x320")

    backbone_weights = (
        MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    )
    return ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=num_classes,
        nms_thresh=nms_threshold,
    )


def build_fasterrcnn_resnet50_fpn_v2(
    *,
    num_classes: int,
    pretrained_backbone: bool = True,
    nms_threshold: float = 0.5,
    image_min_size: int | None = None,
    image_max_size: int | None = None,
) -> FasterRCNN:
    """ResNet50-FPN v2 backbone을 사용하는 Faster R-CNN을 생성한다."""

    _validate_num_classes(num_classes)
    if not isinstance(pretrained_backbone, bool):
        raise TypeError("pretrained_backbone must be a boolean")
    nms_threshold = normalize_probability(nms_threshold, name="nms_threshold")
    _validate_image_size_pair(image_min_size, image_max_size)
    resolved_min_size = 800 if image_min_size is None else image_min_size
    resolved_max_size = 1333 if image_max_size is None else image_max_size

    backbone_weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
    return fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=num_classes,
        box_nms_thresh=nms_threshold,
        min_size=resolved_min_size,
        max_size=resolved_max_size,
    )


MODEL_BUILDERS: dict[str, DetectorBuilder] = {
    "fasterrcnn_resnet50_fpn_v2": build_fasterrcnn_resnet50_fpn_v2,
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
    image_min_size: int | None = None,
    image_max_size: int | None = None,
) -> nn.Module:
    """registry에서 이름으로 탐지 모델 builder를 찾아 실행한다."""

    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string")
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    _validate_num_classes(num_classes)
    if not isinstance(pretrained_backbone, bool):
        raise TypeError("pretrained_backbone must be a boolean")
    nms_threshold = normalize_probability(nms_threshold, name="nms_threshold")
    _validate_image_size_pair(image_min_size, image_max_size)

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
        image_min_size=image_min_size,
        image_max_size=image_max_size,
    )


def _validate_num_classes(num_classes: int) -> None:
    if not isinstance(num_classes, int) or isinstance(num_classes, bool):
        raise TypeError("num_classes must be an integer")
    if num_classes < 2:
        raise ValueError(
            "num_classes must include background and at least one foreground class"
        )


def _validate_image_size_pair(
    image_min_size: int | None,
    image_max_size: int | None,
) -> None:
    if (image_min_size is None) != (image_max_size is None):
        raise ValueError(
            "image_min_size and image_max_size must be configured together"
        )
    if image_min_size is None or image_max_size is None:
        return
    for name, value in (
        ("image_min_size", image_min_size),
        ("image_max_size", image_max_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")
    if image_min_size > image_max_size:
        raise ValueError("image_min_size must be less than or equal to image_max_size")
