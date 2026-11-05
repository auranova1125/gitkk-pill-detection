"""탐지 이미지와 bbox에 같은 학습 증강을 적용한다."""

import math
import random
from collections.abc import Mapping
from typing import Protocol

import torch
from PIL import Image, ImageEnhance

DETECTOR_AUGMENTATION_KEYS = (
    "horizontal_flip",
    "vertical_flip",
    "brightness",
    "contrast",
)
DEFAULT_DETECTOR_AUGMENTATION = {name: 0.0 for name in DETECTOR_AUGMENTATION_KEYS}


class DetectorAugmentationConfig(Protocol):
    horizontal_flip: float
    vertical_flip: float
    brightness: float
    contrast: float


def build_detector_train_augmentation(
    config: DetectorAugmentationConfig,
) -> dict[str, float]:
    """실행 설정에서 검증된 탐지 증강 값만 추출한다."""

    augmentation = {
        "horizontal_flip": config.horizontal_flip,
        "vertical_flip": config.vertical_flip,
        "brightness": config.brightness,
        "contrast": config.contrast,
    }
    validate_detector_augmentation(augmentation)
    return augmentation


def validate_detector_augmentation(augmentation: Mapping[str, float]) -> None:
    """탐지 baseline이 지원하는 증강 항목과 범위를 확인한다."""

    if set(augmentation) != set(DETECTOR_AUGMENTATION_KEYS):
        raise ValueError("증강 설정 항목이 올바르지 않습니다.")
    for name, value in augmentation.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"증강 설정의 숫자 오류: {name}")
        upper = 1.0 if name.endswith("flip") else 0.3
        if not 0 <= value <= upper:
            raise ValueError(f"증강 설정 범위 오류: {name}")


def apply_detector_augmentation(
    image: Image.Image,
    boxes: torch.Tensor,
    *,
    augmentation: Mapping[str, float],
    rng: random.Random,
) -> tuple[Image.Image, torch.Tensor]:
    """이미지와 XYXY bbox를 함께 뒤집고 밝기·대비를 조정한다."""

    width, height = image.size
    horizontal = augmentation["horizontal_flip"]
    vertical = augmentation["vertical_flip"]

    if horizontal > 0 and rng.random() < horizontal:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        left = boxes[:, 0].clone()
        right = boxes[:, 2].clone()
        boxes[:, 0] = width - right
        boxes[:, 2] = width - left

    if vertical > 0 and rng.random() < vertical:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        top = boxes[:, 1].clone()
        bottom = boxes[:, 3].clone()
        boxes[:, 1] = height - bottom
        boxes[:, 3] = height - top

    brightness = augmentation["brightness"]
    contrast = augmentation["contrast"]
    if brightness > 0:
        image = ImageEnhance.Brightness(image).enhance(
            rng.uniform(1 - brightness, 1 + brightness)
        )
    if contrast > 0:
        image = ImageEnhance.Contrast(image).enhance(
            rng.uniform(1 - contrast, 1 + contrast)
        )
    return image, boxes
