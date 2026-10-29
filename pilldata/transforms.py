"""알약 crop, padding, resize와 정규화를 위한 공통 전처리."""

from collections.abc import Sequence
from math import isfinite
from numbers import Real
from typing import Any

import torch
from PIL import Image
from torch.nn import functional
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from pilldata.validation import validate_rgb_image_tensor

CLASSIFIER_IMAGE_SIZE = 224
CROP_MARGIN = 0.05
PADDING_COLOR_RGB = (124, 116, 104)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BBoxXYXY = tuple[float, float, float, float]


def clip_xyxy_to_image(
    bbox_xyxy: Sequence[Real],
    *,
    height: int,
    width: int,
) -> BBoxXYXY | None:
    """XYXY bbox를 이미지 경계로 제한하고 면적이 없으면 None을 반환한다."""

    _validate_image_size(height=height, width=width)
    x1, y1, x2, y2 = _normalize_box(bbox_xyxy, name="bbox_xyxy")
    clipped = (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def cut_pill(
    image: Image.Image,
    bbox_xywh: Sequence[Real],
    margin: float = CROP_MARGIN,
) -> Image.Image:
    """XYWH bbox에 여백을 더하고 경계로 제한해 RGB 직사각형을 자른다."""

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    x, y, box_width, box_height = _normalize_box(bbox_xywh, name="bbox_xywh")
    if box_width <= 0.0 or box_height <= 0.0:
        raise ValueError("bbox_xywh width and height must be positive")
    bounds = _crop_bounds_xyxy(
        (x, y, x + box_width, y + box_height),
        height=image.height,
        width=image.width,
        margin=margin,
    )
    return image.convert("RGB").crop(bounds)


def pad_square(
    image: Image.Image,
    fill: tuple[int, int, int] = PADDING_COLOR_RGB,
) -> Image.Image:
    """짧은 축을 지정 색으로 채워 RGB 이미지를 중앙 정렬한 정사각형으로 만든다."""

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    square_size = max(image.size)
    output = Image.new("RGB", (square_size, square_size), fill)
    output.paste(
        image.convert("RGB"),
        ((square_size - image.width) // 2, (square_size - image.height) // 2),
    )
    return output


def crop_pill(
    image: Image.Image,
    bbox_xywh: Sequence[Real],
    margin: float = CROP_MARGIN,
    fill: tuple[int, int, int] = PADDING_COLOR_RGB,
) -> Image.Image:
    """여백을 포함해 알약을 자르고 정사각형으로 padding한다."""

    return pad_square(cut_pill(image, bbox_xywh, margin), fill)


def crop_tensor_xyxy(
    image: torch.Tensor,
    bbox_xyxy: Sequence[Real],
    *,
    margin: float = CROP_MARGIN,
) -> torch.Tensor:
    """Tensor 이미지에서 여백을 포함한 XYXY 영역을 이미지 경계 안에서 자른다."""

    _validate_image_tensor(image)
    height, width = image.shape[-2:]
    bounds = _crop_bounds_xyxy(
        bbox_xyxy,
        height=height,
        width=width,
        margin=margin,
    )
    left, top, right, bottom = bounds
    return image[:, top:bottom, left:right]


def preprocess_classifier_crop(crop: torch.Tensor) -> torch.Tensor:
    """Tensor crop을 정사각형 padding, 224 resize와 ImageNet 정규화한다."""

    _validate_image_tensor(crop)
    crop_height, crop_width = crop.shape[-2:]
    square_size = max(crop_height, crop_width)
    fill = (
        torch.tensor(
            PADDING_COLOR_RGB,
            dtype=crop.dtype,
            device=crop.device,
        ).view(3, 1, 1)
        / 255.0
    )
    square = fill.expand(3, square_size, square_size).clone()
    top = (square_size - crop_height) // 2
    left = (square_size - crop_width) // 2
    square[:, top : top + crop_height, left : left + crop_width] = crop

    resized = functional.interpolate(
        square.unsqueeze(0),
        size=(CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=crop.dtype,
        device=crop.device,
    ).view(3, 1, 1)
    std = torch.tensor(
        IMAGENET_STD,
        dtype=crop.dtype,
        device=crop.device,
    ).view(3, 1, 1)
    return (resized - mean) / std


class PadSquare:
    """PIL 이미지를 중앙 정렬된 정사각형으로 만드는 transform."""

    def __init__(self, fill: tuple[int, int, int] = PADDING_COLOR_RGB) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        return pad_square(image, self.fill)


def build_classifier_train_transform() -> T.Compose:
    """분류기 학습용 회전, padding, resize와 정규화 transform을 만든다."""

    return T.Compose(
        [
            T.RandomRotation(
                (-180, 180),
                interpolation=InterpolationMode.BILINEAR,
                expand=True,
                fill=PADDING_COLOR_RGB,
            ),
            PadSquare(),
            *_classifier_transform_tail(),
        ]
    )


def build_classifier_validation_transform() -> T.Compose:
    """분류기 validation용 결정적 전처리 transform을 만든다."""

    return T.Compose([PadSquare(), *_classifier_transform_tail()])


def build_classifier_inference_transform() -> T.Compose:
    """이미 정사각형으로 padding된 PIL crop의 추론 transform을 만든다."""

    return T.Compose(_classifier_transform_tail())


def _classifier_transform_tail() -> list[Any]:
    return [
        T.Resize(
            (CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE),
            interpolation=InterpolationMode.BILINEAR,
        ),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]


def _crop_bounds_xyxy(
    bbox_xyxy: Sequence[Real],
    *,
    height: int,
    width: int,
    margin: float,
) -> tuple[int, int, int, int]:
    _validate_image_size(height=height, width=width)
    margin = _validate_margin(margin)
    x1, y1, x2, y2 = _normalize_box(bbox_xyxy, name="bbox_xyxy")
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0.0 or box_height <= 0.0:
        raise ValueError("bbox_xyxy must have positive area")

    expanded = (
        x1 - box_width * margin,
        y1 - box_height * margin,
        x2 + box_width * margin,
        y2 + box_height * margin,
    )
    clipped = clip_xyxy_to_image(expanded, height=height, width=width)
    if clipped is None:
        raise ValueError("bbox_xyxy must have positive area after clipping")
    clipped_x1, clipped_y1, clipped_x2, clipped_y2 = clipped
    left = max(0, min(round(clipped_x1), width - 1))
    top = max(0, min(round(clipped_y1), height - 1))
    right = max(left + 1, min(round(clipped_x2), width))
    bottom = max(top + 1, min(round(clipped_y2), height))
    return left, top, right, bottom


def _normalize_box(
    bbox: Sequence[Real],
    *,
    name: str,
) -> BBoxXYXY:
    if isinstance(bbox, (str, bytes)) or not isinstance(bbox, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(bbox) != 4:
        raise ValueError(f"{name} must contain four numbers")
    if any(not isinstance(value, Real) or isinstance(value, bool) for value in bbox):
        raise TypeError(f"{name} must contain only numbers")
    normalized = tuple(float(value) for value in bbox)
    if any(not isfinite(value) for value in normalized):
        raise ValueError(f"{name} must contain only finite values")
    return normalized


def _validate_margin(margin: float) -> float:
    if not isinstance(margin, Real) or isinstance(margin, bool):
        raise TypeError("margin must be a number")
    margin = float(margin)
    if not isfinite(margin) or margin < 0.0:
        raise ValueError("margin must be finite and non-negative")
    return margin


def _validate_image_size(*, height: int, width: int) -> None:
    for name, value in (("height", height), ("width", width)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")


def _validate_image_tensor(image: torch.Tensor) -> None:
    validate_rgb_image_tensor(
        image,
        require_cpu=False,
        require_finite=False,
        value_range=None,
    )
