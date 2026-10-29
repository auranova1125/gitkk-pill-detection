"""모델별 입력 경계에서 재사용하는 저수준 검증 함수."""

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real
from typing import Any

import torch

ExceptionType = type[Exception]


def require_mapping(
    value: Any,
    name: str,
    *,
    error_type: ExceptionType = TypeError,
) -> Mapping[str, Any]:
    """값이 mapping인지 확인하고 타입을 좁혀 반환한다."""

    if not isinstance(value, Mapping):
        raise error_type(f"{name} must be a mapping")
    return value


def require_fields(
    value: Mapping[str, Any],
    required_fields: set[str],
    name: str,
    *,
    error_type: ExceptionType = ValueError,
) -> None:
    """mapping에 필수 필드가 모두 있는지 확인한다."""

    missing_fields = sorted(required_fields - value.keys())
    if missing_fields:
        raise error_type(f"{name} is missing required fields: {missing_fields}")


def require_non_empty_string(
    value: Any,
    name: str,
    *,
    strip: bool = True,
    type_error: ExceptionType = TypeError,
    value_error: ExceptionType = ValueError,
) -> str:
    """비어 있지 않은 문자열을 검증하고 원래 값을 반환한다."""

    if not isinstance(value, str):
        raise type_error(f"{name} must be a string")
    is_empty = not value.strip() if strip else not value
    if is_empty:
        raise value_error(f"{name} must be a non-empty string")
    return value


def normalize_probability(value: Any, *, name: str) -> float:
    """유한한 0~1 숫자를 float로 정규화한다."""

    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    normalized = float(value)
    if not isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return normalized


def normalize_bbox_xywh(
    value: Any,
    *,
    name: str,
    type_error: ExceptionType = TypeError,
    value_error: ExceptionType = ValueError,
) -> tuple[float, float, float, float]:
    """유한하고 양의 크기를 가진 XYWH bbox를 float tuple로 반환한다."""

    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 4
    ):
        raise value_error(f"{name} must contain four numbers")
    if any(not isinstance(item, Real) or isinstance(item, bool) for item in value):
        raise type_error(f"{name} must contain only numbers")
    normalized = tuple(float(item) for item in value)
    if any(not isfinite(item) for item in normalized):
        raise value_error(f"{name} must contain only finite values")
    if normalized[2] <= 0.0 or normalized[3] <= 0.0:
        raise value_error(f"{name} width and height must be positive")
    return normalized


def validate_rgb_image_tensor(
    image: Any,
    *,
    name: str = "image",
    require_cpu: bool = True,
    require_finite: bool = True,
    value_range: tuple[float, float] | None = (0.0, 1.0),
    type_error: ExceptionType = TypeError,
    value_error: ExceptionType = ValueError,
) -> tuple[int, int]:
    """float32 RGB Tensor의 장치, shape와 선택적 값 범위를 검증한다."""

    if not isinstance(image, torch.Tensor):
        raise type_error(f"{name} must be a torch.Tensor")
    if require_cpu and image.device.type != "cpu":
        raise value_error(f"{name} must be on CPU")
    if image.dtype != torch.float32:
        raise type_error(f"{name} must use torch.float32, got {image.dtype}")
    if image.ndim != 3 or image.shape[0] != 3:
        raise value_error(f"{name} must have shape [3, H, W], got {tuple(image.shape)}")
    height, width = image.shape[-2:]
    if height <= 0 or width <= 0:
        raise value_error(f"{name} height and width must be positive")
    if require_finite and not torch.isfinite(image).all():
        raise value_error(f"{name} must not contain NaN or Inf")
    if value_range is not None:
        minimum, maximum = value_range
        if torch.any((image < minimum) | (image > maximum)):
            raise value_error(f"{name} values must be between {minimum} and {maximum}")
    return height, width
