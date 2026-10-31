"""공통 저수준 입력 검증 함수 테스트."""

import pytest
import torch

from data_loading.validation import (
    normalize_bbox_xywh,
    normalize_probability,
    require_fields,
    require_mapping,
    require_non_empty_string,
    validate_rgb_image_tensor,
)


class DomainValidationError(ValueError):
    """도메인별 예외 주입을 확인하기 위한 테스트 예외."""


def test_mapping_and_required_fields_support_domain_error() -> None:
    with pytest.raises(DomainValidationError, match="must be a mapping"):
        require_mapping([], "batch", error_type=DomainValidationError)
    with pytest.raises(DomainValidationError, match="missing required fields"):
        require_fields(
            {},
            {"images"},
            "batch",
            error_type=DomainValidationError,
        )


def test_non_empty_string_supports_strict_and_legacy_whitespace_rules() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        require_non_empty_string("   ", "name")

    assert require_non_empty_string("   ", "name", strip=False) == "   "


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_normalize_probability_accepts_closed_unit_interval(value: float) -> None:
    assert normalize_probability(value, name="score") == value


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_normalize_probability_rejects_invalid_value(value: float) -> None:
    with pytest.raises(ValueError, match="between"):
        normalize_probability(value, name="score")


def test_normalize_bbox_xywh_returns_float_tuple() -> None:
    assert normalize_bbox_xywh([1, 2, 3, 4], name="bbox") == (1.0, 2.0, 3.0, 4.0)


def test_rgb_image_validation_returns_height_and_width() -> None:
    image = torch.zeros(3, 12, 16, dtype=torch.float32)

    assert validate_rgb_image_tensor(image) == (12, 16)


def test_rgb_image_validation_can_use_domain_error() -> None:
    with pytest.raises(DomainValidationError, match=r"torch\.float32"):
        validate_rgb_image_tensor(
            torch.zeros(3, 4, 4, dtype=torch.float64),
            type_error=DomainValidationError,
            value_error=DomainValidationError,
        )
