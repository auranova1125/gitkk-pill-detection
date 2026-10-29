"""공통 알약 crop과 분류기 전처리 테스트."""

import numpy as np
import pytest
import torch
from PIL import Image

from classifier_data.pill_square_crop import crop_pill as legacy_crop_pill
from pilldata.transforms import (
    CLASSIFIER_IMAGE_SIZE,
    CROP_MARGIN,
    build_classifier_inference_transform,
    build_classifier_validation_transform,
    crop_pill,
    crop_tensor_xyxy,
    cut_pill,
    preprocess_classifier_crop,
)


def make_image_pair(
    *, height: int = 30, width: int = 40
) -> tuple[Image.Image, torch.Tensor]:
    values = torch.arange(3 * height * width, dtype=torch.int64)
    uint8_image = (values % 256).to(torch.uint8).reshape(3, height, width)
    pil_image = Image.fromarray(uint8_image.permute(1, 2, 0).numpy(), mode="RGB")
    tensor_image = uint8_image.to(torch.float32) / 255.0
    return pil_image, tensor_image


def test_pil_and_tensor_crop_use_same_margin_and_pixel_bounds() -> None:
    pil_image, tensor_image = make_image_pair()
    bbox_xywh = (10.0, 8.0, 20.0, 10.0)
    bbox_xyxy = (10.0, 8.0, 30.0, 18.0)

    pil_crop = cut_pill(pil_image, bbox_xywh)
    tensor_crop = crop_tensor_xyxy(tensor_image, bbox_xyxy)
    expected_tensor = torch.from_numpy(np.asarray(pil_crop).copy()).permute(2, 0, 1)
    expected_tensor = expected_tensor.to(torch.float32) / 255.0

    assert CROP_MARGIN == pytest.approx(0.05)
    assert pil_crop.size == (22, 10)
    torch.testing.assert_close(tensor_crop, expected_tensor, rtol=0.0, atol=0.0)


def test_crop_clips_margin_at_image_boundary() -> None:
    pil_image, tensor_image = make_image_pair(height=12, width=16)
    bbox_xywh = (0.0, 0.0, 10.0, 8.0)

    pil_crop = cut_pill(pil_image, bbox_xywh)
    tensor_crop = crop_tensor_xyxy(tensor_image, (0.0, 0.0, 10.0, 8.0))

    assert pil_crop.size == (10, 8)
    assert tuple(tensor_crop.shape) == (3, 8, 10)


def test_preprocess_classifier_crop_returns_classifier_contract() -> None:
    crop = torch.full((3, 8, 12), 0.25, dtype=torch.float32)

    result = preprocess_classifier_crop(crop)

    assert result.dtype == torch.float32
    assert tuple(result.shape) == (
        3,
        CLASSIFIER_IMAGE_SIZE,
        CLASSIFIER_IMAGE_SIZE,
    )
    assert torch.isfinite(result).all()


def test_validation_and_pil_inference_paths_produce_same_tensor() -> None:
    pil_image, _ = make_image_pair()
    bbox_xywh = (10.0, 8.0, 20.0, 10.0)

    validation_image = build_classifier_validation_transform()(
        cut_pill(pil_image, bbox_xywh)
    )
    inference_image = build_classifier_inference_transform()(
        crop_pill(pil_image, bbox_xywh)
    )

    torch.testing.assert_close(validation_image, inference_image, rtol=0.0, atol=0.0)


def test_legacy_crop_import_uses_canonical_implementation() -> None:
    assert legacy_crop_pill is crop_pill
