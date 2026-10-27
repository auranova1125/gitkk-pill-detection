"""탐지 bbox를 기존 분류기 추론 batch로 변환한다."""

from collections.abc import Sequence
from math import ceil, floor, isfinite
from numbers import Real
from typing import Any

import torch
from torch.nn import functional

from detector.inference import DetectorPrediction

CLASSIFIER_IMAGE_SIZE = 224
# ImageNet RGB 평균에 255를 곱한 중립색으로, 정규화 후 padding을 0에 가깝게 만든다.
PADDING_COLOR_RGB = (124, 116, 104)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_classifier_batch(
    image: torch.Tensor,
    detections: Sequence[DetectorPrediction],
) -> dict[str, Any]:
    """원본 이미지와 탐지 목록을 분류기 `predict_detections` 입력으로 만든다."""

    _validate_source_image(image)
    if isinstance(detections, (str, bytes)) or not isinstance(detections, Sequence):
        raise TypeError("detections must be a sequence")
    if not detections:
        return {
            "images": torch.empty(
                (0, 3, CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE),
                dtype=torch.float32,
            ),
            "meta": [],
        }

    classifier_images: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    detection_ids: set[str] = set()
    for index, detection in enumerate(detections):
        if not isinstance(detection, DetectorPrediction):
            raise TypeError(f"detections[{index}] must be a DetectorPrediction")
        if (
            not isinstance(detection.image_id, int)
            or isinstance(detection.image_id, bool)
            or detection.image_id < 0
        ):
            raise ValueError(f"detections[{index}].image_id must be non-negative")
        for field_name, value in (
            ("detection_id", detection.detection_id),
            ("detector_name", detection.detector_name),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"detections[{index}].{field_name} must be a non-empty string"
                )
        if detection.detection_id in detection_ids:
            raise ValueError(f"duplicate detection_id: {detection.detection_id}")
        detection_ids.add(detection.detection_id)
        if detection.label != 1 or detection.label_name != "pill":
            raise ValueError(f"detections[{index}] must represent pill label 1")
        detector_score = _validate_score(
            detection.detector_score,
            name=f"detections[{index}].detector_score",
        )

        clipped_box = _clip_box_to_image(detection.bbox_xyxy, image)
        crop = _crop_with_integer_bounds(image, clipped_box)
        classifier_images.append(_preprocess_crop(crop))
        x1, y1, x2, y2 = clipped_box
        metadata.append(
            {
                "image_id": str(detection.image_id),
                "detection_id": detection.detection_id,
                "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                "detector_score": detector_score,
                "detector_name": detection.detector_name,
            }
        )

    return {
        "images": torch.stack(classifier_images),
        "meta": metadata,
    }


def combine_scores(detector_score: float, classifier_score: float) -> float:
    """검증된 detector와 classifier 확률을 곱해 최종 점수를 반환한다."""

    detector_score = _validate_score(detector_score, name="detector_score")
    classifier_score = _validate_score(classifier_score, name="classifier_score")
    return detector_score * classifier_score


def _validate_source_image(image: Any) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if image.device.type != "cpu":
        raise ValueError("image must be on CPU")
    if image.dtype != torch.float32:
        raise TypeError("image must use torch.float32")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3, H, W]")
    if image.shape[1] <= 0 or image.shape[2] <= 0:
        raise ValueError("image height and width must be positive")
    if not torch.isfinite(image).all():
        raise ValueError("image must not contain NaN or Inf")
    if torch.any((image < 0.0) | (image > 1.0)):
        raise ValueError("image values must be between 0.0 and 1.0")


def _clip_box_to_image(
    bbox_xyxy: Sequence[float],
    image: torch.Tensor,
) -> tuple[float, float, float, float]:
    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain four numbers")
    if any(
        not isinstance(value, Real) or isinstance(value, bool) for value in bbox_xyxy
    ):
        raise TypeError("bbox_xyxy must contain only numbers")
    if any(not isfinite(float(value)) for value in bbox_xyxy):
        raise ValueError("bbox_xyxy must contain only finite values")

    height, width = image.shape[-2:]
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    clipped = (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("bbox_xyxy must have positive area after clipping")
    return clipped


def _crop_with_integer_bounds(
    image: torch.Tensor,
    bbox_xyxy: tuple[float, float, float, float],
) -> torch.Tensor:
    height, width = image.shape[-2:]
    x1, y1, x2, y2 = bbox_xyxy
    left = max(0, min(floor(x1), width - 1))
    top = max(0, min(floor(y1), height - 1))
    right = max(left + 1, min(ceil(x2), width))
    bottom = max(top + 1, min(ceil(y2), height))
    return image[:, top:bottom, left:right]


def _preprocess_crop(crop: torch.Tensor) -> torch.Tensor:
    crop_height, crop_width = crop.shape[-2:]
    square_size = max(crop_height, crop_width)
    fill = torch.tensor(PADDING_COLOR_RGB, dtype=torch.float32).view(3, 1, 1) / 255.0
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
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (resized - mean) / std


def _validate_score(value: Any, *, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    score = float(value)
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return score
