"""탐지 bbox를 기존 분류기 추론 batch로 변환한다."""

from collections.abc import Sequence
from math import isfinite
from numbers import Real
from typing import Any

import torch

from detector.inference import DetectorPrediction
from pilldata.transforms import (
    CLASSIFIER_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PADDING_COLOR_RGB,
    clip_xyxy_to_image,
    crop_tensor_xyxy,
    preprocess_classifier_crop,
)

__all__ = [
    "CLASSIFIER_IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PADDING_COLOR_RGB",
    "build_classifier_batch",
    "combine_scores",
]


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

        height, width = image.shape[-2:]
        clipped_box = clip_xyxy_to_image(
            detection.bbox_xyxy,
            height=height,
            width=width,
        )
        if clipped_box is None:
            raise ValueError("bbox_xyxy must have positive area after clipping")
        crop = crop_tensor_xyxy(image, detection.bbox_xyxy)
        classifier_images.append(preprocess_classifier_crop(crop))
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


def _validate_score(value: Any, *, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    score = float(value)
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return score
