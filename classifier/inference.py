"""탐지 crop 분류와 detector score 결합."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from torch import nn

from classifier.checkpoint import hash_class_map, load_checkpoint
from classifier.input_validation import validate_class_map
from classifier.model import build_model
from classifier.train import select_device


@dataclass(frozen=True, slots=True)
class DetectionPrediction:
    """탐지 하나에 대응하는 분류 결과."""

    image_id: str
    detection_id: str
    bbox_xywh: tuple[float, float, float, float]
    detector_score: float
    detector_name: str
    class_index: int
    category_id: int | None
    item_seq: str | None
    classifier_score: float
    final_score: float

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["bbox_xywh"] = list(self.bbox_xywh)
        return values


@dataclass(frozen=True, slots=True)
class InferenceError:
    """예측을 제거하지 않고 함께 기록할 매핑 오류."""

    image_id: str
    detection_id: str
    class_index: int
    message: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """전체 탐지 분류 결과와 오류 목록."""

    predictions: tuple[DetectionPrediction, ...]
    errors: tuple[InferenceError, ...]

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "errors": [error.to_dict() for error in self.errors],
        }


def _validate_bbox(value: Any, batch_index: int) -> tuple[float, float, float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 4
    ):
        raise ValueError(
            f"batch.meta[{batch_index}].bbox_xywh must contain four numbers"
        )
    if any(not isinstance(item, Real) or isinstance(item, bool) for item in value):
        raise TypeError(f"batch.meta[{batch_index}].bbox_xywh must contain numbers")

    bbox = tuple(float(item) for item in value)
    if any(not isfinite(item) for item in bbox):
        raise ValueError(f"batch.meta[{batch_index}].bbox_xywh must be finite")
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError(
            f"batch.meta[{batch_index}].bbox_xywh width and height must be positive"
        )
    return bbox


def validate_detection_batch(
    batch: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """탐지 crop batch의 tensor와 metadata를 검증한다."""

    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping")
    missing_fields = sorted({"images", "meta"} - batch.keys())
    if missing_fields:
        raise ValueError(f"batch is missing fields: {missing_fields}")

    images = batch["images"]
    meta = batch["meta"]
    if not isinstance(images, torch.Tensor):
        raise TypeError("batch.images must be a torch.Tensor")
    if images.dtype != torch.float32:
        raise TypeError("batch.images must use torch.float32")
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
        raise ValueError("batch.images must have shape [B, 3, 224, 224]")
    if images.shape[0] == 0:
        raise ValueError("batch.images must contain at least one sample")
    if not torch.isfinite(images).all():
        raise ValueError("batch.images must not contain NaN or Inf")
    if not isinstance(meta, list):
        raise TypeError("batch.meta must be a list")
    if len(meta) != images.shape[0]:
        raise ValueError("batch.meta length must equal batch size")

    normalized_meta: list[dict[str, Any]] = []
    required_meta_fields = {
        "image_id",
        "detection_id",
        "bbox_xywh",
        "detector_score",
        "detector_name",
    }
    for index, raw_meta in enumerate(meta):
        if not isinstance(raw_meta, Mapping):
            raise TypeError(f"batch.meta[{index}] must be a mapping")
        missing_meta_fields = sorted(required_meta_fields - raw_meta.keys())
        if missing_meta_fields:
            raise ValueError(
                f"batch.meta[{index}] is missing fields: {missing_meta_fields}"
            )

        image_id = raw_meta["image_id"]
        detection_id = raw_meta["detection_id"]
        detector_name = raw_meta["detector_name"]
        for field_name, value in (
            ("image_id", image_id),
            ("detection_id", detection_id),
            ("detector_name", detector_name),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"batch.meta[{index}].{field_name} must be a non-empty string"
                )

        detector_score = raw_meta["detector_score"]
        if not isinstance(detector_score, Real) or isinstance(detector_score, bool):
            raise TypeError(f"batch.meta[{index}].detector_score must be a number")
        detector_score = float(detector_score)
        if not isfinite(detector_score) or not 0 <= detector_score <= 1:
            raise ValueError(
                f"batch.meta[{index}].detector_score must be between 0 and 1"
            )

        normalized_meta.append(
            {
                "image_id": image_id,
                "detection_id": detection_id,
                "bbox_xywh": _validate_bbox(raw_meta["bbox_xywh"], index),
                "detector_score": detector_score,
                "detector_name": detector_name,
            }
        )
    return normalized_meta


def _class_lookup(
    class_map: Mapping[str, Any],
) -> tuple[int, dict[int, Mapping[str, Any]]]:
    num_classes = class_map.get("num_classes")
    classes = class_map.get("classes")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool):
        raise TypeError("class_map.num_classes must be an integer")
    if num_classes <= 0:
        raise ValueError("class_map.num_classes must be greater than zero")
    if not isinstance(classes, list):
        raise TypeError("class_map.classes must be a list")

    lookup: dict[int, Mapping[str, Any]] = {}
    for entry in classes:
        if not isinstance(entry, Mapping):
            raise TypeError("class_map class entries must be mappings")
        class_index = entry.get("class_index")
        if not isinstance(class_index, int) or isinstance(class_index, bool):
            raise TypeError("class_map class_index values must be integers")
        if class_index in lookup:
            raise ValueError(f"duplicate class_index in class_map: {class_index}")
        lookup[class_index] = entry
    return num_classes, lookup


@torch.inference_mode()
def predict_detections(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    class_map: Mapping[str, Any],
    device: torch.device,
) -> InferenceResult:
    """탐지 crop을 분류하고 detector와 classifier score를 결합한다."""

    num_classes, class_lookup = _class_lookup(class_map)
    model.eval()
    use_amp = device.type == "cuda"
    predictions: list[DetectionPrediction] = []
    errors: list[InferenceError] = []

    for batch in data_loader:
        normalized_meta = validate_detection_batch(batch)
        images = batch["images"].to(device, non_blocking=use_amp)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        if logits.ndim != 2 or tuple(logits.shape) != (
            images.shape[0],
            num_classes,
        ):
            raise ValueError(
                f"model logits must have shape [{images.shape[0]}, {num_classes}]"
            )
        if not torch.isfinite(logits).all():
            raise ValueError("model logits must not contain NaN or Inf")

        probabilities = torch.softmax(logits.float(), dim=1)
        classifier_scores, class_indices = probabilities.max(dim=1)
        for meta, class_index, classifier_score in zip(
            normalized_meta,
            class_indices.cpu().tolist(),
            classifier_scores.cpu().tolist(),
            strict=True,
        ):
            class_entry = class_lookup.get(class_index)
            category_id: int | None = None
            item_seq: str | None = None
            if class_entry is None:
                errors.append(
                    InferenceError(
                        image_id=meta["image_id"],
                        detection_id=meta["detection_id"],
                        class_index=class_index,
                        message="predicted class_index is missing from class_map",
                    )
                )
            else:
                raw_category_id = class_entry.get("category_id")
                raw_item_seq = class_entry.get("item_seq")
                if not isinstance(raw_category_id, int) or isinstance(
                    raw_category_id, bool
                ):
                    errors.append(
                        InferenceError(
                            image_id=meta["image_id"],
                            detection_id=meta["detection_id"],
                            class_index=class_index,
                            message="predicted class has an invalid category_id",
                        )
                    )
                elif not isinstance(raw_item_seq, str) or not raw_item_seq:
                    errors.append(
                        InferenceError(
                            image_id=meta["image_id"],
                            detection_id=meta["detection_id"],
                            class_index=class_index,
                            message="predicted class has an invalid item_seq",
                        )
                    )
                else:
                    category_id = raw_category_id
                    item_seq = raw_item_seq

            final_score = meta["detector_score"] * classifier_score
            predictions.append(
                DetectionPrediction(
                    image_id=meta["image_id"],
                    detection_id=meta["detection_id"],
                    bbox_xywh=meta["bbox_xywh"],
                    detector_score=meta["detector_score"],
                    detector_name=meta["detector_name"],
                    class_index=class_index,
                    category_id=category_id,
                    item_seq=item_seq,
                    classifier_score=classifier_score,
                    final_score=final_score,
                )
            )

    return InferenceResult(predictions=tuple(predictions), errors=tuple(errors))


def load_inference_model(
    checkpoint_path: Path,
    class_map: Mapping[str, Any],
    *,
    expected_num_classes: int = 118,
    device: torch.device | None = None,
) -> tuple[nn.Module, torch.device]:
    """checkpoint에서 사전학습 다운로드 없이 추론 모델을 복원한다."""

    validate_class_map(class_map, expected_num_classes=expected_num_classes)
    selected_device = device if device is not None else select_device()
    checkpoint = load_checkpoint(checkpoint_path, selected_device)
    if checkpoint["class_map_hash"] != hash_class_map(class_map):
        raise ValueError("checkpoint class map hash does not match current class map")

    model = build_model(expected_num_classes, pretrained=False).to(selected_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, selected_device
