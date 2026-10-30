"""탐지 모델 로딩, 예측 filtering과 결과 직렬화."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from detector.checkpoint import hash_label_map, load_checkpoint
from detector.input_validation import validate_detector_label_map
from detector.model import build_detector_model
from pilldata.transforms import clip_xyxy_to_image
from pilldata.validation import normalize_probability, validate_rgb_image_tensor
from utils.io import write_json
from utils.runtime import select_device


@dataclass(frozen=True, slots=True)
class DetectorPrediction:
    """이미지 하나에서 유지된 pill 탐지 결과."""

    image_id: int
    detection_id: str
    bbox_xyxy: tuple[float, float, float, float]
    bbox_xywh: tuple[float, float, float, float]
    label: int
    label_name: str
    detector_score: float
    detector_name: str

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["bbox_xyxy"] = list(self.bbox_xyxy)
        values["bbox_xywh"] = list(self.bbox_xywh)
        return values


@dataclass(frozen=True, slots=True)
class DetectorInferenceError:
    """구조는 유효하지만 사용할 수 없어 제외한 탐지 기록."""

    image_id: int
    prediction_index: int
    message: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DetectorInferenceResult:
    """전체 탐지 예측과 제외된 항목의 오류 목록."""

    predictions: tuple[DetectorPrediction, ...]
    errors: tuple[DetectorInferenceError, ...]

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "errors": [error.to_dict() for error in self.errors],
        }


@torch.inference_mode()
def predict_detector(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    *,
    score_threshold: float,
    detector_name: str,
) -> DetectorInferenceResult:
    """추론 batch를 순회해 filtering 및 clip된 pill 예측을 만든다."""

    score_threshold = normalize_probability(score_threshold, name="score_threshold")
    if not isinstance(detector_name, str) or not detector_name.strip():
        raise ValueError("detector_name must be a non-empty string")

    model.eval()
    use_amp = device.type == "cuda"
    seen_image_ids: set[int] = set()
    predictions: list[DetectorPrediction] = []
    errors: list[DetectorInferenceError] = []

    for batch in data_loader:
        images, metadata = validate_inference_batch(batch)
        for meta in metadata:
            image_id = meta["image_id"]
            if image_id in seen_image_ids:
                raise ValueError(f"duplicate image_id in inference data: {image_id}")
            seen_image_ids.add(image_id)

        device_images = [image.to(device, non_blocking=use_amp) for image in images]
        with torch.autocast(device_type=device.type, enabled=use_amp):
            raw_predictions = model(device_images)
        normalized_predictions = _normalize_model_predictions(
            raw_predictions,
            expected_count=len(images),
        )

        for image, meta, prediction in zip(
            images,
            metadata,
            normalized_predictions,
            strict=True,
        ):
            height, width = image.shape[-2:]
            kept_index = 0
            for prediction_index in range(prediction["boxes"].shape[0]):
                score = float(prediction["scores"][prediction_index].item())
                if score < score_threshold:
                    continue
                label = int(prediction["labels"][prediction_index].item())
                if label != 1:
                    errors.append(
                        DetectorInferenceError(
                            image_id=meta["image_id"],
                            prediction_index=prediction_index,
                            message=f"expected pill label 1, got {label}",
                        )
                    )
                    continue

                clipped_box = clip_xyxy_to_image(
                    prediction["boxes"][prediction_index].tolist(),
                    height=height,
                    width=width,
                )
                if clipped_box is None:
                    errors.append(
                        DetectorInferenceError(
                            image_id=meta["image_id"],
                            prediction_index=prediction_index,
                            message="box has zero area after clipping",
                        )
                    )
                    continue

                kept_index += 1
                x1, y1, x2, y2 = clipped_box
                predictions.append(
                    DetectorPrediction(
                        image_id=meta["image_id"],
                        detection_id=f"{meta['image_id']}-{kept_index:04d}",
                        bbox_xyxy=clipped_box,
                        bbox_xywh=(x1, y1, x2 - x1, y2 - y1),
                        label=1,
                        label_name="pill",
                        detector_score=score,
                        detector_name=detector_name,
                    )
                )

    return DetectorInferenceResult(
        predictions=tuple(predictions),
        errors=tuple(errors),
    )


def validate_inference_batch(
    batch: Mapping[str, Any],
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    """정답 없는 detector 추론 batch를 검증하고 metadata를 정규화한다."""

    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping")
    missing_fields = sorted({"images", "meta"} - batch.keys())
    if missing_fields:
        raise ValueError(f"batch is missing fields: {missing_fields}")
    images = batch["images"]
    metadata = batch["meta"]
    if not isinstance(images, list) or not isinstance(metadata, list):
        raise TypeError("batch images and meta must be lists")
    if not images or len(images) != len(metadata):
        raise ValueError("batch images and meta must have equal non-zero lengths")

    normalized_metadata: list[dict[str, Any]] = []
    for index, (image, raw_meta) in enumerate(zip(images, metadata, strict=True)):
        _validate_inference_image(image, index=index)
        meta = _normalize_inference_meta(raw_meta, index=index)
        if tuple(meta["original_size"]) != tuple(image.shape[-2:]):
            raise ValueError(
                f"batch.meta[{index}].original_size must match image height and width"
            )
        normalized_metadata.append(meta)
    return images, normalized_metadata


def save_inference_artifacts(
    result: DetectorInferenceResult,
    output_dir: Path,
) -> tuple[Path, Path]:
    """탐지 예측과 filtering 오류를 각각 JSON으로 저장한다."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "val_predictions.json"
    errors_path = output_dir / "error_samples.json"
    write_json(
        predictions_path,
        [prediction.to_dict() for prediction in result.predictions],
    )
    write_json(errors_path, [error.to_dict() for error in result.errors])
    return predictions_path, errors_path


def plot_detection_comparison(
    image: torch.Tensor,
    predictions: Sequence[DetectorPrediction],
    *,
    target: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> Any:
    """원본 이미지에 예측 bbox와 선택적인 정답 bbox를 함께 표시한다."""

    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    _validate_inference_image(image, index=0)
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise TypeError("predictions must be a sequence")

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.imshow(image.permute(1, 2, 0).numpy())
    if title:
        axis.set_title(title)

    if target is not None:
        boxes = target.get("boxes") if isinstance(target, Mapping) else None
        if (
            not isinstance(boxes, torch.Tensor)
            or boxes.ndim != 2
            or boxes.shape[1:] != (4,)
        ):
            raise ValueError("target boxes must be a tensor with shape [K, 4]")
        for box in boxes.detach().cpu().tolist():
            x1, y1, x2, y2 = box
            axis.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                    linestyle="--",
                    label="ground truth",
                )
            )

    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, DetectorPrediction):
            raise TypeError(f"predictions[{index}] must be a DetectorPrediction")
        x1, y1, x2, y2 = prediction.bbox_xyxy
        axis.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="red",
                linewidth=2,
                label="prediction",
            )
        )
        axis.text(
            x1,
            y1,
            f"{prediction.detector_score:.3f}",
            color="white",
            fontsize=9,
            bbox={"facecolor": "red", "alpha": 0.7, "pad": 2},
        )
    axis.set_axis_off()
    return figure


def load_inference_model(
    checkpoint_path: Path,
    label_map: Mapping[str, Any],
    *,
    device: torch.device | None = None,
) -> tuple[nn.Module, torch.device]:
    """체크포인트의 학습 설정에 맞춰 탐지기를 복원한다."""

    validate_detector_label_map(label_map)
    selected_device = device if device is not None else select_device()
    checkpoint = load_checkpoint(checkpoint_path, selected_device)

    if checkpoint["label_map_hash"] != hash_label_map(label_map):
        raise ValueError("checkpoint label map hash does not match current label map")

    model_config = checkpoint["model_config"]
    num_classes = model_config.get("num_classes")
    nms_threshold = model_config.get("nms_threshold")
    pretrained_backbone = model_config.get("pretrained_backbone")

    if num_classes != label_map["num_model_classes"]:
        raise ValueError("checkpoint class count does not match current label map")
    nms_threshold = normalize_probability(
        nms_threshold,
        name="checkpoint nms_threshold",
    )
    if not isinstance(pretrained_backbone, bool):
        raise TypeError("체크포인트의 pretrained_backbone 설정이 올바르지 않습니다.")
    model = build_detector_model(
        checkpoint["model_name"],
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        nms_threshold=float(nms_threshold),
    ).to(selected_device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, selected_device


def _validate_inference_image(image: Any, *, index: int) -> None:
    name = f"batch.images[{index}]"
    validate_rgb_image_tensor(image, name=name)


def _normalize_inference_meta(value: Any, *, index: int) -> dict[str, Any]:
    name = f"batch.meta[{index}]"
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    required_fields = {"image_id", "file_name", "image_path", "original_size"}
    missing_fields = sorted(required_fields - value.keys())
    if missing_fields:
        raise ValueError(f"{name} is missing fields: {missing_fields}")

    image_id = value["image_id"]
    if not isinstance(image_id, int) or isinstance(image_id, bool) or image_id < 0:
        raise ValueError(f"{name}.image_id must be a non-negative integer")
    for field_name in ("file_name", "image_path"):
        field_value = value[field_name]
        if not isinstance(field_value, str) or not field_value:
            raise ValueError(f"{name}.{field_name} must be a non-empty string")
    original_size = value["original_size"]
    if (
        isinstance(original_size, (str, bytes))
        or not isinstance(original_size, Sequence)
        or len(original_size) != 2
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in original_size
        )
    ):
        raise ValueError(f"{name}.original_size must contain [height, width]")
    return {
        "image_id": image_id,
        "file_name": value["file_name"],
        "image_path": value["image_path"],
        "original_size": [int(item) for item in original_size],
    }


def _normalize_model_predictions(
    value: Any,
    *,
    expected_count: int,
) -> list[dict[str, torch.Tensor]]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError("model must return one prediction mapping per image")

    predictions: list[dict[str, torch.Tensor]] = []
    for index, raw_prediction in enumerate(value):
        if not isinstance(raw_prediction, Mapping):
            raise TypeError(f"predictions[{index}] must be a mapping")
        missing_fields = {"boxes", "labels", "scores"} - raw_prediction.keys()
        if missing_fields:
            raise ValueError(
                f"predictions[{index}] is missing fields: {sorted(missing_fields)}"
            )
        boxes = raw_prediction["boxes"]
        labels = raw_prediction["labels"]
        scores = raw_prediction["scores"]
        if not all(isinstance(item, torch.Tensor) for item in (boxes, labels, scores)):
            raise TypeError("prediction boxes, labels, and scores must be tensors")
        boxes = boxes.detach().to(device="cpu", dtype=torch.float32)
        labels = labels.detach().to(device="cpu")
        scores = scores.detach().to(device="cpu", dtype=torch.float32)
        object_count = boxes.shape[0] if boxes.ndim > 0 else 0
        if boxes.ndim != 2 or boxes.shape[1:] != (4,):
            raise ValueError("prediction boxes must have shape [P, 4]")
        if labels.dtype != torch.int64 or labels.shape != (object_count,):
            raise ValueError("prediction labels must use int64 with shape [P]")
        if scores.shape != (object_count,):
            raise ValueError("prediction scores must have shape [P]")
        if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
            raise ValueError("prediction boxes and scores must be finite")
        if torch.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("prediction scores must be between 0.0 and 1.0")
        predictions.append({"boxes": boxes, "labels": labels, "scores": scores})
    return predictions
