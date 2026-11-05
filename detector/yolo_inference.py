"""Ultralytics YOLO 출력을 공통 detector 추론 형식으로 변환한다."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from data_loading.image_transforms import clip_xyxy_to_image
from data_loading.validation import normalize_probability
from detector.inference import (
    DetectorInferenceError,
    DetectorInferenceResult,
    DetectorPrediction,
    validate_inference_batch,
)
from detector.input_validation import validate_detector_label_map
from utils.runtime import select_device


def load_yolo_detector_inference_model(
    checkpoint_path: Path,
    label_map: Mapping[str, Any],
    *,
    device: torch.device | None = None,
    model_factory: Any | None = None,
) -> tuple[Any, torch.device]:
    """단일 pill 클래스 YOLO 체크포인트를 추론용으로 불러온다."""

    validate_detector_label_map(label_map)
    if label_map["num_foreground_classes"] != 1:
        raise ValueError("YOLO pill detector must have exactly one foreground class")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"YOLO checkpoint does not exist: {checkpoint}")
    selected_device = device if device is not None else select_device()
    if model_factory is None:
        from ultralytics import YOLO

        model_factory = YOLO
    model = model_factory(str(checkpoint))
    names = getattr(model, "names", None)
    if isinstance(names, Mapping):
        valid_single_class = set(names) == {0}
    else:
        valid_single_class = isinstance(names, list) and len(names) == 1
    if not valid_single_class:
        raise ValueError("YOLO checkpoint must contain exactly one class at index 0")
    return model, selected_device


@torch.inference_mode()
def predict_yolo_detector(
    model: Any,
    data_loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    *,
    score_threshold: float,
    detector_name: str,
    image_size: int,
    nms_threshold: float,
) -> DetectorInferenceResult:
    """YOLO batch 추론 결과를 ``DetectorPrediction`` 목록으로 변환한다."""

    score_threshold = normalize_probability(score_threshold, name="score_threshold")
    nms_threshold = normalize_probability(nms_threshold, name="nms_threshold")
    if not isinstance(image_size, int) or isinstance(image_size, bool):
        raise TypeError("image_size must be an integer")
    if image_size <= 0:
        raise ValueError("image_size must be greater than zero")
    if not isinstance(detector_name, str) or not detector_name.strip():
        raise ValueError("detector_name must be a non-empty string")

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

        sources = [_tensor_to_pil_image(image) for image in images]
        raw_results = model.predict(
            source=sources,
            conf=score_threshold,
            iou=nms_threshold,
            imgsz=image_size,
            device=str(device),
            verbose=False,
        )
        results = list(raw_results)
        if len(results) != len(images):
            raise ValueError("YOLO must return one result per input image")

        for image, meta, result in zip(images, metadata, results, strict=True):
            height, width = image.shape[-2:]
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                raise TypeError("YOLO result must contain boxes")
            xyxy = torch.as_tensor(boxes.xyxy).detach().cpu().reshape(-1, 4)
            scores = torch.as_tensor(boxes.conf).detach().cpu().reshape(-1)
            classes = torch.as_tensor(boxes.cls).detach().cpu().reshape(-1)
            if not (xyxy.shape[0] == scores.shape[0] == classes.shape[0]):
                raise ValueError(
                    "YOLO boxes, scores and classes must have equal length"
                )

            kept_index = 0
            for prediction_index in range(xyxy.shape[0]):
                class_index = int(classes[prediction_index].item())
                if class_index != 0:
                    errors.append(
                        DetectorInferenceError(
                            image_id=meta["image_id"],
                            prediction_index=prediction_index,
                            message=f"expected YOLO pill class 0, got {class_index}",
                        )
                    )
                    continue
                clipped_box = clip_xyxy_to_image(
                    xyxy[prediction_index].tolist(),
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
                        detector_score=float(scores[prediction_index].item()),
                        detector_name=detector_name,
                    )
                )

    return DetectorInferenceResult(
        predictions=tuple(predictions),
        errors=tuple(errors),
    )


def _tensor_to_pil_image(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .mul(255)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")
