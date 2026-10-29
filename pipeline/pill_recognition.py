"""단일 클래스 알약 탐지와 품목 분류를 연결한 pill recognition 추론."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm

from classifier.inference import (
    InferenceError as ClassifierInferenceError,
)
from classifier.inference import (
    load_inference_model as load_classifier_model,
)
from classifier.inference import (
    predict_detections,
)
from classifier.input_validation import validate_class_map
from configs.pill_recognition import PillRecognitionConfig
from detector.inference import (
    DetectorInferenceError,
    DetectorPrediction,
    predict_detector,
    validate_inference_batch,
)
from detector.inference import (
    load_inference_model as load_detector_model,
)
from detector.input_validation import validate_detector_label_map
from pilldata.factory import load_data_factory
from pilldata.transforms import (
    CLASSIFIER_IMAGE_SIZE,
    clip_xyxy_to_image,
    crop_tensor_xyxy,
    preprocess_classifier_crop,
)
from pilldata.validation import normalize_probability, validate_rgb_image_tensor
from utils.io import read_json, write_json
from utils.runtime import resolve_device


@dataclass(frozen=True, slots=True)
class PillPrediction:
    """탐지 bbox와 품목 분류를 결합한 최종 예측."""

    image_id: str
    detection_id: str
    bbox_xywh: tuple[float, float, float, float]
    detector_score: float
    detector_name: str
    class_index: int
    category_id: int
    item_seq: str
    class_name: str
    classifier_score: float
    final_score: float

    def to_dict(self) -> dict[str, Any]:
        """JSON으로 저장할 수 있는 dict로 변환한다."""

        values = asdict(self)
        values["bbox_xywh"] = list(self.bbox_xywh)
        return values


@dataclass(frozen=True, slots=True)
class PillRecognitionResult:
    """최종 예측과 탐지·분류 단계별 오류 목록."""

    predictions: tuple[PillPrediction, ...]
    detector_errors: tuple[DetectorInferenceError, ...]
    classifier_errors: tuple[ClassifierInferenceError, ...]

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """JSON으로 저장할 수 있는 dict로 변환한다."""

        return {
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "detector_errors": [error.to_dict() for error in self.detector_errors],
            "classifier_errors": [error.to_dict() for error in self.classifier_errors],
        }


def build_classifier_batch(
    image: torch.Tensor,
    detections: Sequence[DetectorPrediction],
) -> dict[str, Any]:
    """원본 이미지와 탐지 목록을 분류기 입력 batch로 변환한다."""

    validate_rgb_image_tensor(image)
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
        detector_score = normalize_probability(
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


def recognize_pills(
    detector_model: nn.Module,
    classifier_model: nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    class_map: Mapping[str, Any],
    device: torch.device,
    *,
    score_threshold: float,
    detector_name: str,
    show_progress: bool = True,
) -> PillRecognitionResult:
    """이미지 batch를 탐지하고 crop을 분류해 최종 예측을 반환한다."""

    num_classes = class_map.get("num_classes")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool):
        raise TypeError("class_map.num_classes must be an integer")
    validate_class_map(class_map, expected_num_classes=num_classes)
    if not isinstance(show_progress, bool):
        raise TypeError("show_progress must be a boolean")
    score_threshold = normalize_probability(score_threshold, name="score_threshold")
    if not isinstance(detector_name, str) or not detector_name.strip():
        raise ValueError("detector_name must be a non-empty string")

    class_lookup = {entry["class_index"]: entry for entry in class_map["classes"]}
    detector_predictions: list[DetectorPrediction] = []
    detector_errors: list[DetectorInferenceError] = []
    seen_image_ids: set[int] = set()

    progress = tqdm(
        data_loader,
        desc="Pill recognition inference",
        disable=not show_progress,
        dynamic_ncols=True,
    )

    def classifier_batches() -> Iterable[dict[str, Any]]:
        for batch in progress:
            images, metadata = validate_inference_batch(batch)
            for meta in metadata:
                image_id = meta["image_id"]
                if image_id in seen_image_ids:
                    raise ValueError(
                        f"duplicate image_id in inference data: {image_id}"
                    )
                seen_image_ids.add(image_id)

            detector_result = predict_detector(
                detector_model,
                [batch],
                device,
                score_threshold=score_threshold,
                detector_name=detector_name,
            )
            detector_predictions.extend(detector_result.predictions)
            detector_errors.extend(detector_result.errors)

            predictions_by_image: dict[int, list[DetectorPrediction]] = {
                meta["image_id"]: [] for meta in metadata
            }
            for prediction in detector_result.predictions:
                predictions_by_image[prediction.image_id].append(prediction)

            for image, meta in zip(images, metadata, strict=True):
                classifier_batch = build_classifier_batch(
                    image,
                    predictions_by_image[meta["image_id"]],
                )
                if classifier_batch["meta"]:
                    yield classifier_batch

            progress.set_postfix(
                images=len(seen_image_ids),
                detections=len(detector_predictions),
            )

    classifier_result = predict_detections(
        classifier_model,
        classifier_batches(),
        class_map,
        device,
    )

    predictions: list[PillPrediction] = []
    for prediction in classifier_result.predictions:
        class_entry = class_lookup[prediction.class_index]
        if prediction.category_id is None or prediction.item_seq is None:
            raise ValueError("validated class map produced an incomplete prediction")
        predictions.append(
            PillPrediction(
                image_id=prediction.image_id,
                detection_id=prediction.detection_id,
                bbox_xywh=prediction.bbox_xywh,
                detector_score=prediction.detector_score,
                detector_name=prediction.detector_name,
                class_index=prediction.class_index,
                category_id=prediction.category_id,
                item_seq=prediction.item_seq,
                class_name=class_entry["name"],
                classifier_score=prediction.classifier_score,
                final_score=prediction.final_score,
            )
        )

    return PillRecognitionResult(
        predictions=tuple(predictions),
        detector_errors=tuple(detector_errors),
        classifier_errors=classifier_result.errors,
    )


def recognize_pills_from_config(
    config: PillRecognitionConfig,
) -> PillRecognitionResult:
    """설정의 factory와 두 checkpoint를 불러와 알약 인식을 실행한다."""

    device = resolve_device(config.device)
    factory = load_data_factory(
        config.inference_factory_path,
        setting_name="inference_factory_path",
    )
    detector_label_map = read_json(config.detector_label_map_path)
    class_map = read_json(config.classifier_class_map_path)
    validate_detector_label_map(detector_label_map)
    if not isinstance(class_map, Mapping):
        raise TypeError("class map JSON root must be a mapping")
    num_classes = class_map.get("num_classes")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool):
        raise TypeError("class_map.num_classes must be an integer")
    validate_class_map(class_map, expected_num_classes=num_classes)

    inference_loader = factory(batch_size=config.batch_size)
    validate_inference_batch(_first_batch(inference_loader, name="inference_loader"))
    inference_loader = factory(batch_size=config.batch_size)
    detector_model, device = load_detector_model(
        config.detector_checkpoint_path,
        detector_label_map,
        device=device,
    )
    classifier_model, device = load_classifier_model(
        config.classifier_checkpoint_path,
        class_map,
        expected_num_classes=num_classes,
        device=device,
    )
    result = recognize_pills(
        detector_model,
        classifier_model,
        inference_loader,
        class_map,
        device,
        score_threshold=config.score_threshold,
        detector_name=config.detector_name,
        show_progress=config.show_progress,
    )
    save_pill_recognition_artifacts(result, config.output_dir)
    return result


def save_pill_recognition_artifacts(
    result: PillRecognitionResult,
    output_dir: Path,
) -> tuple[Path, Path]:
    """최종 예측과 단계별 오류를 JSON 파일로 저장한다."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "pill_predictions.json"
    errors_path = output_dir / "pill_recognition_errors.json"
    write_json(
        predictions_path,
        [prediction.to_dict() for prediction in result.predictions],
    )
    write_json(
        errors_path,
        {
            "detector_errors": [error.to_dict() for error in result.detector_errors],
            "classifier_errors": [
                error.to_dict() for error in result.classifier_errors
            ],
        },
    )
    return predictions_path, errors_path


def plot_pill_predictions(
    image: torch.Tensor,
    predictions: Sequence[PillPrediction],
    *,
    title: str | None = None,
) -> Any:
    """원본 이미지에 최종 품목명, bbox와 결합 점수를 표시한다."""

    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    _validate_image(image)
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise TypeError("predictions must be a sequence")

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.imshow(image.permute(1, 2, 0).numpy())
    if title:
        axis.set_title(title)

    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, PillPrediction):
            raise TypeError(f"predictions[{index}] must be an PillPrediction")
        x, y, width, height = prediction.bbox_xywh
        axis.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
        )
        axis.text(
            x,
            y,
            f"{prediction.class_name} ({prediction.final_score:.3f})",
            color="white",
            fontsize=9,
            bbox={"facecolor": "red", "alpha": 0.75, "pad": 2},
        )
    axis.set_axis_off()
    return figure


def _validate_image(image: Any) -> None:
    validate_rgb_image_tensor(image)


def _first_batch(loader: Iterable[Any], *, name: str) -> Any:
    try:
        return next(iter(loader))
    except StopIteration as error:
        raise ValueError(f"{name} must contain at least one batch") from error
