"""단일 클래스 알약 탐지와 품목 분류를 연결한 pill recognition 추론."""

import csv
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm

from classifier.inference import (
    ClassifierInferenceError,
    classify_detections,
    load_classifier_inference_model,
)
from classifier.input_validation import validate_classifier_class_map
from configs.pill_recognition import PillRecognitionConfig
from data_loading.factory import first_batch, load_data_factory
from data_loading.image_transforms import (
    CLASSIFIER_IMAGE_SIZE,
    clip_xyxy_to_image,
    crop_tensor_xyxy,
    preprocess_classifier_crop,
)
from data_loading.validation import (
    normalize_bbox_xywh,
    normalize_probability,
    validate_rgb_image_tensor,
)
from detector.inference import (
    DetectorInferenceError,
    DetectorInferenceResult,
    DetectorPrediction,
    load_detector_inference_model,
    predict_detector,
    validate_inference_batch,
)
from detector.input_validation import validate_detector_label_map
from detector.yolo_inference import (
    load_yolo_detector_inference_model,
    predict_yolo_detector,
)
from utils.io import read_json, write_json
from utils.plotting import resolve_korean_font
from utils.runtime import resolve_device

SUBMISSION_FIELD_NAMES = (
    "annotation_id",
    "image_id",
    "category_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "score",
)
DetectorPredictor = Callable[..., DetectorInferenceResult]


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


@dataclass(frozen=True, slots=True)
class PillRecognitionData:
    """검증을 마쳐 미리보기와 통합 추론에 재사용할 입력."""

    data_loader: Iterable[Mapping[str, Any]]
    preview_batch: Mapping[str, Any]
    device: torch.device


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
    detector_model: Any,
    classifier_model: nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    class_map: Mapping[str, Any],
    device: torch.device,
    *,
    score_threshold: float,
    detector_name: str,
    show_progress: bool = True,
    detector_predictor: DetectorPredictor = predict_detector,
) -> PillRecognitionResult:
    """이미지 batch를 탐지하고 crop을 분류해 최종 예측을 반환한다."""

    validate_classifier_class_map(class_map)
    if not isinstance(show_progress, bool):
        raise TypeError("show_progress must be a boolean")
    score_threshold = normalize_probability(score_threshold, name="score_threshold")
    if not isinstance(detector_name, str) or not detector_name.strip():
        raise ValueError("detector_name must be a non-empty string")
    if not callable(detector_predictor):
        raise TypeError("detector_predictor must be callable")

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

            detector_result = detector_predictor(
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

    classifier_result = classify_detections(
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


def load_configured_detector(
    config: PillRecognitionConfig,
    label_map: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[Any, torch.device, DetectorPredictor]:
    """설정한 backend의 탐지 모델과 공통 예측 함수를 반환한다."""

    if config.detector_backend == "ultralytics":
        model, selected_device = load_yolo_detector_inference_model(
            config.detector_checkpoint_path,
            label_map,
            device=device,
        )
        predictor = partial(
            predict_yolo_detector,
            image_size=config.detector_image_size,
            nms_threshold=config.detector_nms_threshold,
        )
        return model, selected_device, predictor

    model, selected_device = load_detector_inference_model(
        config.detector_checkpoint_path,
        label_map,
        device=device,
    )
    return model, selected_device, predict_detector


def prepare_pill_recognition_data(
    config: PillRecognitionConfig,
) -> PillRecognitionData:
    """검증용 factory 결과를 소비한 뒤 새 추론 입력을 반환한다."""

    device = resolve_device(config.device)
    factory = load_data_factory(
        config.inference_factory_path,
        setting_name="inference_factory_path",
    )
    validation_loader = factory(config=config, batch_size=config.batch_size)
    preview_batch = first_batch(validation_loader, name="inference_loader")
    validate_inference_batch(preview_batch)
    data_loader = factory(config=config, batch_size=config.batch_size)
    return PillRecognitionData(
        data_loader=data_loader,
        preview_batch=preview_batch,
        device=device,
    )


def recognize_pills_from_config(
    config: PillRecognitionConfig,
    *,
    data: PillRecognitionData | None = None,
) -> PillRecognitionResult:
    """검증된 입력과 두 checkpoint로 알약 인식을 실행한다."""

    prepared_data = data or prepare_pill_recognition_data(config)
    run_dir = _prepare_pill_recognition_run(config)
    device = prepared_data.device
    detector_label_map = read_json(config.detector_label_map_path)
    class_map = read_json(config.classifier_class_map_path)
    validate_detector_label_map(detector_label_map)
    if not isinstance(class_map, Mapping):
        raise TypeError("class map JSON root must be a mapping")
    validate_classifier_class_map(class_map)

    detector_model, device, detector_predictor = load_configured_detector(
        config,
        detector_label_map,
        device=device,
    )
    classifier_model, device = load_classifier_inference_model(
        config.classifier_checkpoint_path,
        class_map,
        device=device,
    )
    result = recognize_pills(
        detector_model,
        classifier_model,
        prepared_data.data_loader,
        class_map,
        device,
        score_threshold=config.score_threshold,
        detector_name=config.detector_name,
        show_progress=config.show_progress,
        detector_predictor=detector_predictor,
    )
    save_pill_recognition_artifacts(result, run_dir)
    save_submission_csv(result, run_dir)
    return result


def _prepare_pill_recognition_run(config: PillRecognitionConfig) -> Path:
    """기존 결과 덮어쓰기를 막고 실제 추론 설정을 실행 폴더에 기록한다."""

    run_dir = config.run_dir
    artifact_names = (
        "pill_predictions.json",
        "pill_recognition_errors.json",
        "submission.csv",
    )
    existing = [run_dir / name for name in artifact_names if (run_dir / name).exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"run directory already contains inference artifacts: {names}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "resolved_config.json", config.to_dict(), sort_keys=True)
    return run_dir


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


def save_submission_csv(
    result: PillRecognitionResult,
    output_dir: Path,
) -> Path:
    """통합 예측을 Kaggle 제출 형식의 CSV로 저장한다."""

    if not isinstance(result, PillRecognitionResult):
        raise TypeError("result must be a PillRecognitionResult")

    normalized_predictions: list[
        tuple[int, PillPrediction, tuple[float, float, float, float]]
    ] = []
    detection_ids: set[str] = set()
    for index, prediction in enumerate(result.predictions):
        if not isinstance(prediction.image_id, str):
            raise TypeError(f"predictions[{index}].image_id must be a string")
        if not prediction.image_id.isascii() or not prediction.image_id.isdigit():
            raise ValueError(
                f"predictions[{index}].image_id must be an ASCII integer string"
            )
        if prediction.detection_id in detection_ids:
            raise ValueError(f"duplicate detection_id: {prediction.detection_id}")
        detection_ids.add(prediction.detection_id)
        if (
            not isinstance(prediction.category_id, int)
            or isinstance(prediction.category_id, bool)
            or prediction.category_id <= 0
        ):
            raise ValueError(
                f"predictions[{index}].category_id must be a positive integer"
            )
        bbox_xywh = normalize_bbox_xywh(
            prediction.bbox_xywh,
            name=f"predictions[{index}].bbox_xywh",
        )
        normalize_probability(
            prediction.final_score,
            name=f"predictions[{index}].final_score",
        )
        normalized_predictions.append((int(prediction.image_id), prediction, bbox_xywh))

    normalized_predictions.sort(
        key=lambda item: (item[0], item[1].detection_id),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission.csv"
    with submission_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUBMISSION_FIELD_NAMES)
        writer.writeheader()
        for annotation_id, (image_id, prediction, bbox_xywh) in enumerate(
            normalized_predictions,
            start=1,
        ):
            x, y, width, height = bbox_xywh
            writer.writerow(
                {
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "category_id": prediction.category_id,
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": width,
                    "bbox_h": height,
                    "score": prediction.final_score,
                }
            )
    return submission_path


def plot_pill_predictions(
    image: torch.Tensor,
    predictions: Sequence[PillPrediction],
    *,
    title: str | None = None,
) -> Any:
    """원본 이미지에 최종 품목명, bbox와 결합 점수를 표시한다."""

    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    validate_rgb_image_tensor(image)
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise TypeError("predictions must be a sequence")

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.imshow(image.permute(1, 2, 0).numpy())
    font_properties = resolve_korean_font()
    if title:
        axis.set_title(title, fontproperties=font_properties)

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
            fontproperties=font_properties,
            bbox={"facecolor": "red", "alpha": 0.75, "pad": 2},
        )
    axis.set_axis_off()
    return figure
