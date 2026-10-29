"""단일 클래스 알약 탐지와 품목 분류를 연결한 end-to-end 추론."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm

from classifier.inference import (
    InferenceError as ClassifierInferenceError,
)
from classifier.inference import (
    predict_detections,
)
from classifier.input_validation import validate_class_map
from detector.classifier_bridge import build_classifier_batch
from detector.inference import (
    DetectorInferenceError,
    DetectorPrediction,
    predict_detector,
    validate_inference_batch,
)
from utils.io import write_json


@dataclass(frozen=True, slots=True)
class EndToEndPrediction:
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
class EndToEndInferenceResult:
    """최종 예측과 탐지·분류 단계별 오류 목록."""

    predictions: tuple[EndToEndPrediction, ...]
    detector_errors: tuple[DetectorInferenceError, ...]
    classifier_errors: tuple[ClassifierInferenceError, ...]

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """JSON으로 저장할 수 있는 dict로 변환한다."""

        return {
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "detector_errors": [error.to_dict() for error in self.detector_errors],
            "classifier_errors": [error.to_dict() for error in self.classifier_errors],
        }


def run_end_to_end_inference(
    detector_model: nn.Module,
    classifier_model: nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    class_map: Mapping[str, Any],
    device: torch.device,
    *,
    score_threshold: float,
    detector_name: str,
    show_progress: bool = True,
) -> EndToEndInferenceResult:
    """이미지 batch를 탐지하고 crop을 분류해 최종 예측을 반환한다."""

    num_classes = class_map.get("num_classes")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool):
        raise TypeError("class_map.num_classes must be an integer")
    validate_class_map(class_map, expected_num_classes=num_classes)
    if not isinstance(show_progress, bool):
        raise TypeError("show_progress must be a boolean")
    if not isinstance(score_threshold, Real) or isinstance(score_threshold, bool):
        raise TypeError("score_threshold must be a number")
    if not isfinite(float(score_threshold)) or not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be between 0.0 and 1.0")
    if not isinstance(detector_name, str) or not detector_name.strip():
        raise ValueError("detector_name must be a non-empty string")

    class_lookup = {entry["class_index"]: entry for entry in class_map["classes"]}
    detector_predictions: list[DetectorPrediction] = []
    detector_errors: list[DetectorInferenceError] = []
    seen_image_ids: set[int] = set()

    progress = tqdm(
        data_loader,
        desc="End-to-end inference",
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

    predictions: list[EndToEndPrediction] = []
    for prediction in classifier_result.predictions:
        class_entry = class_lookup[prediction.class_index]
        if prediction.category_id is None or prediction.item_seq is None:
            raise ValueError("validated class map produced an incomplete prediction")
        predictions.append(
            EndToEndPrediction(
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

    return EndToEndInferenceResult(
        predictions=tuple(predictions),
        detector_errors=tuple(detector_errors),
        classifier_errors=classifier_result.errors,
    )


def save_end_to_end_artifacts(
    result: EndToEndInferenceResult,
    output_dir: Path,
) -> tuple[Path, Path]:
    """최종 예측과 단계별 오류를 JSON 파일로 저장한다."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "end_to_end_predictions.json"
    errors_path = output_dir / "end_to_end_errors.json"
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


def plot_end_to_end_predictions(
    image: torch.Tensor,
    predictions: Sequence[EndToEndPrediction],
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
        if not isinstance(prediction, EndToEndPrediction):
            raise TypeError(f"predictions[{index}] must be an EndToEndPrediction")
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
