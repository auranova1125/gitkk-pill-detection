"""검증셋에서 알약 탐지·분류 통합 결과를 평가한다."""

import csv
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

from data_loading.validation import normalize_probability
from pipeline.pill_recognition import PillPrediction, PillRecognitionResult
from utils.io import write_json

HIGH_IOU_THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]


@dataclass(frozen=True, slots=True)
class PillRecognitionEvaluation:
    """검증셋의 탐지·분류 통합 성능 요약."""

    map_75: float
    map_80: float
    map_85: float
    map_90: float
    map_95: float
    map_75_95: float
    mean_average_recall: float
    detection_precision: float
    detection_recall: float
    matched_classification_accuracy: float
    end_to_end_precision: float
    end_to_end_recall: float
    end_to_end_f1: float
    num_images: int
    num_ground_truths: int
    num_predictions: int
    num_detection_matches: int
    num_correct_predictions: int
    num_misclassified: int
    num_false_positives: int
    num_false_negatives: int
    matching_iou_threshold: float

    def to_dict(self) -> dict[str, int | float]:
        """JSON으로 저장할 수 있는 dict로 변환한다."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PillRecognitionEvaluationDetail:
    """예측과 정답을 매칭한 이미지별 오류 분석 행."""

    image_id: str
    status: str
    detection_id: str | None
    predicted_category_id: int | None
    target_category_id: int | None
    predicted_bbox_xywh: tuple[float, float, float, float] | None
    target_bbox_xyxy: tuple[float, float, float, float] | None
    iou: float | None
    detector_score: float | None
    classifier_score: float | None
    final_score: float | None

    def to_dict(self) -> dict[str, Any]:
        """CSV로 저장할 수 있는 dict로 변환한다."""

        return asdict(self)


def evaluate_pill_recognition(
    result: PillRecognitionResult,
    validation_loader: Iterable[Mapping[str, Any]],
    *,
    matching_iou_threshold: float = 0.5,
) -> tuple[
    PillRecognitionEvaluation,
    tuple[PillRecognitionEvaluationDetail, ...],
]:
    """검증 정답과 통합 예측을 매칭해 성능과 오류 상세를 반환한다."""

    matching_iou_threshold = normalize_probability(
        matching_iou_threshold,
        name="matching_iou_threshold",
    )
    targets_by_image = _collect_validation_targets(validation_loader)
    predictions_by_image = _group_predictions(result.predictions)
    unknown_image_ids = predictions_by_image.keys() - targets_by_image.keys()
    if unknown_image_ids:
        unknown = sorted(unknown_image_ids)[0]
        raise ValueError(
            f"prediction image_id is absent from validation data: {unknown}"
        )

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=HIGH_IOU_THRESHOLDS,
        extended_summary=True,
    )
    details: list[PillRecognitionEvaluationDetail] = []
    detection_matches = 0
    correct_predictions = 0
    misclassified = 0
    false_positives = 0
    false_negatives = 0

    for image_id, target in targets_by_image.items():
        predictions = predictions_by_image.get(image_id, [])
        metric.update(
            [_prediction_tensors(predictions)],
            [target],
        )
        image_details = _match_image_predictions(
            image_id,
            predictions,
            target,
            iou_threshold=matching_iou_threshold,
        )
        details.extend(image_details)
        statuses = [detail.status for detail in image_details]
        detection_matches += statuses.count("correct") + statuses.count("misclassified")
        correct_predictions += statuses.count("correct")
        misclassified += statuses.count("misclassified")
        false_positives += statuses.count("false_positive")
        false_negatives += statuses.count("false_negative")

    num_images = len(targets_by_image)
    num_ground_truths = sum(
        int(target["boxes"].shape[0]) for target in targets_by_image.values()
    )
    num_predictions = len(result.predictions)
    if num_images == 0:
        raise ValueError("validation_loader must contain at least one image")
    if num_ground_truths == 0:
        raise ValueError("validation data must contain at least one ground-truth box")

    metric_values = metric.compute()
    precision = metric_values["precision"]
    maps = {}
    for i, threshold in enumerate(HIGH_IOU_THRESHOLDS):
        p = precision[i, :, :, 0, -1]
        valid = p[p >= 0]
        maps[threshold] = valid.mean().item() if valid.numel() else -1.0

    detection_precision = _safe_divide(detection_matches, num_predictions)
    detection_recall = detection_matches / num_ground_truths
    end_to_end_precision = _safe_divide(correct_predictions, num_predictions)
    end_to_end_recall = correct_predictions / num_ground_truths
    evaluation = PillRecognitionEvaluation(
        map_75=maps[0.75],
        map_80=maps[0.80],
        map_85=maps[0.85],
        map_90=maps[0.90],
        map_95=maps[0.95],
        map_75_95=_metric_value(metric_values, "map"),
        mean_average_recall=_metric_value(metric_values, "mar_100"),
        detection_precision=detection_precision,
        detection_recall=detection_recall,
        matched_classification_accuracy=_safe_divide(
            correct_predictions,
            detection_matches,
        ),
        end_to_end_precision=end_to_end_precision,
        end_to_end_recall=end_to_end_recall,
        end_to_end_f1=_f1(end_to_end_precision, end_to_end_recall),
        num_images=num_images,
        num_ground_truths=num_ground_truths,
        num_predictions=num_predictions,
        num_detection_matches=detection_matches,
        num_correct_predictions=correct_predictions,
        num_misclassified=misclassified,
        num_false_positives=false_positives,
        num_false_negatives=false_negatives,
        matching_iou_threshold=float(matching_iou_threshold),
    )
    return evaluation, tuple(details)


def save_pill_recognition_evaluation(
    evaluation: PillRecognitionEvaluation,
    details: Iterable[PillRecognitionEvaluationDetail],
    output_dir: Path,
) -> tuple[Path, Path]:
    """평가 요약 JSON과 오류 분석 CSV를 저장한다."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "validation_metrics.json"
    details_path = output_dir / "validation_details.csv"
    write_json(metrics_path, evaluation.to_dict())
    field_names = list(PillRecognitionEvaluationDetail.__dataclass_fields__)
    with details_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(detail.to_dict() for detail in details)
    return metrics_path, details_path


def _collect_validation_targets(
    validation_loader: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, torch.Tensor]]:
    targets_by_image: dict[str, dict[str, torch.Tensor]] = {}
    for batch_index, batch in enumerate(validation_loader):
        metadata = batch.get("meta")
        if not isinstance(metadata, list):
            raise TypeError(f"validation batch {batch_index} meta must be a list")
        for item_index, meta in enumerate(metadata):
            if not isinstance(meta, Mapping):
                raise TypeError("validation metadata items must be mappings")
            image_id = str(meta.get("image_id"))
            if image_id in targets_by_image:
                raise ValueError(f"duplicate validation image_id: {image_id}")
            boxes = torch.as_tensor(
                meta.get("transformed_boxes_xyxy"),
                dtype=torch.float32,
            ).reshape(-1, 4)
            labels = torch.as_tensor(
                meta.get("category_ids"),
                dtype=torch.int64,
            ).reshape(-1)
            if boxes.shape[0] != labels.shape[0]:
                raise ValueError(
                    "validation boxes and category_ids must have equal lengths: "
                    f"batch {batch_index}, item {item_index}"
                )
            targets_by_image[image_id] = {"boxes": boxes, "labels": labels}
    return targets_by_image


def _group_predictions(
    predictions: Iterable[PillPrediction],
) -> dict[str, list[PillPrediction]]:
    grouped: dict[str, list[PillPrediction]] = {}
    for prediction in predictions:
        grouped.setdefault(str(prediction.image_id), []).append(prediction)
    return grouped


def _prediction_tensors(
    predictions: list[PillPrediction],
) -> dict[str, torch.Tensor]:
    boxes = [_xywh_to_xyxy(prediction.bbox_xywh) for prediction in predictions]
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(
            [prediction.category_id for prediction in predictions],
            dtype=torch.int64,
        ),
        "scores": torch.tensor(
            [prediction.final_score for prediction in predictions],
            dtype=torch.float32,
        ),
    }


def _match_image_predictions(
    image_id: str,
    predictions: list[PillPrediction],
    target: Mapping[str, torch.Tensor],
    *,
    iou_threshold: float,
) -> list[PillRecognitionEvaluationDetail]:
    ordered_predictions = sorted(
        predictions,
        key=lambda prediction: prediction.final_score,
        reverse=True,
    )
    target_boxes = target["boxes"]
    target_labels = target["labels"]
    matched_target_indices: set[int] = set()
    details: list[PillRecognitionEvaluationDetail] = []

    for prediction in ordered_predictions:
        best_index: int | None = None
        best_iou = 0.0
        prediction_box = torch.tensor(
            [_xywh_to_xyxy(prediction.bbox_xywh)],
            dtype=torch.float32,
        )
        if target_boxes.numel():
            overlaps = box_iou(prediction_box, target_boxes)[0]
            for target_index, overlap in enumerate(overlaps.tolist()):
                if target_index not in matched_target_indices and overlap > best_iou:
                    best_index = target_index
                    best_iou = overlap

        if best_index is None or best_iou < iou_threshold:
            details.append(_prediction_detail(image_id, "false_positive", prediction))
            continue

        matched_target_indices.add(best_index)
        target_category_id = int(target_labels[best_index])
        status = (
            "correct"
            if prediction.category_id == target_category_id
            else "misclassified"
        )
        details.append(
            _prediction_detail(
                image_id,
                status,
                prediction,
                target_category_id=target_category_id,
                target_bbox_xyxy=tuple(
                    float(value) for value in target_boxes[best_index].tolist()
                ),
                iou=best_iou,
            )
        )

    for target_index, target_category_id in enumerate(target_labels.tolist()):
        if target_index not in matched_target_indices:
            details.append(
                PillRecognitionEvaluationDetail(
                    image_id=image_id,
                    status="false_negative",
                    detection_id=None,
                    predicted_category_id=None,
                    target_category_id=int(target_category_id),
                    predicted_bbox_xywh=None,
                    target_bbox_xyxy=tuple(
                        float(value) for value in target_boxes[target_index].tolist()
                    ),
                    iou=None,
                    detector_score=None,
                    classifier_score=None,
                    final_score=None,
                )
            )
    return details


def _prediction_detail(
    image_id: str,
    status: str,
    prediction: PillPrediction,
    *,
    target_category_id: int | None = None,
    target_bbox_xyxy: tuple[float, float, float, float] | None = None,
    iou: float | None = None,
) -> PillRecognitionEvaluationDetail:
    return PillRecognitionEvaluationDetail(
        image_id=image_id,
        status=status,
        detection_id=prediction.detection_id,
        predicted_category_id=prediction.category_id,
        target_category_id=target_category_id,
        predicted_bbox_xywh=prediction.bbox_xywh,
        target_bbox_xyxy=target_bbox_xyxy,
        iou=iou,
        detector_score=prediction.detector_score,
        classifier_score=prediction.classifier_score,
        final_score=prediction.final_score,
    )


def _xywh_to_xyxy(
    box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x, y, width, height = box
    return x, y, x + width, y + height


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _metric_value(values: Mapping[str, torch.Tensor], name: str) -> float:
    value = values.get(name)
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError(f"metric output is missing scalar {name}")
    return float(value.item())
