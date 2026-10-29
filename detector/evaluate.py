"""탐지기 validation 평가와 지표 계산."""

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from tqdm.auto import tqdm

from configs.detector import DetectorConfig
from detector.train import move_detection_batch_to_device
from pilldata.validation import normalize_probability
from utils.io import write_json

HIGH_IOU_THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """탐지기 validation 지표와 예측 개수 분석 결과."""

    map: float
    map_50: float
    map_75: float
    map_75_95: float
    mean_average_recall: float
    recall_before_threshold: float
    recall_after_threshold: float
    mean_count_error_before_threshold: float
    mean_count_error_after_threshold: float
    num_images: int
    num_ground_truths: int
    num_predictions_before_threshold: int
    num_predictions_after_threshold: int
    score_threshold: float
    recall_iou_threshold: float

    def to_dict(self) -> dict[str, int | float]:
        """직렬화 가능한 dict로 변환한다."""

        return asdict(self)


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    *,
    score_threshold: float,
    recall_iou_threshold: float = 0.5,
    show_progress: bool = False,
    progress_description: str = "Detector validation",
) -> EvaluationResult:
    """validation 전체의 COCO 지표와 threshold 전후 recall을 계산한다."""

    score_threshold = normalize_probability(score_threshold, name="score_threshold")
    recall_iou_threshold = normalize_probability(
        recall_iou_threshold,
        name="recall_iou_threshold",
    )
    if not isinstance(show_progress, bool):
        raise TypeError("show_progress must be a boolean")
    if not isinstance(progress_description, str) or not progress_description.strip():
        raise ValueError("progress_description must be a non-empty string")
    model.eval()

    coco_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    high_iou_metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=HIGH_IOU_THRESHOLDS,
    )

    num_images = 0
    num_ground_truths = 0
    num_predictions_before = 0
    num_predictions_after = 0
    matches_before = 0
    matches_after = 0
    count_error_before = 0
    count_error_after = 0

    progress = tqdm(
        data_loader,
        desc=progress_description,
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for batch in progress:
        images, device_targets = move_detection_batch_to_device(batch, device)
        raw_predictions = model(images)
        predictions = _prepare_predictions(raw_predictions, expected_count=len(images))
        targets = [_prepare_target(target) for target in device_targets]
        filtered_predictions = filter_predictions_by_score(
            predictions,
            score_threshold=score_threshold,
        )

        coco_metric.update(predictions, targets)
        high_iou_metric.update(predictions, targets)

        for prediction, filtered_prediction, target in zip(
            predictions,
            filtered_predictions,
            targets,
            strict=True,
        ):
            ground_truth_count = target["boxes"].shape[0]
            prediction_count = prediction["boxes"].shape[0]
            filtered_count = filtered_prediction["boxes"].shape[0]

            num_images += 1
            num_ground_truths += ground_truth_count
            num_predictions_before += prediction_count
            num_predictions_after += filtered_count
            count_error_before += abs(prediction_count - ground_truth_count)
            count_error_after += abs(filtered_count - ground_truth_count)
            matches_before += _count_matches(
                prediction,
                target,
                iou_threshold=recall_iou_threshold,
            )
            matches_after += _count_matches(
                filtered_prediction,
                target,
                iou_threshold=recall_iou_threshold,
            )
        progress.set_postfix(images=num_images, predictions=num_predictions_after)

    if num_images == 0:
        raise ValueError("data_loader must contain at least one batch")
    if num_ground_truths == 0:
        raise ValueError("validation data must contain at least one ground-truth box")

    coco_values = coco_metric.compute()
    high_iou_values = high_iou_metric.compute()
    return EvaluationResult(
        map=_metric_value(coco_values, "map"),
        map_50=_metric_value(coco_values, "map_50"),
        map_75=_metric_value(coco_values, "map_75"),
        map_75_95=_metric_value(high_iou_values, "map"),
        mean_average_recall=_metric_value(coco_values, "mar_100"),
        recall_before_threshold=matches_before / num_ground_truths,
        recall_after_threshold=matches_after / num_ground_truths,
        mean_count_error_before_threshold=count_error_before / num_images,
        mean_count_error_after_threshold=count_error_after / num_images,
        num_images=num_images,
        num_ground_truths=num_ground_truths,
        num_predictions_before_threshold=num_predictions_before,
        num_predictions_after_threshold=num_predictions_after,
        score_threshold=float(score_threshold),
        recall_iou_threshold=float(recall_iou_threshold),
    )


def filter_predictions_by_score(
    predictions: list[dict[str, torch.Tensor]],
    *,
    score_threshold: float,
) -> list[dict[str, torch.Tensor]]:
    """각 이미지 예측에서 score threshold 이상인 항목만 유지한다."""

    score_threshold = normalize_probability(score_threshold, name="score_threshold")
    filtered_predictions: list[dict[str, torch.Tensor]] = []
    for index, prediction in enumerate(predictions):
        _validate_prediction(prediction, index=index)
        scores = prediction["scores"]
        keep = scores >= score_threshold
        filtered_predictions.append(
            {
                name: value[keep]
                if value.ndim > 0 and value.shape[0] == scores.shape[0]
                else value
                for name, value in prediction.items()
            }
        )
    return filtered_predictions


def _prepare_predictions(
    value: Any,
    *,
    expected_count: int,
) -> list[dict[str, torch.Tensor]]:
    if not isinstance(value, list):
        raise TypeError("model must return a list of prediction mappings")
    if len(value) != expected_count:
        raise ValueError("model prediction count must match image count")

    predictions: list[dict[str, torch.Tensor]] = []
    for index, raw_prediction in enumerate(value):
        if not isinstance(raw_prediction, Mapping):
            raise TypeError(f"predictions[{index}] must be a mapping")
        prediction = {
            name: tensor.detach().cpu() for name, tensor in raw_prediction.items()
        }
        _validate_prediction(prediction, index=index)
        predictions.append(prediction)
    return predictions


def _prepare_target(target: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    metric_target: dict[str, torch.Tensor] = {}
    for name in ("boxes", "labels", "area", "iscrowd"):
        value = target.get(name)
        if isinstance(value, torch.Tensor):
            metric_target[name] = value.detach().cpu()
    if "boxes" not in metric_target or "labels" not in metric_target:
        raise ValueError("targets must contain boxes and labels tensors")
    return metric_target


def _validate_prediction(prediction: Mapping[str, Any], *, index: int) -> None:
    missing_fields = {"boxes", "labels", "scores"} - prediction.keys()
    if missing_fields:
        raise ValueError(
            f"predictions[{index}] is missing required fields: {sorted(missing_fields)}"
        )
    boxes = prediction["boxes"]
    labels = prediction["labels"]
    scores = prediction["scores"]
    if not all(isinstance(value, torch.Tensor) for value in (boxes, labels, scores)):
        raise TypeError("prediction boxes, labels, and scores must be tensors")
    if boxes.dtype != torch.float32 or boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError("prediction boxes must use float32 with shape [K, 4]")
    object_count = boxes.shape[0]
    if labels.dtype != torch.int64 or labels.shape != (object_count,):
        raise ValueError("prediction labels must use int64 with shape [K]")
    if scores.dtype != torch.float32 or scores.shape != (object_count,):
        raise ValueError("prediction scores must use float32 with shape [K]")
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError("prediction boxes and scores must contain only finite values")
    if torch.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("prediction scores must be between 0.0 and 1.0")


def _count_matches(
    prediction: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    *,
    iou_threshold: float,
) -> int:
    prediction_boxes = prediction["boxes"]
    prediction_labels = prediction["labels"]
    scores = prediction["scores"]
    target_boxes = target["boxes"]
    target_labels = target["labels"]
    if prediction_boxes.shape[0] == 0 or target_boxes.shape[0] == 0:
        return 0

    matched_targets = torch.zeros(target_boxes.shape[0], dtype=torch.bool)
    matched_count = 0
    for prediction_index in scores.argsort(descending=True).tolist():
        candidate_mask = (~matched_targets) & (
            target_labels == prediction_labels[prediction_index]
        )
        candidate_indices = candidate_mask.nonzero(as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            continue
        candidate_ious = box_iou(
            prediction_boxes[prediction_index].unsqueeze(0),
            target_boxes[candidate_indices],
        ).squeeze(0)
        best_position = int(candidate_ious.argmax().item())
        if candidate_ious[best_position] < iou_threshold:
            continue
        matched_targets[candidate_indices[best_position]] = True
        matched_count += 1
    return matched_count


def _metric_value(values: Mapping[str, torch.Tensor], name: str) -> float:
    value = float(values[name].item())
    if value < 0:
        raise ValueError(f"metric {name} is undefined for the provided validation data")
    return value


def evaluate_detector_checkpoint(
    config: DetectorConfig,
    checkpoint_path: Path,
    *,
    output_dir: Path | None = None,
) -> EvaluationResult:
    """설정의 validation 데이터로 탐지기 checkpoint를 평가하고 저장한다."""

    from detector.inference import load_inference_model
    from detector.train import prepare_detector_data

    data = prepare_detector_data(config)
    model, device = load_inference_model(
        checkpoint_path,
        data.label_map,
        device=data.device,
    )
    result = evaluate_model(
        model,
        data.val_loader,
        device,
        score_threshold=config.score_threshold,
        show_progress=config.show_progress,
        progress_description="Detector evaluation",
    )
    destination = (
        Path(output_dir) if output_dir is not None else Path(checkpoint_path).parent
    )
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "evaluation_metrics.json", result.to_dict())
    return result
