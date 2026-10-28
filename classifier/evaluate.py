"""분류기 validation 평가와 지표 계산."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm

from classifier.input_validation import validate_batch, validate_class_map


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """한 클래스의 validation 지표."""

    class_index: int
    precision: float | None
    recall: float | None
    f1: float | None
    support: int

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConfusionPair:
    """정답 클래스에서 예측 클래스로 발생한 오분류 빈도."""

    true_class_index: int
    predicted_class_index: int
    count: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """전체 validation 평가 결과."""

    loss: float
    top1_accuracy: float
    top5_accuracy: float
    macro_f1: float
    validation_class_coverage: float
    num_samples: int
    per_class: tuple[ClassMetrics, ...]
    confusion_pairs: tuple[ConfusionPair, ...]

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["per_class"] = [metric.to_dict() for metric in self.per_class]
        values["confusion_pairs"] = [pair.to_dict() for pair in self.confusion_pairs]
        return values


@dataclass(frozen=True, slots=True)
class ValidationPrediction:
    """validation 샘플 하나의 정답과 예측."""

    sample_id: str
    image_id: str
    image_path: str
    bbox_xywh: tuple[float, float, float, float]
    group_id: str
    drug_dir: str | None
    target_class_index: int
    predicted_class_index: int
    target_category_id: int
    predicted_category_id: int
    target_item_seq: str
    predicted_item_seq: str
    classifier_score: float
    correct: bool

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["bbox_xywh"] = list(self.bbox_xywh)
        return values


@dataclass(frozen=True, slots=True)
class ValidationArtifacts:
    """best 모델의 validation 지표와 샘플별 결과."""

    metrics: EvaluationResult
    predictions: tuple[ValidationPrediction, ...]
    error_samples: tuple[ValidationPrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "error_samples": [sample.to_dict() for sample in self.error_samples],
        }


def _validate_metric_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
    loss: float,
    max_confusion_pairs: int,
) -> None:
    if not isinstance(num_classes, int) or isinstance(num_classes, bool):
        raise TypeError("num_classes must be an integer")
    if num_classes <= 0:
        raise ValueError("num_classes must be greater than zero")
    if logits.ndim != 2 or logits.shape[1] != num_classes:
        raise ValueError(f"logits must have shape [N, {num_classes}]")
    if logits.shape[0] == 0:
        raise ValueError("logits must contain at least one sample")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must not contain NaN or Inf")
    if targets.dtype != torch.int64:
        raise TypeError("targets must use torch.int64")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError(f"targets must have shape [{logits.shape[0]}]")
    if torch.any((targets < 0) | (targets >= num_classes)):
        raise ValueError(f"targets must be between 0 and {num_classes - 1}")
    if not isfinite(loss):
        raise ValueError("loss must be finite")
    if not isinstance(max_confusion_pairs, int) or isinstance(
        max_confusion_pairs, bool
    ):
        raise TypeError("max_confusion_pairs must be an integer")
    if max_confusion_pairs < 0:
        raise ValueError("max_confusion_pairs must be non-negative")


def compute_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
    loss: float,
    max_confusion_pairs: int = 10,
) -> EvaluationResult:
    """
    logit과 정답으로 전체 및 클래스별 분류 지표를 계산한다.
    Validation loss, Top-1 acc, Top-5 acc, Macro F1,
    Validation class coverage
    클래스별 precision, rcall, F1, support
    가장 많이 발생한 오분류 클래스 쌍
    """

    logits = logits.detach().cpu()
    targets = targets.detach().cpu()
    _validate_metric_inputs(
        logits,
        targets,
        num_classes=num_classes,
        loss=loss,
        max_confusion_pairs=max_confusion_pairs,
    )

    predictions = logits.argmax(dim=1)
    top_k = min(5, num_classes)
    top_predictions = logits.topk(k=top_k, dim=1).indices
    top1_accuracy = (predictions == targets).float().mean().item()
    top5_accuracy = (
        (top_predictions == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    )

    per_class: list[ClassMetrics] = []
    supported_f1_scores: list[float] = []
    for class_index in range(num_classes):
        target_mask = targets == class_index
        prediction_mask = predictions == class_index
        support = int(target_mask.sum().item())
        if support == 0:
            per_class.append(
                ClassMetrics(
                    class_index=class_index,
                    precision=None,
                    recall=None,
                    f1=None,
                    support=0,
                )
            )
            continue

        true_positives = int((target_mask & prediction_mask).sum().item())
        predicted_count = int(prediction_mask.sum().item())
        precision = true_positives / predicted_count if predicted_count > 0 else 0.0
        recall = true_positives / support
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        supported_f1_scores.append(f1)
        per_class.append(
            ClassMetrics(
                class_index=class_index,
                precision=precision,
                recall=recall,
                f1=f1,
                support=support,
            )
        )

    confusion_counts: dict[tuple[int, int], int] = {}
    for true_class, predicted_class in zip(
        targets.tolist(), predictions.tolist(), strict=True
    ):
        if true_class == predicted_class:
            continue
        pair = (true_class, predicted_class)
        confusion_counts[pair] = confusion_counts.get(pair, 0) + 1
    sorted_confusions = sorted(
        confusion_counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )[:max_confusion_pairs]
    confusion_pairs = tuple(
        ConfusionPair(
            true_class_index=pair[0],
            predicted_class_index=pair[1],
            count=count,
        )
        for pair, count in sorted_confusions
    )

    supported_class_count = len(supported_f1_scores)
    return EvaluationResult(
        loss=float(loss),
        top1_accuracy=top1_accuracy,
        top5_accuracy=top5_accuracy,
        macro_f1=sum(supported_f1_scores) / supported_class_count,
        validation_class_coverage=supported_class_count / num_classes,
        num_samples=targets.shape[0],
        per_class=tuple(per_class),
        confusion_pairs=confusion_pairs,
    )


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    *,
    num_classes: int,
    max_confusion_pairs: int = 10,
    show_progress: bool = False,
    progress_description: str = "Classifier validation",
) -> EvaluationResult:
    """DataLoader 전체를 순회해 validation 결과를 계산한다."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    collected_logits: list[torch.Tensor] = []
    collected_targets: list[torch.Tensor] = []
    use_amp = device.type == "cuda"
    if not isinstance(show_progress, bool):
        raise TypeError("show_progress must be a boolean")
    if not isinstance(progress_description, str) or not progress_description.strip():
        raise ValueError("progress_description must be a non-empty string")

    progress = tqdm(
        data_loader,
        desc=progress_description,
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for batch in progress:
        images = batch["images"].to(device, non_blocking=use_amp)
        targets = batch["targets"].to(device, non_blocking=use_amp)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = targets.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        collected_logits.append(logits.detach().cpu())
        collected_targets.append(targets.detach().cpu())
        progress.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    if total_samples == 0:
        raise ValueError("data_loader must contain at least one sample")

    return compute_classification_metrics(
        torch.cat(collected_logits),
        torch.cat(collected_targets),
        num_classes=num_classes,
        loss=total_loss / total_samples,
        max_confusion_pairs=max_confusion_pairs,
    )


@torch.inference_mode()
def evaluate_with_predictions(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    class_map: Mapping[str, Any],
    criterion: nn.Module,
    device: torch.device,
    *,
    expected_num_classes: int = 118,
    max_confusion_pairs: int = 10,
    show_progress: bool = False,
    progress_description: str = "Classifier detailed validation",
) -> ValidationArtifacts:
    """
    Best checkpoint 최종 분석용 함수입니다.
    지표뿐 아니라 각 샘플에 대해 다음 내용을 남깁니다.
    - 실제 클래스와 예측 클래스
    - category_id, item_seq
    - classifier confidence
    - 이미지 경로와 bbox
    - 정답 여부
    """

    validate_class_map(class_map, expected_num_classes=expected_num_classes)
    class_lookup = {entry["class_index"]: entry for entry in class_map["classes"]}
    model.eval()
    use_amp = device.type == "cuda"
    total_loss = 0.0
    total_samples = 0
    collected_logits: list[torch.Tensor] = []
    collected_targets: list[torch.Tensor] = []
    collected_meta: list[Mapping[str, Any]] = []
    if not isinstance(show_progress, bool):
        raise TypeError("show_progress must be a boolean")
    if not isinstance(progress_description, str) or not progress_description.strip():
        raise ValueError("progress_description must be a non-empty string")

    progress = tqdm(
        data_loader,
        desc=progress_description,
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for batch in progress:
        validate_batch(
            batch,
            class_map,
            expected_num_classes=expected_num_classes,
        )
        images = batch["images"].to(device, non_blocking=use_amp)
        targets = batch["targets"].to(device, non_blocking=use_amp)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        if logits.ndim != 2 or logits.shape[1] != expected_num_classes:
            raise ValueError(
                "model logits must have shape "
                f"[B, {expected_num_classes}], got {tuple(logits.shape)}"
            )
        if not torch.isfinite(logits).all():
            raise ValueError("model logits must not contain NaN or Inf")

        batch_size = targets.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        collected_logits.append(logits.detach().cpu())
        collected_targets.append(targets.detach().cpu())
        collected_meta.extend(batch["meta"])
        progress.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    if total_samples == 0:
        raise ValueError("data_loader must contain at least one sample")

    all_logits = torch.cat(collected_logits)
    all_targets = torch.cat(collected_targets)
    metrics = compute_classification_metrics(
        all_logits,
        all_targets,
        num_classes=expected_num_classes,
        loss=total_loss / total_samples,
        max_confusion_pairs=max_confusion_pairs,
    )
    probabilities = torch.softmax(all_logits.float(), dim=1)
    classifier_scores, predicted_indices = probabilities.max(dim=1)
    predictions: list[ValidationPrediction] = []
    for meta, target_index, predicted_index, classifier_score in zip(
        collected_meta,
        all_targets.tolist(),
        predicted_indices.tolist(),
        classifier_scores.tolist(),
        strict=True,
    ):
        target_entry = class_lookup[target_index]
        predicted_entry = class_lookup[predicted_index]
        prediction = ValidationPrediction(
            sample_id=meta["sample_id"],
            image_id=meta["image_id"],
            image_path=meta["image_path"],
            bbox_xywh=tuple(float(value) for value in meta["bbox_xywh"]),
            group_id=meta["group_id"],
            drug_dir=meta["drug_dir"],
            target_class_index=target_index,
            predicted_class_index=predicted_index,
            target_category_id=target_entry["category_id"],
            predicted_category_id=predicted_entry["category_id"],
            target_item_seq=target_entry["item_seq"],
            predicted_item_seq=predicted_entry["item_seq"],
            classifier_score=classifier_score,
            correct=target_index == predicted_index,
        )
        predictions.append(prediction)

    error_samples = tuple(
        prediction for prediction in predictions if not prediction.correct
    )
    return ValidationArtifacts(
        metrics=metrics,
        predictions=tuple(predictions),
        error_samples=error_samples,
    )


def save_validation_artifacts(
    artifacts: ValidationArtifacts,
    output_dir: Path,
) -> tuple[Path, Path]:
    """validation 예측과 오분류 샘플을 JSON 파일로 저장한다."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "val_predictions.json"
    errors_path = output_dir / "error_samples.json"
    predictions_path.write_text(
        json.dumps(
            [prediction.to_dict() for prediction in artifacts.predictions],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    errors_path.write_text(
        json.dumps(
            [sample.to_dict() for sample in artifacts.error_samples],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return predictions_path, errors_path
