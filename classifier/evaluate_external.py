"""분류기 checkpoint를 외부 평가 패키지(예: AI Hub val)로 평가한다.

학습에 쓰지 않은 평가 전용 데이터로 실험끼리 비교할 때 쓴다. best epoch 선택은 기존 validation으로 하고,
학습이 끝난 checkpoint를 여기서 평가한다.

결과 (output_dir)
    evaluation_metrics.json   전체·클래스별 지표 (Macro F1은 평가셋에 있는 클래스 평균)
    per_class.csv             클래스별 support, precision, recall, F1
    confusion_pairs.csv       오분류 (정답 → 예측) 쌍과 개수
    predictions.csv           crop마다 정답, 예측, 확신도
    error_samples.csv         오분류 crop (원본 사진 확인용)
    evaluation_report.md      위 내용을 사람이 읽기 쉽게 정리

사용:
    python -m classifier.evaluate_external --checkpoint data/best.pt \\
        --package data/classifier/export/eval_package_aihub_val --output-dir outputs/classifier_eval/aihub_val
"""

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

from classifier.evaluate import (
    EvaluationResult,
    ValidationArtifacts,
    evaluate_with_predictions,
)
from classifier.inference import load_classifier_inference_model
from data_loading.classifier import build_classifier_eval_loader, read_samples
from utils.io import write_json
from utils.runtime import select_device


@dataclass(frozen=True, slots=True)
class ExternalEvaluation:
    metrics: EvaluationResult
    eval_summary: dict[str, Any]
    per_class: pd.DataFrame
    confusion_pairs: pd.DataFrame
    predictions: pd.DataFrame
    clean_scores: dict[str, float | int]  # near_dup_in_train=False인 crop만의 점수


def subset_scores(predictions: pd.DataFrame) -> dict[str, float | int]:
    """crop 일부의 Top-1과 Macro F1 (그 일부에 정답으로 있는 클래스 평균)."""
    if predictions.empty:
        return {
            "num_samples": 0,
            "top1_accuracy": float("nan"),
            "macro_f1": float("nan"),
        }
    f1_scores = []
    for class_index in predictions.target_class_index.unique():
        target = predictions.target_class_index == class_index
        predicted = predictions.predicted_class_index == class_index
        true_positive = int((target & predicted).sum())
        precision = true_positive / predicted.sum() if predicted.any() else 0.0
        recall = true_positive / target.sum()
        f1_scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return {
        "num_samples": len(predictions),
        "top1_accuracy": float(predictions.correct.mean()),
        "macro_f1": float(sum(f1_scores) / len(f1_scores)),
    }


def _class_names(class_map: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "class_index": entry["class_index"],
                "code": f"K-{entry['category_id']:06d}",
                "name": entry["name"],
            }
            for entry in class_map["classes"]
        ]
    ).set_index("class_index")


def summarize_external_evaluation(
    artifacts: ValidationArtifacts,
    class_map: dict[str, Any],
    samples: pd.DataFrame,
    eval_summary: dict[str, Any],
) -> ExternalEvaluation:
    """평가 결과를 표(클래스별, 오분류 쌍, crop별)로 바꾼다."""
    names = _class_names(class_map)
    per_class = pd.DataFrame(
        [m.to_dict() for m in artifacts.metrics.per_class if m.support > 0]
    ).join(names, on="class_index")
    per_class = per_class[
        ["class_index", "code", "name", "support", "precision", "recall", "f1"]
    ].sort_values(["recall", "support"], ascending=[True, False])

    predictions = pd.DataFrame([p.to_dict() for p in artifacts.predictions])
    predictions = predictions.join(
        names.add_prefix("target_"), on="target_class_index"
    ).join(names.add_prefix("predicted_"), on="predicted_class_index")
    columns = ["sample_id", "crop_path"] + (
        ["near_dup_in_train"] if "near_dup_in_train" in samples else []
    )
    predictions = predictions.merge(samples[columns], on="sample_id", how="left")
    if "near_dup_in_train" not in predictions:
        predictions["near_dup_in_train"] = False
    predictions["near_dup_in_train"] = (
        predictions.near_dup_in_train.astype(str).str.lower() == "true"
    )
    errors = predictions[~predictions.correct]
    pairs = (
        errors.groupby(
            ["target_code", "target_name", "predicted_code", "predicted_name"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "target_code"], ascending=[False, True])
    )
    return ExternalEvaluation(
        metrics=artifacts.metrics,
        eval_summary=eval_summary,
        per_class=per_class.reset_index(drop=True),
        confusion_pairs=pairs.reset_index(drop=True),
        predictions=predictions,
        clean_scores=subset_scores(predictions[~predictions.near_dup_in_train]),
    )


def write_external_report(
    result: ExternalEvaluation,
    path: Path,
    *,
    checkpoint_path: Path,
    package_dir: Path,
) -> None:
    m = result.metrics
    errors = result.predictions[~result.predictions.correct]
    lines = [
        "# 외부 평가셋 평가 결과",
        "",
        f"- 실행일: {date.today()}",
        f"- checkpoint: `{checkpoint_path}`",
        f"- 평가 패키지: `{package_dir}` (eval_hash `{result.eval_summary['eval_hash'][:8]}…`)",
        f"- crop {m.num_samples:,}개, 클래스 {len(result.per_class)}종",
        "",
        "| 지표 | 값 |",
        "| --- | ---: |",
        f"| Top-1 | {m.top1_accuracy:.4f} |",
        f"| Top-5 | {m.top5_accuracy:.4f} |",
        f"| Macro F1 (평가셋에 있는 클래스 평균) | {m.macro_f1:.4f} |",
        f"| Loss | {m.loss:.4f} |",
        f"| 오분류 | {len(errors)} |",
        "",
        "학습 사진과 닮은 사진(`near_dup_in_train=True`)을 뺀 점수:",
        "",
        "| crop | Top-1 | Macro F1 |",
        "| ---: | ---: | ---: |",
        f"| {result.clean_scores['num_samples']:,} | {result.clean_scores['top1_accuracy']:.4f} | {result.clean_scores['macro_f1']:.4f} |",
        "",
        "## 오분류 쌍 (정답 → 예측)",
        "",
    ]
    if result.confusion_pairs.empty:
        lines.append("없음")
    else:
        lines += ["| 정답 | 예측 | 개수 |", "| --- | --- | ---: |"]
        lines += [
            f"| {r.target_code} {r.target_name} | {r.predicted_code} {r.predicted_name} | {r.count} |"
            for r in result.confusion_pairs.itertuples()
        ]
    lines += [
        "",
        "## Recall이 1 미만인 클래스",
        "",
    ]
    low = result.per_class[result.per_class.recall < 1]
    if low.empty:
        lines.append("없음")
    else:
        lines += [
            "| 코드 | 품목 | crop | Recall | Precision | F1 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        lines += [
            f"| {r.code} | {r.name} | {r.support} | {r.recall:.3f} | {r.precision:.3f} | {r.f1:.3f} |"
            for r in low.itertuples()
        ]
    lines += [
        "",
        "## 오분류 crop",
        "",
    ]
    if errors.empty:
        lines.append("없음")
    else:
        lines += [
            "| sample_id | 정답 | 예측 | 확신도 |",
            "| --- | --- | --- | ---: |",
        ]
        lines += [
            f"| {r.sample_id} | {r.target_code} | {r.predicted_code} | {r.classifier_score:.3f} |"
            for r in errors.sort_values("sample_id").itertuples()
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_external_package(
    checkpoint_path: Path,
    package_dir: Path,
    output_dir: Path,
    *,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> ExternalEvaluation:
    """checkpoint를 평가 패키지로 평가하고 output_dir에 결과를 저장한다."""
    device = device or select_device()
    loader, class_map, eval_summary = build_classifier_eval_loader(
        package_dir, batch_size=batch_size, device=device
    )
    model, device = load_classifier_inference_model(
        checkpoint_path,
        class_map,
        expected_num_classes=class_map["num_classes"],
        device=device,
    )
    artifacts = evaluate_with_predictions(
        model,
        loader,
        class_map,
        nn.CrossEntropyLoss(),
        device,
        expected_num_classes=class_map["num_classes"],
        max_confusion_pairs=50,
        show_progress=True,
        progress_description="External evaluation",
    )
    result = summarize_external_evaluation(
        artifacts, class_map, read_samples(package_dir), eval_summary
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "evaluation_metrics.json",
        {
            **artifacts.metrics.to_dict(),
            "eval_hash": eval_summary["eval_hash"],
            "without_near_dup": result.clean_scores,
        },
    )
    result.per_class.to_csv(
        output_dir / "per_class.csv", index=False, encoding="utf-8-sig"
    )
    result.confusion_pairs.to_csv(
        output_dir / "confusion_pairs.csv", index=False, encoding="utf-8-sig"
    )
    result.predictions.to_csv(
        output_dir / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    result.predictions[~result.predictions.correct].to_csv(
        output_dir / "error_samples.csv", index=False, encoding="utf-8-sig"
    )
    write_external_report(
        result,
        output_dir / "evaluation_report.md",
        checkpoint_path=checkpoint_path,
        package_dir=package_dir,
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True, help="평가 패키지 폴더")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    result = evaluate_external_package(
        args.checkpoint, args.package, args.output_dir, batch_size=args.batch_size
    )
    m = result.metrics
    print(
        f"Top-1 {m.top1_accuracy:.4f}, Macro F1 {m.macro_f1:.4f}, "
        f"오분류 {int((~result.predictions.correct).sum())}/{m.num_samples}"
    )
    print("저장:", args.output_dir)


if __name__ == "__main__":
    main()
