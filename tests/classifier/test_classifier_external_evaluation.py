"""외부 평가셋 결과 정리(클래스별 표, 오분류 쌍, 보고서)를 검사한다."""

from pathlib import Path

import pandas as pd
import torch

from classifier.evaluate import (
    ValidationArtifacts,
    ValidationPrediction,
    compute_classification_metrics,
)
from classifier.evaluate_external import (
    summarize_external_evaluation,
    write_external_report,
)

CLASS_MAP = {
    "num_classes": 3,
    "classes": [
        {"class_index": 0, "category_id": 1900, "item_seq": "1", "name": "가정"},
        {"class_index": 1, "category_id": 16548, "item_seq": "2", "name": "나정"},
        {"class_index": 2, "category_id": 19607, "item_seq": "3", "name": "다정"},
    ],
}


def _artifacts() -> ValidationArtifacts:
    # 정답 0,0,2 / 예측 0,2,2 → 클래스 0 하나가 2로 틀림. 클래스 1은 평가셋에 없다
    targets = torch.tensor([0, 0, 2])
    predicted = [0, 2, 2]
    logits = torch.full((3, 3), -5.0)
    for row, column in enumerate(predicted):
        logits[row, column] = 5.0
    metrics = compute_classification_metrics(logits, targets, num_classes=3, loss=0.1)
    predictions = tuple(
        ValidationPrediction(
            sample_id=f"s{k}",
            image_id=f"img{k}",
            image_path=f"aihub_val/x.zip/img{k}.png",
            bbox_xywh=(0.0, 0.0, 1.0, 1.0),
            group_id="g",
            drug_dir="앞면",
            target_class_index=int(t),
            predicted_class_index=p,
            target_category_id=CLASS_MAP["classes"][int(t)]["category_id"],
            predicted_category_id=CLASS_MAP["classes"][p]["category_id"],
            target_item_seq="",
            predicted_item_seq="",
            classifier_score=0.9,
            correct=int(t) == p,
        )
        for k, (t, p) in enumerate(zip(targets, predicted, strict=True))
    )
    return ValidationArtifacts(
        metrics=metrics,
        predictions=predictions,
        error_samples=tuple(p for p in predictions if not p.correct),
    )


def _samples() -> pd.DataFrame:
    return pd.DataFrame(
        {"sample_id": ["s0", "s1", "s2"], "crop_path": ["c0.png", "c1.png", "c2.png"]}
    )


def test_summary_keeps_only_classes_in_eval_set() -> None:
    result = summarize_external_evaluation(
        _artifacts(), CLASS_MAP, _samples(), {"eval_hash": "abc"}
    )

    assert set(result.per_class.code) == {"K-001900", "K-019607"}
    worst = result.per_class.iloc[0]
    assert worst.code == "K-001900"
    assert worst.recall == 0.5
    assert result.metrics.macro_f1 == (2 / 3 + 2 / 3) / 2


def test_confusion_pairs_and_errors_use_codes_and_names() -> None:
    result = summarize_external_evaluation(
        _artifacts(), CLASS_MAP, _samples(), {"eval_hash": "abc"}
    )

    pair = result.confusion_pairs.iloc[0]
    assert (pair.target_code, pair.predicted_code, pair["count"]) == (
        "K-001900",
        "K-019607",
        1,
    )
    errors = result.predictions[~result.predictions.correct]
    assert errors.sample_id.tolist() == ["s1"]
    assert errors.crop_path.tolist() == ["c1.png"]


def test_report_lists_metrics_and_errors(tmp_path: Path) -> None:
    result = summarize_external_evaluation(
        _artifacts(), CLASS_MAP, _samples(), {"eval_hash": "abcdef0123"}
    )
    path = tmp_path / "evaluation_report.md"

    write_external_report(
        result, path, checkpoint_path=Path("best.pt"), package_dir=Path("pkg")
    )

    text = path.read_text(encoding="utf-8")
    assert "Macro F1" in text
    assert "K-001900 가정 | K-019607 다정 | 1" in text
    assert "| s1 | K-001900 | K-019607 |" in text


def test_clean_scores_drop_near_duplicate_crops() -> None:
    samples = _samples().assign(near_dup_in_train=[False, True, False])

    result = summarize_external_evaluation(
        _artifacts(), CLASS_MAP, samples, {"eval_hash": "abc"}
    )

    assert result.clean_scores["num_samples"] == 2
    assert result.clean_scores["top1_accuracy"] == 1.0
    assert result.clean_scores["macro_f1"] == 1.0
