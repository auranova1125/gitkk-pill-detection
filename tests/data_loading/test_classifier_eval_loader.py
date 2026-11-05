"""평가 전용 패키지(train 없음) 로더를 검사한다."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from PIL import Image

from data_loading.classifier import build_classifier_eval_loader
from data_loading.classifier_package import (
    calculate_classifier_eval_hash,
    read_classifier_samples,
    summarize_classifier_eval_samples,
)


def _class_map(num_classes: int = 3) -> dict[str, Any]:
    return {
        "num_classes": num_classes,
        "classes": [
            {
                "class_index": index,
                "category_id": 1900 + index,
                "item_seq": f"item-{index:03d}",
                "name": f"합성 품목 {index}",
            }
            for index in range(num_classes)
        ],
    }


def _write_package(root: Path, *, split: str = "eval") -> dict[str, Any]:
    class_map = _class_map()
    (root / "class_map.json").write_text(
        json.dumps(class_map, ensure_ascii=False), encoding="utf-8"
    )
    rows = []
    for k, class_index in enumerate((0, 0, 2)):  # 클래스 1은 평가셋에 없다
        sample_id = f"img{k}_0"
        crop_path = Path("crops") / f"K-{1900 + class_index:06d}" / f"{sample_id}.png"
        (root / crop_path).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20 + k, 12), color=(90, 100, 110 + k)).save(root / crop_path)
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "class_index": class_index,
                "code": f"K-{1900 + class_index:06d}",
                "category_id": 1900 + class_index,
                "item_seq": f"item-{class_index:03d}",
                "name": f"합성 품목 {class_index}",
                "image_id": f"img{k}",
                "image_path": f"aihub_val/x.zip/img{k}.png",
                "bbox_xywh": json.dumps([1.0, 2.0, 8.0, 6.0]),
                "group_id": f"group{k // 2}",
                "drug_dir": "앞면",
                "source": "aihub_val",
                "near_dup_in_train": False,
                "crop_path": str(crop_path),
            }
        )
    pd.DataFrame(rows).to_csv(root / "samples.csv", index=False)
    summary = summarize_classifier_eval_samples(
        read_classifier_samples(root),
        class_map["num_classes"],
    )
    (root / "eval_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def test_eval_summary_counts_only_present_classes(tmp_path: Path) -> None:
    summary = _write_package(tmp_path)

    assert summary["num_eval_samples"] == 3
    assert summary["num_eval_images"] == 3
    assert summary["eval_group_count"] == 2
    assert summary["eval_class_counts"] == {"0": 2, "1": 0, "2": 1}
    assert summary["eval_class_indices"] == [0, 2]


def test_eval_hash_changes_when_label_changes(tmp_path: Path) -> None:
    _write_package(tmp_path)
    samples = read_classifier_samples(tmp_path)
    changed = samples.copy()
    changed.loc[0, "class_index"] = 1

    assert calculate_classifier_eval_hash(samples) == calculate_classifier_eval_hash(
        samples.iloc[::-1]
    )
    assert calculate_classifier_eval_hash(samples) != calculate_classifier_eval_hash(
        changed
    )


def test_eval_loader_reads_all_samples_without_train(tmp_path: Path) -> None:
    summary = _write_package(tmp_path)

    loader, class_map, loaded_summary = build_classifier_eval_loader(
        tmp_path, batch_size=2, device=torch.device("cpu")
    )
    batches = list(loader)

    assert loaded_summary == summary
    assert class_map["num_classes"] == 3
    assert sum(len(batch["targets"]) for batch in batches) == 3
    assert batches[0]["images"].shape == (2, 3, 224, 224)
    assert batches[0]["targets"].tolist() == [0, 0]
    torch.testing.assert_close(
        next(iter(loader))["images"], next(iter(loader))["images"], rtol=0.0, atol=0.0
    )


def test_eval_loader_rejects_training_splits(tmp_path: Path) -> None:
    _write_package(tmp_path, split="val")

    with pytest.raises(ValueError, match="split"):
        build_classifier_eval_loader(tmp_path, device=torch.device("cpu"))


def test_eval_loader_rejects_empty_samples(tmp_path: Path) -> None:
    _write_package(tmp_path)
    samples = pd.read_csv(tmp_path / "samples.csv").iloc[:0]
    samples.to_csv(tmp_path / "samples.csv", index=False)

    with pytest.raises(ValueError, match="샘플이 없습니다"):
        build_classifier_eval_loader(tmp_path, device=torch.device("cpu"))


def test_eval_loader_rejects_mismatched_summary(tmp_path: Path) -> None:
    _write_package(tmp_path)
    summary = json.loads((tmp_path / "eval_summary.json").read_text(encoding="utf-8"))
    summary["num_eval_samples"] = 99
    (tmp_path / "eval_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="eval_summary"):
        build_classifier_eval_loader(tmp_path, device=torch.device("cpu"))
