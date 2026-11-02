"""분류 학습 패키지를 읽고 패키지 내부 파일의 일관성을 확인한다."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils.io import read_json

SPLIT_VERSION = "detector-split-manifest-v1"


def read_classifier_samples(package_dir: str | Path) -> pd.DataFrame:
    """학습 패키지의 ``samples.csv``를 runtime 자료형으로 읽는다."""

    samples = pd.read_csv(
        Path(package_dir) / "samples.csv",
        dtype=str,
        keep_default_na=False,
    )
    samples["class_index"] = samples.class_index.astype(int)
    samples["category_id"] = samples.category_id.astype(int)
    samples["bbox_xywh"] = samples.bbox_xywh.map(
        lambda value: [float(number) for number in json.loads(value)]
    )
    samples["drug_dir"] = samples.drug_dir.astype(object).where(
        samples.drug_dir != "",
        None,
    )
    return samples


def calculate_classifier_split_hash(samples: pd.DataFrame) -> str:
    """고정 train/validation 배정을 재현 가능한 해시로 만든다."""

    key = {
        "version": SPLIT_VERSION,
        "train": sorted(samples[samples.split == "train"].sample_id),
        "validation": sorted(samples[samples.split == "val"].sample_id),
    }
    return hashlib.sha256(
        json.dumps(key, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def summarize_classifier_samples(
    samples: pd.DataFrame,
    num_classes: int,
) -> dict[str, Any]:
    """샘플 표에서 ``data_summary.json``과 같은 요약을 계산한다."""

    train_samples = samples[samples.split == "train"]
    validation_samples = samples[samples.split == "val"]

    def count_classes(split_samples: pd.DataFrame) -> dict[str, int]:
        class_counts = split_samples.class_index.value_counts()
        return {
            str(class_index): int(class_counts.get(class_index, 0))
            for class_index in range(num_classes)
        }

    validation_class_counts = count_classes(validation_samples)
    return {
        "num_classes": num_classes,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(validation_samples),
        "train_class_counts": count_classes(train_samples),
        "val_class_counts": validation_class_counts,
        "train_group_count": train_samples.group_id.nunique(),
        "val_group_count": validation_samples.group_id.nunique(),
        "group_overlap_count": len(
            set(train_samples.group_id) & set(validation_samples.group_id)
        ),
        "validation_class_indices": [
            class_index
            for class_index in range(num_classes)
            if validation_class_counts[str(class_index)] > 0
        ],
        "split_hash": calculate_classifier_split_hash(samples),
    }


def load_classifier_package(
    *,
    class_map_path: str | Path,
    data_summary_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """분류 패키지를 읽고 저장된 요약과 실제 샘플 구성이 같은지 확인한다."""

    class_map_path = Path(class_map_path)
    data_summary_path = Path(data_summary_path)
    package_dir = class_map_path.parent
    class_map = read_json(class_map_path)
    data_summary = read_json(data_summary_path)
    samples = read_classifier_samples(package_dir)
    actual_summary = summarize_classifier_samples(
        samples,
        class_map["num_classes"],
    )
    if actual_summary != data_summary:
        raise ValueError(
            "data_summary.json이 samples.csv와 맞지 않습니다. "
            "같은 패키지의 파일인지 확인하세요."
        )
    return samples, class_map, data_summary


# 기존 ETL 및 외부 호출의 점진적 이전을 위한 이름이다.
read_samples = read_classifier_samples
split_hash = calculate_classifier_split_hash
make_data_summary = summarize_classifier_samples
