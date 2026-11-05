"""분류 학습 패키지를 읽고 패키지 내부 파일의 일관성을 확인한다."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils.io import read_json

CLASSIFIER_SPLIT_VERSION = "detector-split-manifest-v1"
CLASSIFIER_EVAL_VERSION = "external-eval-v1"
CLASSIFIER_EVAL_SPLIT = "eval"
CLASSIFIER_MAPPING_COLUMNS = ("category_id", "item_seq", "name")


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
        "version": CLASSIFIER_SPLIT_VERSION,
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


def calculate_classifier_eval_hash(samples: pd.DataFrame) -> str:
    """평가 샘플과 정답 라벨로 외부 평가 패키지의 식별자를 만든다."""

    key = {
        "version": CLASSIFIER_EVAL_VERSION,
        "samples": sorted(
            f"{sample_id}:{class_index}"
            for sample_id, class_index in zip(
                samples.sample_id,
                samples.class_index,
                strict=True,
            )
        ),
    }
    return hashlib.sha256(
        json.dumps(key, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def summarize_classifier_eval_samples(
    samples: pd.DataFrame,
    num_classes: int,
) -> dict[str, Any]:
    """외부 평가 샘플에서 ``eval_summary.json`` 내용을 계산한다."""

    class_counts = samples.class_index.value_counts()
    counts = {
        str(class_index): int(class_counts.get(class_index, 0))
        for class_index in range(num_classes)
    }
    return {
        "num_classes": num_classes,
        "num_eval_samples": len(samples),
        "num_eval_images": samples.image_id.nunique(),
        "eval_group_count": samples.group_id.nunique(),
        "eval_class_counts": counts,
        "eval_class_indices": [
            class_index
            for class_index in range(num_classes)
            if counts[str(class_index)] > 0
        ],
        "eval_hash": calculate_classifier_eval_hash(samples),
    }


def validate_classifier_sample_mappings(
    samples: pd.DataFrame,
    class_map: dict[str, Any],
) -> None:
    """``samples.csv``의 중복 품목 정보가 class map과 같은지 확인한다."""

    required_columns = {"class_index", "code", *CLASSIFIER_MAPPING_COLUMNS}
    missing_columns = required_columns - set(samples.columns)
    if missing_columns:
        raise ValueError(f"samples.csv에 필수 열이 없습니다: {sorted(missing_columns)}")

    class_lookup = {
        entry["class_index"]: entry for entry in class_map.get("classes", [])
    }
    for row_index, row in samples.iterrows():
        class_index = row["class_index"]
        class_entry = class_lookup.get(class_index)
        if class_entry is None:
            raise ValueError(
                f"samples.csv {row_index}행의 class_index가 class_map에 없습니다: "
                f"{class_index}"
            )
        for column in CLASSIFIER_MAPPING_COLUMNS:
            if row[column] != class_entry[column]:
                raise ValueError(
                    f"samples.csv {row_index}행의 {column}이 class_map과 다릅니다."
                )
        expected_code = f"K-{class_entry['category_id']:06d}"
        if row["code"] != expected_code:
            raise ValueError(
                f"samples.csv {row_index}행의 code가 class_map과 다릅니다."
            )


def load_classifier_package(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """분류 패키지를 읽고 저장된 요약과 실제 샘플 구성이 같은지 확인한다."""

    package_dir = Path(data_dir).expanduser().resolve()
    class_map = read_json(package_dir / "class_map.json")
    data_summary = read_json(package_dir / "data_summary.json")
    samples = read_classifier_samples(package_dir)
    validate_classifier_sample_mappings(samples, class_map)
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


def load_classifier_eval_package(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """학습과 분리된 외부 평가 패키지를 읽고 내부 일관성을 확인한다."""

    package_dir = Path(data_dir).expanduser().resolve()
    class_map = read_json(package_dir / "class_map.json")
    eval_summary = read_json(package_dir / "eval_summary.json")
    samples = read_classifier_samples(package_dir)
    if samples.empty:
        raise ValueError("평가 패키지에 샘플이 없습니다.")
    validate_classifier_sample_mappings(samples, class_map)

    invalid_splits = set(samples.split) - {CLASSIFIER_EVAL_SPLIT}
    if invalid_splits:
        raise ValueError(
            f"평가 패키지의 split은 모두 {CLASSIFIER_EVAL_SPLIT!r}이어야 합니다: "
            f"{sorted(invalid_splits)}"
        )
    actual_summary = summarize_classifier_eval_samples(
        samples,
        class_map["num_classes"],
    )
    if actual_summary != eval_summary:
        raise ValueError(
            "eval_summary.json이 samples.csv와 맞지 않습니다. "
            "같은 패키지의 파일인지 확인하세요."
        )
    return samples, class_map, eval_summary
