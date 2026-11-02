"""분류 데이터 v2 조치 적용과 탐지 조치 변환을 검사한다."""

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from etl.classifier.build_classifier_v2 import (
    apply_label_actions,
    write_classifier_v2_package,
)
from etl.classifier.label_audit.export_detector_actions import (
    build_detector_actions,
)
from utils.io import write_json

CLASS_MAP = {
    "num_classes": 2,
    "classes": [
        {"class_index": 0, "category_id": 1, "item_seq": "100", "name": "가정"},
        {"class_index": 1, "category_id": 2, "item_seq": "200", "name": "나정"},
    ],
}


def _source_samples() -> pd.DataFrame:
    rows = [
        ("a_0", "a", "K-000001", "0"),
        ("a_1", "a", "K-000002", "1"),
        ("b_0", "b", "K-000001", "0"),
        ("b_1", "b", "K-000002", "1"),
        ("c_0", "c", "K-000001", "0"),
    ]
    return pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "image_id": image_id,
                "code": code,
                "class_index": class_index,
                "category_id": str(int(code[2:])),
                "item_seq": str(100 * (int(class_index) + 1)),
                "name": "가정" if class_index == "0" else "나정",
                "crop_path": f"crops/{code}/{sample_id}.png",
                "split": "val",
            }
            for sample_id, image_id, code, class_index in rows
        ]
    )


def _actions(rows: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["sample_id", "image_id", "code", "new_code", "action"],
    )


def test_applies_drops_and_label_swaps() -> None:
    actions = _actions(
        [
            ("a_0", "a", "K-000001", "", "이미지 제외"),
            ("a_1", "a", "K-000002", "", "이미지 제외"),
            ("c_0", "c", "K-000001", "", "bbox 제외"),
            ("b_0", "b", "K-000001", "K-000002", "라벨 교환 (수정)"),
            ("b_1", "b", "K-000002", "K-000001", "라벨 교환 (수정)"),
        ]
    )

    result = apply_label_actions(_source_samples(), actions, CLASS_MAP).set_index(
        "sample_id"
    )

    assert list(result.index) == ["b_0", "b_1"]
    assert result.loc[
        "b_0", ["code", "class_index", "category_id", "item_seq", "name"]
    ].tolist() == ["K-000002", "1", "2", "200", "나정"]
    assert result.loc["b_0", "crop_path"] == "crops/K-000002/b_0.png"
    assert result.loc["b_1", "code"] == "K-000001"


def test_rejects_partial_image_exclusion_and_unknown_sample() -> None:
    partial = _actions([("a_0", "a", "K-000001", "", "이미지 제외")])
    with pytest.raises(ValueError, match="image exclusion is missing"):
        apply_label_actions(_source_samples(), partial, CLASS_MAP)

    unknown = _actions([("z_0", "z", "K-000001", "", "bbox 제외")])
    with pytest.raises(ValueError, match="outside the source package"):
        apply_label_actions(_source_samples(), unknown, CLASS_MAP)

    mismatched = _actions([("c_0", "wrong-image", "K-000001", "", "bbox 제외")])
    with pytest.raises(ValueError, match="metadata differs"):
        apply_label_actions(_source_samples(), mismatched, CLASS_MAP)

    mixed = _actions(
        [
            ("a_0", "a", "K-000001", "", "이미지 제외"),
            ("a_1", "a", "K-000002", "", "bbox 제외"),
        ]
    )
    with pytest.raises(ValueError, match="image exclusion is missing"):
        apply_label_actions(_source_samples(), mixed, CLASS_MAP)


def _package_samples() -> pd.DataFrame:
    rows = []
    for split, prefix in (("train", "t"), ("val", "v")):
        for class_index, code in enumerate(("K-000001", "K-000002")):
            sample_id = f"{prefix}_{class_index}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "class_index": str(class_index),
                    "code": code,
                    "category_id": str(class_index + 1),
                    "item_seq": str((class_index + 1) * 100),
                    "name": "가정" if class_index == 0 else "나정",
                    "image_id": sample_id,
                    "image_path": f"images/{sample_id}.png",
                    "bbox_xywh": json.dumps([0, 0, 12, 12]),
                    "group_id": f"group-{sample_id}",
                    "drug_dir": "",
                    "source": "synthetic",
                    "near_dup_in_train": "False",
                    "crop_path": f"crops/{code}/{sample_id}.png",
                }
            )
    return pd.DataFrame(rows)


def test_writes_valid_package_without_overwriting(tmp_path: Path) -> None:
    source_dir = tmp_path / "v1"
    source_dir.mkdir()
    samples = _package_samples()
    samples.to_csv(source_dir / "samples.csv", index=False)
    write_json(source_dir / "class_map.json", CLASS_MAP)
    for row in samples.to_dict("records"):
        path = source_dir / row["crop_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12), color=(220, 220, 220)).save(path)

    output_dir = tmp_path / "v2"
    result = write_classifier_v2_package(
        source_package_dir=source_dir,
        output_package_dir=output_dir,
        source_samples=samples,
        v2_samples=samples,
    )

    assert result.package_dir == output_dir.resolve()
    assert result.zip_path.is_file()
    assert result.data_summary["num_train_samples"] == 2
    assert result.data_summary["num_val_samples"] == 2
    with pytest.raises(FileExistsError, match="already exists"):
        write_classifier_v2_package(
            source_package_dir=source_dir,
            output_package_dir=output_dir,
            source_samples=samples,
            v2_samples=samples,
        )


def test_converts_label_swap_to_detector_metadata_action() -> None:
    actions = pd.DataFrame(
        [
            {
                "sample_id": "a_0",
                "image_id": "a",
                "code": "K-000001",
                "new_code": "K-000002",
                "action": "라벨 교환 (수정)",
                "split": "val",
                "source": "kaggle",
                "출처": "클래스 검수",
                "메모": "교환",
            }
        ]
    )
    manifest = pd.DataFrame(
        [
            {
                "image_path": "codeit/images/codeit_a.png",
                "dataset": "codeit",
                "split": "validation",
                "group_id": "K-000001-000002",
            }
        ]
    )

    output, missing = build_detector_actions(actions, manifest)

    assert not missing
    assert output.loc[0, "원본_사진"] == "a"
    assert output.loc[0, "탐지기_권장_조치"] == (
        "drug_code 메타데이터 수정 (탐지 label은 pill 유지)"
    )
