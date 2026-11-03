"""구형 Drive 패키지의 읽기 전용 변환과 현행 strict 검증을 확인한다."""

import json

import pytest

from etl.detector.build_detector_v2 import (
    local_settings,
    normalize_sources,
    read_records,
    refresh_metadata,
    write_csv,
)
from utils.io import write_json


def _legacy_package(tmp_path):
    package = tmp_path / "v1"
    (package / "annotations").mkdir(parents=True)
    manifest = []
    for image_id, split, folder in ((1, "train", "train"), (2, "validation", "val")):
        name = f"source_1_K-{image_id:06d}.png"
        relative = f"{folder}/images/{name}"
        image = package / relative
        image.parent.mkdir(parents=True)
        image.write_bytes(b"pixel bytes are not read by metadata inspection")
        portable = f"select/images/{relative}"
        write_json(
            package / "annotations" / f"{split}.json",
            {
                "images": [
                    {"id": image_id, "file_name": portable, "width": 20, "height": 20}
                ],
                "annotations": [
                    {
                        "id": image_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "drug_code": "K-000001",
                        "bbox": [1, 2, 5, 6],
                        "area": 30,
                    }
                ],
            },
        )
        manifest.append(
            {
                "image_id": image_id,
                "dataset": "select",
                "file_name": name,
                "image_path": portable,
                "original_image_id": name[:-4],
                "annotation_count": 1,
                "group_id": f"group-{image_id}",
                "split": split,
            }
        )
    write_csv(package / "split_manifest.csv", list(manifest[0]), manifest)
    write_json(
        package / "data_settings.json",
        {
            "package_dir": "/content/drive/v1",
            "image_roots": {"select": "/content/drive/v1"},
            "augmentation": {"horizontal_flip": 0},
        },
    )
    return package


def test_legacy_inspection_preserves_source_and_uses_real_image_paths(tmp_path):
    package = _legacy_package(tmp_path)
    original = {p: p.read_bytes() for p in package.rglob("*") if p.is_file()}
    records, _, summary = read_records(local_settings(package))
    assert summary["num_train_images"] == summary["num_val_images"] == 1
    record = records["train"][0]
    assert record["relative_path"] == "train/images/source_1_K-000001.png"
    assert record["bbox_xywh"] == [[1.0, 2.0, 5.0, 6.0]]
    assert record["image_path"] == str(package / record["relative_path"])
    assert {p: p.read_bytes() for p in package.rglob("*") if p.is_file()} == original


def test_normalized_package_keeps_strict_summary_validation(tmp_path):
    package = _legacy_package(tmp_path)
    normalize_sources(package, package)
    refresh_metadata(package)
    records, _, summary = read_records(local_settings(package))
    assert len(records["validation"]) == 1
    assert "augmentation" not in summary
    summary["unexpected_build_note"] = "must live in audit summary"
    write_json(package / "detector_data_summary.json", summary)
    with pytest.raises(ValueError, match="실제 집계와 다릅니다"):
        read_records(local_settings(package))
    labels = json.loads((package / "annotations/train.json").read_text())
    assert labels["images"][0]["file_name"].startswith("train/images/")
