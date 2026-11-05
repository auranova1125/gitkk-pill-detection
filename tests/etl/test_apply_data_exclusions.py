"""확정 검수 조치를 현재 학습 패키지에 적용하는 테스트."""

import json
from pathlib import Path

import pandas as pd

from data_loading.classifier_package import (
    load_classifier_package,
    read_classifier_samples,
    summarize_classifier_samples,
)
from data_loading.detector_package import load_detector_package
from etl.classifier.apply_review_actions import apply_classifier_review_actions
from etl.detector.apply_image_exclusions import (
    apply_detector_exclusions,
    archive_excluded_detector_images,
)
from etl.detector.training_package import write_training_package_metadata
from utils.io import write_json


def test_apply_detector_exclusions_updates_metadata(tmp_path: Path) -> None:
    package_dir = tmp_path / "detector"
    image_dir = package_dir / "images"
    image_dir.mkdir(parents=True)
    images = []
    annotations = []
    manifest_rows = []
    for image_id, (name, split) in enumerate(
        (("bad", "train"), ("train-good", "train"), ("val-good", "validation")),
        start=1,
    ):
        (image_dir / f"{name}.png").write_bytes(b"image")
        images.append(
            {
                "id": image_id,
                "file_name": f"images/{name}.png",
                "width": 10,
                "height": 10,
            }
        )
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "drug_code": "K-000001",
                "bbox": [0, 0, 5, 5],
                "area": 25,
                "iscrowd": 0,
                "ignore": 0,
            }
        )
        manifest_rows.append(
            {
                "image_id": image_id,
                "image_path": f"images/{name}.png",
                "annotation_count": 1,
                "group_id": f"group-{image_id}",
                "split": split,
            }
        )
    write_json(
        package_dir / "annotations.json",
        {"images": images, "annotations": annotations},
    )
    pd.DataFrame(manifest_rows).to_csv(
        package_dir / "split_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_training_package_metadata(package_dir)
    exclusions_path = tmp_path / "detector_exclusions.csv"
    pd.DataFrame(
        {
            "file_name": ["source_1_bad.png"],
            "split": ["train"],
            "annotation_count": [1],
            "applied_action": ["이미지 및 bbox 전체 제외"],
        }
    ).to_csv(exclusions_path, index=False)

    result = apply_detector_exclusions(
        package_dir,
        exclusions_path=exclusions_path,
    )
    records, _, summary = load_detector_package(package_dir)
    archive_dir = tmp_path / "excluded_images"
    archived_count = archive_excluded_detector_images(
        package_dir,
        archive_dir,
        exclusions_path=exclusions_path,
    )

    assert result.excluded_images == 1
    assert result.excluded_annotations == 1
    assert archived_count == 1
    assert not (package_dir / "images/bad.png").exists()
    assert (archive_dir / "bad.png").is_file()
    assert len(records["train"]) == 1
    assert len(records["validation"]) == 1
    assert summary["num_train_images"] == 1
    assert summary["num_val_images"] == 1


def test_apply_classifier_review_actions_updates_samples_and_crop(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "classifier"
    package_dir.mkdir()
    class_map = {
        "num_classes": 2,
        "classes": [
            {
                "class_index": 0,
                "category_id": 1,
                "item_seq": "item-1",
                "name": "drug-1",
            },
            {
                "class_index": 1,
                "category_id": 2,
                "item_seq": "item-2",
                "name": "drug-2",
            },
        ],
    }
    write_json(package_dir / "class_map.json", class_map)
    sample_specs = [
        ("a-0", "image-a", "train", 0),
        ("a-1", "image-a", "train", 1),
        ("b-0", "image-b", "train", 0),
        ("c-0", "image-c", "train", 0),
        ("d-0", "image-d", "val", 1),
        ("e-0", "image-e", "train", 0),
    ]
    rows = []
    for sample_id, image_id, split, class_index in sample_specs:
        entry = class_map["classes"][class_index]
        code = f"K-{entry['category_id']:06d}"
        crop_path = f"crops/{code}/{sample_id}.png"
        crop = package_dir / crop_path
        crop.parent.mkdir(parents=True, exist_ok=True)
        crop.write_bytes(b"crop")
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "class_index": class_index,
                "code": code,
                "category_id": entry["category_id"],
                "item_seq": entry["item_seq"],
                "name": entry["name"],
                "image_id": image_id,
                "image_path": f"images/{image_id}.png",
                "bbox_xywh": json.dumps([0, 0, 5, 5]),
                "group_id": image_id,
                "drug_dir": "",
                "source": "test",
                "near_dup_in_train": False,
                "crop_path": crop_path,
            }
        )
    pd.DataFrame(rows).to_csv(package_dir / "samples.csv", index=False)
    summary = summarize_classifier_samples(
        read_classifier_samples(package_dir),
        class_map["num_classes"],
    )
    write_json(package_dir / "data_summary.json", summary)
    actions_path = tmp_path / "classifier_actions.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "a-0",
                "image_id": "image-a",
                "code": "K-000001",
                "new_code": "",
                "action": "이미지 제외",
            },
            {
                "sample_id": "a-1",
                "image_id": "image-a",
                "code": "K-000002",
                "new_code": "",
                "action": "이미지 제외",
            },
            {
                "sample_id": "b-0",
                "image_id": "image-b",
                "code": "K-000001",
                "new_code": "",
                "action": "bbox 제외",
            },
            {
                "sample_id": "c-0",
                "image_id": "image-c",
                "code": "K-000001",
                "new_code": "K-000002",
                "action": "라벨 교환 (수정)",
            },
        ]
    ).to_csv(actions_path, index=False)

    result = apply_classifier_review_actions(
        package_dir,
        actions_path=actions_path,
    )
    samples, _, data_summary = load_classifier_package(package_dir)

    assert result.excluded_images == 1
    assert result.excluded_samples == 3
    assert result.relabeled_samples == 1
    assert set(samples["sample_id"]) == {"c-0", "d-0", "e-0"}
    assert samples.set_index("sample_id").loc["c-0", "code"] == "K-000002"
    assert (package_dir / "crops/K-000002/c-0.png").is_file()
    assert data_summary["num_train_samples"] == 2
    assert data_summary["num_val_samples"] == 1
