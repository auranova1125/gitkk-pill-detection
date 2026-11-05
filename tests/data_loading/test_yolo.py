"""검증된 탐지 패키지의 YOLO 학습 view 생성을 검사한다."""

import csv
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from data_loading.detector_package import calculate_detector_package_metadata
from data_loading.yolo import export_yolo_detector_dataset
from utils.io import write_json


def _write_detector_package(root: Path) -> Path:
    package_dir = root / "detector"
    images_dir = package_dir / "images"
    images_dir.mkdir(parents=True)
    images = []
    annotations = []
    manifest_rows = []
    for image_id, split in ((1, "train"), (2, "validation")):
        file_name = f"images/sample-{image_id}.png"
        Image.new("RGB", (100, 50), color=(220, 220, 220)).save(package_dir / file_name)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": 100,
                "height": 50,
            }
        )
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": image_id,
                "drug_code": f"K-{image_id:06d}",
                "bbox": [10.0, 5.0, 20.0, 10.0],
                "area": 200.0,
                "iscrowd": 0,
            }
        )
        manifest_rows.append(
            {
                "image_id": image_id,
                "image_path": file_name,
                "annotation_count": 1,
                "group_id": f"K-{image_id:06d}",
                "split": split,
            }
        )
    (package_dir / "annotations.json").write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [],
            }
        ),
        encoding="utf-8",
    )
    with (package_dir / "split_manifest.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    label_map, data_summary = calculate_detector_package_metadata(package_dir)
    write_json(package_dir / "detector_label_map.json", label_map, sort_keys=True)
    write_json(
        package_dir / "detector_data_summary.json",
        data_summary,
        sort_keys=True,
    )
    return package_dir


def test_yolo_export_reuses_images_and_preserves_fixed_split(tmp_path: Path) -> None:
    package_dir = _write_detector_package(tmp_path)
    original_images = sorted((package_dir / "images").iterdir())

    result = export_yolo_detector_dataset(package_dir)

    assert sorted((package_dir / "images").iterdir()) == original_images
    assert result.train_list_path.read_text(encoding="utf-8").splitlines() == [
        str((package_dir / "images/sample-1.png").resolve())
    ]
    assert result.validation_list_path.read_text(encoding="utf-8").splitlines() == [
        str((package_dir / "images/sample-2.png").resolve())
    ]
    assert (result.label_dir / "sample-1.txt").read_text(
        encoding="utf-8"
    ) == "0 0.2000000000 0.2000000000 0.2000000000 0.2000000000\n"
    assert (result.label_dir / "sample-2.txt").is_file()

    dataset_document = yaml.safe_load(
        result.dataset_yaml_path.read_text(encoding="utf-8")
    )
    assert dataset_document == {
        "path": str(package_dir.resolve()),
        "train": "yolo_train.txt",
        "val": "yolo_validation.txt",
        "names": {0: "pill"},
    }
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["num_train_images"] == 1
    assert summary["num_validation_images"] == 1
    assert summary["num_train_instances"] == 1
    assert summary["num_validation_instances"] == 1


def test_yolo_export_requires_owned_output_before_overwrite(tmp_path: Path) -> None:
    package_dir = _write_detector_package(tmp_path)
    export_yolo_detector_dataset(package_dir)

    with pytest.raises(FileExistsError, match="이미 존재"):
        export_yolo_detector_dataset(package_dir)

    export_yolo_detector_dataset(package_dir, overwrite=True)
    (package_dir / "yolo_export_summary.json").unlink()
    with pytest.raises(ValueError, match="소유권"):
        export_yolo_detector_dataset(package_dir, overwrite=True)
