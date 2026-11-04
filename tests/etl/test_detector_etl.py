"""탐지 학습 패키지 ETL의 확정 경계와 플레이스홀더를 검사한다."""

import copy
import csv
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

import etl.detector.training_package as package_module
from data_loading.detector_package import calculate_detector_package_metadata
from etl.detector.build_detector_dataset import build_detector_dataset
from etl.detector.codeit_source import load_codeit_annotations
from etl.detector.export_yolo_dataset import export_yolo_detector_dataset
from etl.detector.select_source import load_select_annotations
from utils.io import write_json


def _write_codeit_source(root: Path) -> tuple[Path, dict]:
    source_dir = root / "codeit_train_validated"
    annotations_dir = source_dir / "annotations"
    images_dir = source_dir / "images"
    annotations_dir.mkdir(parents=True)
    images_dir.mkdir()

    file_name = "codeit_K-000001-000002_0_2_0_2_70_000_200.png"
    Image.new("RGB", (32, 32), color=(220, 220, 220)).save(images_dir / file_name)
    document = {
        "images": [{"id": 7, "file_name": file_name, "width": 32, "height": 32}],
        "annotations": [
            {
                "id": 10,
                "image_id": 7,
                "category_id": 1,
                "drug_code": "K-000001",
                "bbox": [1, 2, 10, 12],
                "area": 120,
                "iscrowd": 0,
            },
            {
                "id": 11,
                "image_id": 7,
                "category_id": 2,
                "drug_code": "K-000002",
                "bbox": [16, 4, 12, 14],
                "area": 168,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "K-000001"},
            {"id": 2, "name": "K-000002"},
        ],
    }
    (annotations_dir / "sample.json").write_text(
        json.dumps(document),
        encoding="utf-8",
    )
    return source_dir, document


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


def test_codeit_adapter_loads_complete_image_annotations(tmp_path: Path) -> None:
    source_dir, _ = _write_codeit_source(tmp_path)

    samples = load_codeit_annotations(source_dir)

    assert len(samples) == 1
    assert samples[0].dataset == "codeit"
    assert samples[0].group_id == "K-000001-000002"
    assert len(samples[0].document["annotations"]) == 2


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["annotations"].pop(), "cover every pill code"),
        (
            lambda value: value["annotations"][1].update(drug_code="K-000001"),
            "category and drug_code differ",
        ),
        (
            lambda value: value["annotations"][0].update(bbox=[24, 2, 10, 12]),
            "outside the image",
        ),
    ],
)
def test_codeit_adapter_rejects_incomplete_or_invalid_annotations(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    source_dir, document = _write_codeit_source(tmp_path)
    invalid_document = copy.deepcopy(document)
    mutate(invalid_document)
    (source_dir / "annotations/sample.json").write_text(
        json.dumps(invalid_document),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_codeit_annotations(source_dir)


def test_uncertain_detector_sources_remain_explicit_placeholders(
    tmp_path: Path,
) -> None:
    with pytest.raises(NotImplementedError, match="SELECT source adapter"):
        load_select_annotations(tmp_path)

    with pytest.raises(NotImplementedError, match="group-split rule"):
        build_detector_dataset(
            select_source_dir=tmp_path / "select",
            codeit_source_dir=tmp_path / "codeit",
            output_dir=tmp_path / "output",
        )


def test_training_package_metadata_is_written_only_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label_map = {"num_model_classes": 2, "classes": [{"label": 1, "name": "pill"}]}
    summary = {"num_train_images": 3, "num_val_images": 1}
    monkeypatch.setattr(
        package_module,
        "calculate_detector_package_metadata",
        lambda settings: (label_map, summary),
    )

    label_path, summary_path = package_module.write_training_package_metadata(
        tmp_path,
    )

    assert json.loads(label_path.read_text(encoding="utf-8")) == label_map
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    with pytest.raises(FileExistsError, match="metadata already exists"):
        package_module.write_training_package_metadata(
            tmp_path,
        )


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
