"""탐지 학습 패키지 ETL의 확정 경계와 플레이스홀더를 검사한다."""

import copy
import json
from pathlib import Path

import pytest
from PIL import Image

import etl.detector.training_package as package_module
from etl.detector.build_detector_dataset import build_detector_dataset
from etl.detector.codeit_source import load_codeit_annotations
from etl.detector.select_source import load_select_annotations


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
        "read_records",
        lambda settings: ({}, label_map, summary),
    )

    label_path, summary_path = package_module.write_training_package_metadata(
        tmp_path,
        image_roots={"select": tmp_path / "select", "codeit": tmp_path / "codeit"},
    )

    assert json.loads(label_path.read_text(encoding="utf-8")) == label_map
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    with pytest.raises(FileExistsError, match="metadata already exists"):
        package_module.write_training_package_metadata(
            tmp_path,
            image_roots={
                "select": tmp_path / "select",
                "codeit": tmp_path / "codeit",
            },
        )
