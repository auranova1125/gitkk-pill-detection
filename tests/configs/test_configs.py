"""작업별 YAML 설정과 데이터 준비 함수 테스트."""

from pathlib import Path
from typing import Any

import pytest
import torch

import classifier.train as classifier_train_module
import data_loading.factory as data_loading_factory_module
from classifier.train import train_classifier
from classifier.training_data import prepare_classifier_data
from configs import (
    ClassifierConfig,
    PillRecognitionConfig,
    load_classifier_config,
    load_detector_config,
    load_pill_recognition_config,
)
from data_loading.classifier import build_classifier_inputs
from data_loading.factory import load_data_factory
from detector.train import train_detector
from etl.classifier.paths import CLASSES, REPORTS, ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_default_yaml_configs_load_into_existing_model_configs() -> None:
    classifier = load_classifier_config(PROJECT_ROOT / "configs" / "classifier.yaml")
    detector = load_detector_config(PROJECT_ROOT / "configs" / "detector.yaml")
    pill_recognition = load_pill_recognition_config(
        PROJECT_ROOT / "configs" / "pill_recognition.yaml"
    )

    assert classifier.run_name == "20261103-resnet50-cls-v1-baseline-01"
    assert classifier.output_dir == Path("outputs/classifier")
    assert classifier.run_dir == Path(
        "outputs/classifier/20261103-resnet50-cls-v1-baseline-01"
    )
    assert detector.model_name == "fasterrcnn_resnet50_fpn_v2"
    assert detector.image_min_size == 384
    assert detector.image_max_size == 640
    assert detector.output_dir == Path("outputs/detector")
    assert detector.run_dir == Path(
        "outputs/detector/20261104-fasterrcnn-r50-fpn-v2-det-default-baseline-02"
    )
    assert detector.data_factory_path == "data_loading.detector:build_detector_inputs"
    assert (
        pill_recognition.inference_factory_path
        == "data_loading.recognition:build_pill_recognition_inputs"
    )
    assert pill_recognition.input_dir == Path("data/inference")
    assert pill_recognition.run_name == (
        "20261103-ssdlite320-resnet50-infer-default-smoke-01"
    )
    assert pill_recognition.run_dir == Path(
        "outputs/inference/20261103-ssdlite320-resnet50-infer-default-smoke-01"
    )
    assert pill_recognition.detector_checkpoint_path == (
        pill_recognition.detector_run_dir / "best.pt"
    )
    assert pill_recognition.classifier_class_map_path == (
        pill_recognition.classifier_run_dir / "class_map.json"
    )


def test_classifier_yaml_rejects_unknown_field(tmp_path: Path) -> None:
    config_path = tmp_path / "classifier.yaml"
    config_path.write_text(
        """task: classifier
data_factory_path: package.module:factory
unknown_field: true
run_name: test
output_dir: outputs
data_dir: data/classifier
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown fields"):
        load_classifier_config(config_path)


def test_pill_recognition_config_rejects_non_finite_threshold(tmp_path: Path) -> None:
    values = {
        "run_name": "inference-test",
        "inference_factory_path": None,
        "input_dir": tmp_path / "images",
        "detector_run_dir": tmp_path / "detector-run",
        "classifier_run_dir": tmp_path / "classifier-run",
        "output_dir": tmp_path / "outputs",
        "score_threshold": float("nan"),
    }

    with pytest.raises(ValueError, match="score_threshold"):
        PillRecognitionConfig(**values)


def test_pill_recognition_config_rejects_run_name_path(tmp_path: Path) -> None:
    values = {
        "run_name": "inference/invalid",
        "inference_factory_path": None,
        "input_dir": tmp_path / "images",
        "detector_run_dir": tmp_path / "detector-run",
        "classifier_run_dir": tmp_path / "classifier-run",
        "output_dir": tmp_path / "outputs",
    }

    with pytest.raises(ValueError, match="path separators"):
        PillRecognitionConfig(**values)


def test_pill_recognition_config_rejects_nonstandard_run_name(
    tmp_path: Path,
) -> None:
    values = {
        "run_name": "inference_test",
        "inference_factory_path": None,
        "input_dir": tmp_path / "images",
        "detector_run_dir": tmp_path / "detector-run",
        "classifier_run_dir": tmp_path / "classifier-run",
        "output_dir": tmp_path / "outputs",
    }

    with pytest.raises(ValueError, match="lowercase letters"):
        PillRecognitionConfig(**values)


def test_factory_loader_reports_missing_configuration() -> None:
    with pytest.raises(RuntimeError, match="data_factory_path is not configured"):
        load_data_factory(None)


def test_factory_loader_imports_callable() -> None:
    factory = load_data_factory("utils.io:write_json")

    assert callable(factory)


def test_classifier_yaml_uses_canonical_runtime_factory() -> None:
    config = load_classifier_config(PROJECT_ROOT / "configs" / "classifier.yaml")

    factory = load_data_factory(config.data_factory_path)

    assert factory is build_classifier_inputs


def test_classifier_etl_paths_use_etl_and_reports_directories() -> None:
    assert ROOT == PROJECT_ROOT
    assert CLASSES == PROJECT_ROOT / "etl" / "classifier" / "classes.csv"
    assert REPORTS == PROJECT_ROOT / "reports" / "data" / "classifier"
    assert not (PROJECT_ROOT / "classifier_data_build").exists()


def test_classifier_training_reuses_validated_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    classifier_class_map: dict[str, Any],
) -> None:
    num_classes = classifier_class_map["num_classes"]
    counts = {str(index): 1 for index in range(num_classes)}
    data_summary = {
        "num_classes": num_classes,
        "num_train_samples": num_classes,
        "num_val_samples": num_classes,
        "train_class_counts": counts,
        "val_class_counts": counts,
        "train_group_count": num_classes,
        "val_group_count": num_classes,
        "group_overlap_count": 0,
        "validation_class_indices": list(range(num_classes)),
        "split_hash": "training-test",
    }
    batch = {
        "images": torch.zeros(num_classes, 3, 224, 224, dtype=torch.float32),
        "targets": torch.arange(num_classes, dtype=torch.int64),
        "meta": [
            {
                "sample_id": f"sample-{index}",
                "image_id": f"image-{index}",
                "image_path": f"/synthetic/image-{index}.png",
                "category_id": entry["category_id"],
                "item_seq": entry["item_seq"],
                "bbox_xywh": [1.0, 2.0, 3.0, 4.0],
                "group_id": f"group-{index}",
                "drug_dir": None,
            }
            for index, entry in enumerate(classifier_class_map["classes"])
        ],
    }
    factory_calls = 0

    def factory(*, config: ClassifierConfig, device: torch.device) -> tuple[Any, ...]:
        nonlocal factory_calls
        assert config.run_name == "training-test"
        assert device == torch.device("cpu")
        factory_calls += 1
        return [batch], [batch], classifier_class_map, data_summary

    config = ClassifierConfig(
        run_name="training-test",
        output_dir=tmp_path / "outputs",
        data_dir=tmp_path,
        data_factory_path="synthetic.module:factory",
        pretrained=False,
        device="cpu",
    )
    monkeypatch.setattr(
        data_loading_factory_module,
        "load_data_factory",
        lambda _: factory,
    )
    prepared_data = prepare_classifier_data(config)
    sentinel = object()
    monkeypatch.setattr(
        classifier_train_module,
        "run_training",
        lambda *args, **kwargs: sentinel,
    )

    result = train_classifier(config, data=prepared_data)

    assert result is sentinel
    assert factory_calls == 2
    assert prepared_data.device == torch.device("cpu")


def test_detector_training_reports_missing_factory() -> None:
    config = load_detector_config(PROJECT_ROOT / "configs" / "detector.yaml")
    config.data_factory_path = None

    with pytest.raises(RuntimeError, match="data_factory_path is not configured"):
        train_detector(config)
