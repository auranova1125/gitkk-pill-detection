"""YAML 실행 설정 schema와 factory 로더 테스트."""

from pathlib import Path

import pytest

from configs.schema import (
    PipelineExperimentConfig,
    load_classifier_experiment_config,
    load_detector_experiment_config,
    load_pipeline_experiment_config,
)
from pilldata.factory import load_data_factory
from pipeline.workflows import train_detector_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_configs_load_into_existing_model_configs() -> None:
    classifier = load_classifier_experiment_config(
        PROJECT_ROOT / "configs" / "classifier.yaml"
    )
    detector = load_detector_experiment_config(
        PROJECT_ROOT / "configs" / "detector.yaml"
    )
    pipeline = load_pipeline_experiment_config(
        PROJECT_ROOT / "configs" / "pipeline.yaml"
    )

    assert classifier.training.run_name == "resnet50_baseline"
    assert classifier.expected_num_classes == 118
    assert detector.training.model_name == "ssdlite320_mobilenet_v3_large"
    assert detector.data_factory_path is None
    assert pipeline.inference_factory_path is None


def test_classifier_yaml_rejects_unknown_field(tmp_path: Path) -> None:
    config_path = tmp_path / "classifier.yaml"
    config_path.write_text(
        """task: classifier
data_factory_path: package.module:factory
unknown_field: true
training:
  run_name: test
  output_dir: outputs
  class_map_path: class_map.json
  data_summary_path: data_summary.json
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown fields"):
        load_classifier_experiment_config(config_path)


def test_pipeline_config_rejects_non_finite_threshold(tmp_path: Path) -> None:
    values = {
        "inference_factory_path": None,
        "detector_checkpoint_path": tmp_path / "detector.pt",
        "classifier_checkpoint_path": tmp_path / "classifier.pt",
        "detector_label_map_path": tmp_path / "label_map.json",
        "classifier_class_map_path": tmp_path / "class_map.json",
        "output_dir": tmp_path / "outputs",
        "detector_name": "detector",
        "score_threshold": float("nan"),
    }

    with pytest.raises(ValueError, match="score_threshold"):
        PipelineExperimentConfig(**values)


def test_factory_loader_reports_missing_configuration() -> None:
    with pytest.raises(RuntimeError, match="data_factory_path is not configured"):
        load_data_factory(None)


def test_factory_loader_imports_callable() -> None:
    factory = load_data_factory("utils.io:write_json")

    assert callable(factory)


def test_detector_workflow_reports_missing_factory() -> None:
    experiment = load_detector_experiment_config(
        PROJECT_ROOT / "configs" / "detector.yaml"
    )

    with pytest.raises(RuntimeError, match="data_factory_path is not configured"):
        train_detector_experiment(experiment)
