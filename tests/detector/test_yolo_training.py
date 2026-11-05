"""Ultralytics YOLO 학습 연결부 테스트."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import detector.yolo_training as yolo_training_module
from configs.yolo_detector import YoloDetectorConfig
from data_loading.yolo import YoloDetectorDatasetExport
from detector.yolo_training import train_yolo_detector
from utils.io import write_json


class FakeYoloModel:
    """실제 가중치 다운로드 없이 train 인자를 기록하는 test double."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.train_arguments: dict[str, Any] | None = None

    def train(self, **arguments: Any) -> SimpleNamespace:
        self.train_arguments = arguments
        run_dir = Path(arguments["project"]) / arguments["name"]
        (run_dir / "weights").mkdir(parents=True)
        (run_dir / "weights" / "best.pt").touch()
        (run_dir / "weights" / "last.pt").touch()
        (run_dir / "results.csv").write_text("epoch\n0\n", encoding="utf-8")
        return SimpleNamespace(save_dir=run_dir)


def make_config(tmp_path: Path) -> YoloDetectorConfig:
    return YoloDetectorConfig(
        run_name="yolo-training-test",
        output_dir=tmp_path / "outputs",
        data_dir=tmp_path / "data",
        device="cpu",
        max_epochs=2,
        close_mosaic_epochs=1,
        optimizer="SGD",
        initial_learning_rate=0.002,
    )


def prepare_fake_export(tmp_path: Path) -> YoloDetectorDatasetExport:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dataset_yaml_path = data_dir / "yolo_dataset.yaml"
    dataset_yaml_path.touch()
    for file_name, value in (
        ("detector_label_map.json", {"classes": [{"label": 1, "name": "pill"}]}),
        ("detector_data_summary.json", {"num_train_images": 1}),
        ("yolo_export_summary.json", {"format": "test"}),
    ):
        write_json(data_dir / file_name, value)
    return YoloDetectorDatasetExport(
        dataset_yaml_path=dataset_yaml_path,
        train_list_path=data_dir / "yolo_train.txt",
        validation_list_path=data_dir / "yolo_validation.txt",
        label_dir=data_dir / "labels",
        summary_path=data_dir / "yolo_export_summary.json",
    )


def test_train_yolo_detector_uses_validated_export_and_saves_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    dataset_export = prepare_fake_export(tmp_path)
    export_calls: list[tuple[Path, bool]] = []
    fake_model = FakeYoloModel(config.model_name)

    def fake_export(data_dir: Path, *, overwrite: bool) -> YoloDetectorDatasetExport:
        export_calls.append((Path(data_dir), overwrite))
        return dataset_export

    monkeypatch.setattr(
        yolo_training_module,
        "export_yolo_detector_dataset",
        fake_export,
    )

    result = train_yolo_detector(config, model_factory=lambda _: fake_model)

    assert export_calls == [(config.data_dir, True)]
    assert result.run_dir == config.run_dir.resolve()
    assert result.best_checkpoint_path.is_file()
    assert (result.run_dir / "resolved_config.json").is_file()
    assert (result.run_dir / "label_map.json").is_file()
    assert (result.run_dir / "data_summary.json").is_file()
    assert (result.run_dir / "yolo_export_summary.json").is_file()
    assert fake_model.train_arguments is not None
    assert fake_model.train_arguments["data"] == str(
        dataset_export.dataset_yaml_path.resolve()
    )
    assert fake_model.train_arguments["device"] == "cpu"
    assert fake_model.train_arguments["single_cls"] is True
    assert fake_model.train_arguments["epochs"] == 2
    assert fake_model.train_arguments["lr0"] == pytest.approx(0.002)


def test_train_yolo_detector_rejects_existing_run_before_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    config.run_dir.mkdir(parents=True)
    export_called = False

    def fake_export(*args: Any, **kwargs: Any) -> None:
        nonlocal export_called
        export_called = True

    monkeypatch.setattr(
        yolo_training_module,
        "export_yolo_detector_dataset",
        fake_export,
    )

    with pytest.raises(FileExistsError, match="new run_name"):
        train_yolo_detector(config, model_factory=FakeYoloModel)

    assert not export_called
