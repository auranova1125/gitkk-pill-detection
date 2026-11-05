"""검증된 detector 패키지로 Ultralytics YOLO 탐지기를 학습한다."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from configs.yolo_detector import YoloDetectorConfig
from data_loading.detector_package import (
    DETECTOR_DATA_SUMMARY_FILE,
    DETECTOR_LABEL_MAP_FILE,
)
from etl.detector.export_yolo_dataset import (
    YOLO_EXPORT_SUMMARY_FILE,
    export_yolo_detector_dataset,
)
from utils.io import read_json, write_json
from utils.runtime import resolve_device, set_seed


@dataclass(frozen=True, slots=True)
class YoloTrainingResult:
    """YOLO 학습 실행이 생성한 주요 artifact 경로."""

    run_dir: Path
    dataset_yaml_path: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    metrics_path: Path


def train_yolo_detector(
    config: YoloDetectorConfig,
    *,
    model_factory: Callable[[str], Any] | None = None,
) -> YoloTrainingResult:
    """데이터 view를 갱신하고 Ultralytics 기본 학습기를 실행한다."""

    run_dir = config.run_dir.expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError(
            f"run directory already exists; use a new run_name: {run_dir}"
        )

    dataset_export = export_yolo_detector_dataset(
        config.data_dir,
        overwrite=True,
    )
    selected_device = resolve_device(config.device)
    set_seed(config.seed)

    if model_factory is None:
        from ultralytics import YOLO

        model_factory = YOLO
    model = model_factory(config.model_name)
    train_arguments = _build_train_arguments(
        config,
        dataset_yaml_path=dataset_export.dataset_yaml_path,
        device_name=selected_device.type,
    )
    training_metrics = model.train(**train_arguments)
    actual_run_dir = Path(training_metrics.save_dir).expanduser().resolve()
    if actual_run_dir != run_dir:
        raise RuntimeError(
            f"Ultralytics saved the run to an unexpected directory: {actual_run_dir}"
        )

    result = YoloTrainingResult(
        run_dir=run_dir,
        dataset_yaml_path=dataset_export.dataset_yaml_path,
        best_checkpoint_path=run_dir / "weights" / "best.pt",
        last_checkpoint_path=run_dir / "weights" / "last.pt",
        metrics_path=run_dir / "results.csv",
    )
    _validate_training_artifacts(result)
    _save_run_metadata(config, result)
    return result


def _build_train_arguments(
    config: YoloDetectorConfig,
    *,
    dataset_yaml_path: Path,
    device_name: str,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "data": str(dataset_yaml_path.resolve()),
        "project": str(config.output_dir.expanduser().resolve()),
        "name": config.run_name,
        "exist_ok": False,
        "epochs": config.max_epochs,
        "patience": config.early_stopping_patience,
        "imgsz": config.image_size,
        "batch": config.batch_size,
        "device": device_name,
        "workers": config.workers,
        "cache": config.cache_images,
        "seed": config.seed,
        "deterministic": config.deterministic,
        "optimizer": config.optimizer,
        "weight_decay": config.weight_decay,
        "single_cls": True,
        "val": True,
        "save": True,
        "plots": True,
        "verbose": config.show_progress,
        "iou": config.nms_threshold,
        "fliplr": config.horizontal_flip,
        "flipud": config.vertical_flip,
        "hsv_h": config.hsv_hue,
        "hsv_s": config.hsv_saturation,
        "hsv_v": config.hsv_value,
        "translate": config.translate,
        "scale": config.scale,
        "mosaic": config.mosaic,
        "close_mosaic": config.close_mosaic_epochs,
    }
    if config.initial_learning_rate is not None:
        arguments["lr0"] = config.initial_learning_rate
    return arguments


def _validate_training_artifacts(result: YoloTrainingResult) -> None:
    for path in (
        result.best_checkpoint_path,
        result.last_checkpoint_path,
        result.metrics_path,
    ):
        if not path.is_file():
            raise RuntimeError(f"YOLO training artifact was not created: {path}")


def _save_run_metadata(
    config: YoloDetectorConfig,
    result: YoloTrainingResult,
) -> None:
    data_dir = config.data_dir.expanduser().resolve()
    write_json(
        result.run_dir / "resolved_config.json",
        {**config.to_dict(), "device": resolve_device(config.device).type},
        sort_keys=True,
    )
    for source_name, destination_name in (
        (DETECTOR_LABEL_MAP_FILE, "label_map.json"),
        (DETECTOR_DATA_SUMMARY_FILE, "data_summary.json"),
        (YOLO_EXPORT_SUMMARY_FILE, "yolo_export_summary.json"),
    ):
        write_json(
            result.run_dir / destination_name,
            read_json(data_dir / source_name),
            sort_keys=True,
        )
