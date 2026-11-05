"""YOLO 라벨 변환, 선택 규칙, 실제 forward/backward/checkpoint 검증."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

from data_loading.detector_package import calculate_detector_package_metadata
from data_loading.yolo_detector import encode_boxes, export_yolo_package, verify_export
from detector.yolo_backend import (
    RepositoryTrainer,
    RepositoryValidator,
    evaluate_yolo_checkpoint,
    train_yolo,
)
from utils.io import write_json


@pytest.fixture
def tiny_package(tmp_path):
    root = tmp_path / "source"
    (root / "annotations").mkdir(parents=True)
    rows = []
    for split, ids in (("train", [1, 2]), ("validation", [3, 4])):
        images, annotations = [], []
        for image_id in ids:
            relative = f"{split}/{image_id}.png"
            path = root / relative
            path.parent.mkdir(exist_ok=True)
            Image.new("RGB", (64, 64), (100, image_id * 30, 200)).save(path)
            images.append(dict(id=image_id, file_name=relative, width=64, height=64))
            annotations.append(
                dict(
                    id=image_id,
                    image_id=image_id,
                    category_id=1,
                    drug_code="K-000001",
                    bbox=[16, 16, 32, 32],
                    area=1024,
                )
            )
            rows.append(
                dict(
                    image_id=image_id,
                    dataset="tiny",
                    image_path=relative,
                    original_image_id=str(image_id),
                    group_id=f"g-{image_id}",
                    split=split,
                    annotation_count=1,
                )
            )
        write_json(
            root / "annotations" / f"{split}.json",
            dict(images=images, annotations=annotations),
        )
    with (root / "split_manifest.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    settings = dict(package_dir=str(root), image_roots={"tiny": str(root)})
    label_map, summary = calculate_detector_package_metadata(settings)
    write_json(root / "detector_label_map.json", label_map)
    write_json(root / "detector_data_summary.json", summary)
    return settings, summary


def test_box_conversion_and_export_guard(tiny_package, tmp_path):
    settings, summary = tiny_package
    assert encode_boxes(dict(width=100, height=200, boxes=[[10, 20, 30, 60]])) == (
        "0 0.2 0.2 0.2 0.2\n"
    )
    destination = tmp_path / "converted"
    with pytest.raises(ValueError, match="split_hash"):
        export_yolo_package(settings, destination)
    data = export_yolo_package(
        settings, destination, expected_hash=summary["split_hash"]
    )
    assert verify_export(data) == summary
    with pytest.raises(FileExistsError):
        export_yolo_package(settings, destination, expected_hash=summary["split_hash"])
    (destination / "labels/val/3.txt").write_text("0 0.1 0.1 0.1 0.1\n")
    with pytest.raises(ValueError, match="bbox"):
        verify_export(data)


def test_best_selection_is_lexicographic():
    trainer = RepositoryTrainer.__new__(RepositoryTrainer)
    trainer.repository_best = (-1.0, -1.0)
    trainer.best_fitness = None
    results = iter([(0.4, 0.8), (0.4, 0.9), (0.39, 1.0), (0.4, 0.9), (0.41, 0.5)])
    trainer.validator = lambda _: dict(
        zip(("repo/map_75_95", "repo/mean_average_recall"), next(results), strict=True)
    )
    fitness = [trainer.validate()[1] for _ in range(5)]
    assert fitness == [1.0, 2.0, 1.5, 1.5, 3.0]
    assert trainer.repository_best == (0.41, 0.5)


def test_letterbox_metrics_and_threshold_counts(tmp_path):
    validator = RepositoryValidator(
        args={"plots": False, "save_json": False}, save_dir=tmp_path
    )
    validator.training = True
    validator.data = {"val": "images/val"}
    validator.device = torch.device("cpu")
    validator.init_metrics(SimpleNamespace(names={0: "pill"}))
    prediction = {
        "bboxes": torch.tensor([[16.0, 24.0, 48.0, 40.0]]),
        "conf": torch.tensor([0.2]),
        "cls": torch.tensor([0.0]),
        "extra": torch.empty((1, 0)),
    }
    batch = {
        "img": torch.zeros((1, 3, 64, 64)),
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.25]]),
        "ori_shape": [(64, 128)],
        "ratio_pad": [((0.5, 0.5), (0, 16))],
        "im_file": ["1.png"],
    }
    validator.update_metrics([prediction], batch)
    values = validator.get_stats()
    assert values["repo/map_75_95"] == pytest.approx(1.0)
    assert values["repo/mean_average_recall"] == pytest.approx(1.0)
    assert values["repo/recall_before_threshold"] == 1
    assert values["repo/recall_after_threshold"] == 0
    assert values["repo/mean_count_error_before_threshold"] == 0
    assert values["repo/mean_count_error_after_threshold"] == 1


@pytest.mark.smoke
def test_yolo11s_train_restore_and_standalone_val_loss(
    tiny_package, tmp_path, monkeypatch
):
    """다운로드 없이 실제 YOLO11s 1 epoch와 독립 loss 평가를 검증한다."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    torch.set_num_threads(1)
    settings, summary = tiny_package
    data = export_yolo_package(
        settings, tmp_path / "converted", expected_hash=summary["split_hash"]
    )
    run_dir = tmp_path / "runs" / "smoke"
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "resolved_config.json",
        {
            "model": "yolo11s.pt",
            "backend": "ultralytics-yolo11",
            "split_hash": summary["split_hash"],
        },
    )
    model = YOLO("yolo11s.yaml")
    model.train(
        trainer=RepositoryTrainer,
        data=str(data),
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        epochs=1,
        imgsz=64,
        batch=2,
        device="cpu",
        workers=0,
        amp=False,
        plots=False,
        optimizer="SGD",
        lr0=0.005,
        warmup_epochs=0,
        mosaic=0,
        mixup=0,
        scale=0,
        translate=0,
        fliplr=0,
        hsv_h=0,
        hsv_s=0,
        hsv_v=0,
        augmentations=[],
        conf=0.001,
        iou=0.45,
        max_det=300,
        verbose=False,
    )
    checkpoint = torch.load(
        run_dir / "weights/last.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["optimizer"] is not None
    assert checkpoint["repository_metadata"]["split_hash"] == summary["split_hash"]
    assert checkpoint["repository_metadata"]["best_selection"] == [0.0, 0.0]
    assert checkpoint["epoch"] == 0
    with (run_dir / "results.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert float(rows[0]["train/box_loss"]) > 0
    assert float(rows[0]["val/box_loss"]) > 0
    values = evaluate_yolo_checkpoint(
        run_dir / "weights/best.pt",
        data,
        device="cpu",
        batch=2,
        imgsz=64,
        destination=run_dir / "evaluation",
    )
    assert values["repo/num_images"] == 2
    assert values["val/box_loss"] > 0
    assert values["val/cls_loss"] > 0
    assert values["val/dfl_loss"] > 0
    saved = json.loads((run_dir / "evaluation/evaluation.json").read_text())
    assert saved["metrics"] == values


@pytest.mark.smoke
def test_notebook_entrypoint_writes_report(tiny_package, tmp_path, monkeypatch):
    """Colab에서 부르는 진입점 전체를 다운로드 없는 CPU 데이터로 실행한다."""
    torch.set_num_threads(1)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    settings, summary = tiny_package
    data = export_yolo_package(
        settings, tmp_path / "converted", expected_hash=summary["split_hash"]
    )
    config = yaml.safe_load(
        (Path(__file__).parents[2] / "configs/detector_yolo11s.yaml").read_text()
    )
    config.update(
        run_name="20261103-yolo11s-det-mini-smoke-01",
        output_dir=str(tmp_path / "runs"),
        data_yaml=str(data),
        expected_split_hash=summary["split_hash"],
    )
    config["train"].update(
        epochs=2, device="cpu", batch=2, imgsz=64, workers=0, plots=False, amp=False
    )
    config["tags"][-1] = "device:cpu"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    # 실제 학습은 COCO 가중치, 이 검사는 동일 구조의 초기 가중치를 사용한다.
    def factory(name):
        model = YOLO("yolo11s.yaml" if str(name) == "yolo11s.pt" else name)
        if str(name) == "yolo11s.pt":

            def interrupt(trainer):
                if trainer.epoch == 0:
                    raise RuntimeError("simulated Colab interruption")

            model.add_callback("on_fit_epoch_end", interrupt)
        return model

    monkeypatch.setattr("detector.yolo_backend.YOLO", factory)
    with pytest.raises(RuntimeError, match="simulated Colab interruption"):
        train_yolo(path)
    result = Path(config["output_dir"]) / config["run_name"]
    result = train_yolo(path, resume=result / "weights/last.pt")
    assert (result / "notion_report.md").is_file()
    assert (result / "actual_train_args.json").is_file()
    assert (result / "resources.json").is_file()
    report = (result / "notion_report.md").read_text()
    assert "Best epoch (1-based): 1" in report
    assert "(2 epoch)" in report
    with pytest.raises(FileExistsError):
        train_yolo(path)
    config["train"]["imgsz"] = 96
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="재개 설정"):
        train_yolo(path, resume=result / "weights/last.pt")
