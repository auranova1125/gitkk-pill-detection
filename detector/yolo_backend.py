"""Ultralytics 8.4.183 전용 탐지 adapter. torchvision checkpoint와 분리한다."""

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from copy import copy, deepcopy
from pathlib import Path

import torch
import ultralytics
import yaml
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import ops

from configs.yaml_loading import validate_run_name
from data_loading.yolo_detector import V2_SPLIT_HASH, verify_export
from detector.evaluate import HIGH_IOU_THRESHOLDS, _count_matches
from utils.io import write_json

SUPPORTED_VERSION = "8.4.183"


class RepositoryValidator(DetectionValidator):
    """동일 NMS 결과에 저장소 COCO 지표와 threshold 전후 통계를 적용한다."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.coco = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        self.high = MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox", iou_thresholds=HIGH_IOU_THRESHOLDS
        )
        self.counts = dict(
            images=0,
            targets=0,
            before=0,
            after=0,
            matches_before=0,
            matches_after=0,
            error_before=0,
            error_after=0,
        )
        self.standalone_loss = {}
        self.loss_batches = 0

    def update_metrics(self, preds, batch):
        # 독립 평가에서도 BN을 eval로 유지하고 YOLO 손실을 계산한다.
        if not self.training and getattr(self, "loss_model", None) is not None:
            with torch.no_grad():
                _, losses = self.loss_model.loss(batch)
            for name, value in losses.items():
                value = float(value.detach().cpu())
                if not math.isfinite(value):
                    raise ValueError(f"검증 손실이 유한하지 않습니다: {name}")
                self.standalone_loss[name] = self.standalone_loss.get(name, 0.0) + value
            self.loss_batches += 1
        for i, raw in enumerate(preds):
            target_batch = self._prepare_batch(i, batch)
            shape, original = target_batch["imgsz"], target_batch["ori_shape"]
            padding = target_batch["ratio_pad"]
            boxes = ops.scale_boxes(
                shape, raw["bboxes"].clone(), original, padding
            ).cpu()
            target_boxes = ops.scale_boxes(
                shape, target_batch["bboxes"].clone(), original, padding
            ).cpu()
            pred = {
                "boxes": boxes.float(),
                "scores": raw["conf"].float().cpu(),
                "labels": raw["cls"].long().cpu() + 1,
            }
            target = {
                "boxes": target_boxes.float(),
                "labels": target_batch["cls"].long().cpu() + 1,
                "iscrowd": torch.zeros(len(target_boxes), dtype=torch.int64),
                "area": (target_boxes[:, 2:] - target_boxes[:, :2]).prod(1),
            }
            self.coco.update([pred], [target])
            self.high.update([pred], [target])
            filtered = {
                key: value[pred["scores"] >= 0.25] for key, value in pred.items()
            }
            counts = self.counts
            counts["images"] += 1
            counts["targets"] += len(target_boxes)
            for suffix, prediction in (("before", pred), ("after", filtered)):
                counts[suffix] += len(prediction["boxes"])
                counts[f"matches_{suffix}"] += _count_matches(
                    prediction, target, iou_threshold=0.5
                )
                counts[f"error_{suffix}"] += abs(
                    len(prediction["boxes"]) - len(target_boxes)
                )
        super().update_metrics(preds, batch)

    def get_stats(self):
        stats = super().get_stats()
        coco, high = self.coco.compute(), self.high.compute()
        self.repository_selection = (float(high["map"]), float(coco["mar_100"]))
        stats.update(
            {
                "repo/map": float(coco["map"]),
                "repo/map_50": float(coco["map_50"]),
                "repo/map_75": float(coco["map_75"]),
                "repo/map_75_95": float(high["map"]),
                "repo/mean_average_recall": float(coco["mar_100"]),
            }
        )
        counts = self.counts
        if counts["images"] == 0 or counts["targets"] == 0:
            raise ValueError("검증 데이터가 비어 있습니다.")
        for suffix in ("before", "after"):
            stats[f"repo/recall_{suffix}_threshold"] = (
                counts[f"matches_{suffix}"] / counts["targets"]
            )
            stats[f"repo/mean_count_error_{suffix}_threshold"] = (
                counts[f"error_{suffix}"] / counts["images"]
            )
        for key in ("images", "targets", "before", "after"):
            stats[f"repo/num_{key}"] = counts[key]
        if self.loss_batches:
            for name, value in self.standalone_loss.items():
                stats[f"val/{name}"] = value / self.loss_batches
        if not all(math.isfinite(float(value)) for value in stats.values()):
            raise ValueError("검증 지표에 NaN/Inf가 있습니다.")
        return stats


class RepositoryTrainer(DetectionTrainer):
    """고 IoU mAP, MAR 순서로 best·early stopping을 선택한다."""

    def __init__(self, *args, **kwargs):
        self.repository_best = (-1.0, -1.0)
        super().__init__(*args, **kwargs)

    def get_validator(self):
        return RepositoryValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def validate(self):
        metrics = self.validator(self)
        metrics.pop("fitness", None)
        current = getattr(
            self.validator,
            "repository_selection",
            (metrics["repo/map_75_95"], metrics["repo/mean_average_recall"]),
        )
        # 개선 순번을 native stopper/save_model에 전달해 사전식 비교를 유지한다.
        best_fitness = self.best_fitness or 0.0
        if current > self.repository_best:
            self.repository_best = current
            fitness = best_fitness + 1.0
            self.best_fitness = fitness
        else:
            fitness = best_fitness - 0.5
        return metrics, fitness

    def save_model(self):
        result = super().save_model()
        metadata = json.loads((self.save_dir / "resolved_config.json").read_text())
        metadata["best_selection"] = list(self.repository_best)
        paths = [self.last]
        if self.best_fitness == self.fitness:
            paths.append(self.best)
        for path in paths:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint["repository_metadata"] = metadata
            temporary = path.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(path)
        return result

    def _load_checkpoint_state(self, checkpoint):
        super()._load_checkpoint_state(checkpoint)
        self.repository_best = tuple(
            checkpoint["repository_metadata"]["best_selection"]
        )

    def final_eval(self):
        # optimizer를 보존해 last 재개를 지원한다. 최종 평가는 외부에서 수행한다.
        return None


def evaluate_yolo_checkpoint(
    checkpoint: Path,
    data_yaml: Path,
    *,
    device="0",
    batch=8,
    imgsz=640,
    destination: Path,
):
    """best/last를 다시 읽어 증강 없는 val loss와 지표를 재계산한다."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(checkpoint))
    loss_model = deepcopy(model.model).float().eval()
    loss_model.args = get_cfg(overrides=model.ckpt["train_args"])
    validator = RepositoryValidator(
        args={
            "data": str(data_yaml),
            "device": device,
            "batch": batch,
            "imgsz": imgsz,
            "conf": 0.001,
            "iou": 0.45,
            "max_det": 300,
            "plots": True,
            "workers": 0,
            "save_json": False,
            "verbose": False,
        },
        save_dir=destination,
    )
    loss_model.to(torch.device("cuda:0" if str(device) == "0" else device))
    validator.loss_model = loss_model
    started = time.perf_counter()
    values = validator(model=model.model)
    values.pop("fitness", None)
    write_json(
        destination / "evaluation.json",
        {
            "checkpoint": str(checkpoint),
            "metrics": values,
            "speed_ms_per_image": validator.speed,
            "validation_seconds": time.perf_counter() - started,
            "loss_aggregation": "mean-of-batch-component-means; no augmentation; eval mode",
            "timing_note": "speed components exclude the separate loss forward; wall time includes it",
            "score_threshold": 0.25,
            "prediction_conf_floor": 0.001,
            "nms_iou": 0.45,
            "max_det": 300,
            "coco_max_detections": [1, 10, 100],
        },
    )
    return values


def write_report(run_dir: Path, values: dict):
    """실측값만 사용한 Notion 붙여넣기용 Markdown을 만든다."""
    config = json.loads((run_dir / "resolved_config.json").read_text())
    rows = list(csv.DictReader((run_dir / "results.csv").open()))
    best = torch.load(
        run_dir / "weights/best.pt", map_location="cpu", weights_only=False
    )
    selected = int(best["epoch"]) + 1
    report = [
        "# YOLO11s · 탐지기 v2 학습 결과",
        "",
        f"- 실행: `{run_dir.name}`",
        f"- 상태: 학습 및 best checkpoint 평가 완료 ({len(rows)} epoch)",
        f"- Best epoch (1-based): {selected}",
        "- 선택 기준: COCO mAP@[0.75:0.95], 동률이면 COCO MAR@100",
        f"- split_hash: `{config['split_hash']}`",
        f"- train / val: {config['data_summary']['num_train_images']} / "
        f"{config['data_summary']['num_val_images']}장",
        "- 클래스: pill 1개 (저장소 label 1 → YOLO class 0)",
        f"- 초기 가중치: `{config['model']}` (COCO 사전학습)",
        f"- 설정: `{json.dumps(config['train'], ensure_ascii=False)}`",
        "",
        "## Best checkpoint 검증 실측값",
        "",
        "| 지표 | 값 |",
        "| --- | ---: |",
    ]
    report.extend(f"| {key} | {float(value):.6f} |" for key, value in values.items())
    report.extend(
        [
            "",
            "## 해석 및 비교 시 주의",
            "",
            "- val은 매 epoch 선택·조기 종료에 사용했으므로 독립 test 성능이 아니다.",
            "- YOLO loss는 box/cls/dfl 성분의 batch 평균이며 SSD loss와 수치를 직접 비교하지 않는다.",
            "- 모델·사전학습(COCO 전체 detector)·입력 크기·letterbox 전처리 차이를 함께 기록한다.",
            "- threshold 전후는 conf 0.001로 생성한 NMS 결과에 score 0.25를 적용한 값이다.",
            "- 예측은 4개로 자르지 않는다. NMS max_det=300, COCO MAR은 maxDet=100이다.",
            "- 기존 torchvision 통합 추론 loader는 이 YOLO checkpoint를 직접 읽지 못한다.",
            "",
            "## 첨부 자료",
            "",
            "- results.csv, results.png: epoch별 train/val loss 및 지표",
            "- weights/best.pt, weights/last.pt: YOLO backend 전용 checkpoint",
            "- best_validation/: 최종 검증 지표·confusion matrix·예측 그림",
            "- resolved_config.json, environment.json, detector_data_summary.json, label map",
            "",
            "## 작성자가 확인할 분석",
            "",
            "- [ ] 손실 곡선과 best epoch 부근 과적합 여부",
            "- [ ] val bbox 그림의 미탐·중복탐지·위치 오차 사례",
            "- [ ] 같은 v2 split의 SSDLite 결과와 고 IoU mAP·recall·개수 오차 비교",
            "- [ ] 추론 시간·GPU peak memory·checkpoint 크기 비교 (동일 장치/측정 조건)",
            "- [ ] YOLO 예측 bbox → 분류기 연결 후 end-to-end 평가 (후속 작업)",
            "",
        ]
    )
    (run_dir / "notion_report.md").write_text("\n".join(report), encoding="utf-8")


def train_yolo(config_path: Path, *, resume: Path | None = None):
    """Notebook/CLI 공유 진입점. 기존 실행과 다른 데이터 재개를 거부한다."""
    if ultralytics.__version__ != SUPPORTED_VERSION:
        raise RuntimeError(f"검증된 ultralytics=={SUPPORTED_VERSION}가 필요합니다.")
    config = yaml.safe_load(Path(config_path).read_text())
    validate_run_name(config["run_name"])
    data_yaml = Path(config["data_yaml"]).resolve()
    summary = verify_export(data_yaml)
    if summary["split_hash"] != config.get("expected_split_hash", V2_SPLIT_HASH):
        raise ValueError("실험의 expected_split_hash와 데이터가 다릅니다.")
    if config["model"] != "yolo11s.pt":
        raise ValueError("이 실험의 모델은 사전학습 yolo11s.pt입니다.")
    options = config["train"]
    for key, value in {"conf": 0.001, "iou": 0.45, "max_det": 300}.items():
        if options.get(key) != value:
            raise ValueError(f"저장소 비교 평가에는 {key}={value}가 필요합니다.")
    if str(options["device"]) not in {"0", "cpu", "mps"}:
        raise ValueError("이 adapter는 단일 GPU/CPU/MPS 실행만 지원합니다.")
    if options.get("val") is not True or options.get("fraction") != 1.0:
        raise ValueError("전체 train과 매 epoch val을 사용해야 합니다.")
    if options.get("save_period", -1) != -1:
        raise ValueError("metadata 보존을 위해 best/last checkpoint만 지원합니다.")
    run_dir = Path(config["output_dir"]).resolve() / config["run_name"]
    resolved = dict(
        config,
        split_hash=summary["split_hash"],
        data_summary=summary,
        backend="ultralytics-yolo11",
        selection="map_75_95_then_mar_100",
    )
    resolved["data_yaml"] = str(data_yaml)
    if resume:
        resume = Path(resume).resolve()
        if resume != run_dir / "weights/last.pt":
            raise ValueError("같은 실행 폴더의 last.pt로만 재개할 수 있습니다.")
        previous = torch.load(resume, map_location="cpu", weights_only=False)
        metadata = dict(previous["repository_metadata"])
        metadata.pop("best_selection")
        if metadata != resolved:
            raise ValueError("재개 설정·모델·데이터가 원래 실행과 다릅니다.")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "resolved_config.json", resolved)
        for name in (
            "detector_label_map.json",
            "yolo_label_map.json",
            "detector_data_summary.json",
            "export_complete.json",
        ):
            (run_dir / name).write_bytes((data_yaml.parent / name).read_bytes())
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        )
        write_json(
            run_dir / "environment.json",
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "ultralytics": ultralytics.__version__,
                "git_commit": result.stdout.strip(),
                "git_status": dirty.stdout,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
                "tags": config["tags"],
                "source_snapshot": (
                    json.loads(Path("SOURCE_SNAPSHOT.json").read_text())
                    if Path("SOURCE_SNAPSHOT.json").is_file()
                    else None
                ),
            },
        )
    model = YOLO(str(resume) if resume else config["model"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train(
        trainer=RepositoryTrainer,
        data=str(data_yaml),
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        resume=bool(resume),
        **options,
    )
    write_json(run_dir / "actual_train_args.json", vars(model.trainer.args))
    write_json(
        run_dir / "resources.json",
        {
            "training_wall_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()
            if torch.cuda.is_available()
            else None,
            "best_checkpoint_bytes": (run_dir / "weights/best.pt").stat().st_size,
            "actual_batch_size": model.trainer.batch_size,
            "gradient_accumulation": model.trainer.accumulate,
            "optimizer_groups": [
                {key: group[key] for key in ("lr", "momentum", "weight_decay")}
                for group in model.trainer.optimizer.param_groups
            ],
            "note": "time and peak cover this session (including validation), not earlier interrupted sessions; checkpoint retains optimizer",
        },
    )
    values = evaluate_yolo_checkpoint(
        run_dir / "weights/best.pt",
        data_yaml,
        device=options["device"],
        batch=model.trainer.batch_size,
        imgsz=options["imgsz"],
        destination=run_dir / "best_validation",
    )
    write_report(run_dir, values)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    print(train_yolo(args.config, resume=args.resume))
