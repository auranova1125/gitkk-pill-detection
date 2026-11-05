"""YOLO11s + 고정 ResNet50 추론 및 v2 val 클래스별 COCO 평가.

분류기를 학습하지 않는다. 기존 저장소의 crop/분류/점수 결합 함수를 사용한다.
팀 채점 코드가 제공되기 전에는 reference COCO 평가로 명시한다.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import ultralytics
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm.auto import tqdm
from ultralytics import YOLO

from classifier.checkpoint import hash_class_map
from classifier.inference import classify_detections, load_classifier_inference_model
from configs.yaml_loading import validate_run_name
from data_loading.detector_package import load_detector_package
from data_loading.yolo_detector import V2_SPLIT_HASH
from detector.inference import DetectorPrediction
from pipeline.pill_recognition import build_classifier_batch
from utils.io import write_json

IOUS = [0.75, 0.80, 0.85, 0.90, 0.95]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    write_json(tmp, value)
    tmp.replace(path)


def prepare_records(package_dir, class_map):
    """train/val 무결성 검증 후 category ID를 원형 그대로 유지한다."""
    root = str(Path(package_dir).resolve())
    records, _, summary = load_detector_package(
        {
            "package_dir": root,
            "image_roots": {"select": root, "codeit": root},
        }
    )
    if summary["split_hash"] != V2_SPLIT_HASH:
        raise ValueError("확정 v2 split_hash와 다릅니다.")
    if summary["num_val_images"] != 2026:
        raise ValueError("v2 val 2,026장을 기대합니다.")
    valid_ids = {v["category_id"] for v in class_map["classes"]}
    unknown = {c for r in records["validation"] for c in r["category_ids"]} - valid_ids
    if unknown:
        raise ValueError(f"정답 약품이 공통 class map에 없습니다: {sorted(unknown)}")
    return records["validation"], summary


def score_rows(
    records, predictions_by_image, *, threshold_on, threshold=0.25, ignore_classes=False
):
    """GT가 있는 클래스의 AP를 평균한다. 미탐 이미지도 반드시 평가에 넣는다."""
    if not records:
        raise ValueError("평가할 이미지가 없습니다.")
    if threshold_on not in {"detector", "final"}:
        raise ValueError("threshold_on은 detector 또는 final이어야 합니다.")
    metrics = [
        MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=ious,
            class_metrics=True,
            max_detection_thresholds=[1, 10, 100],
            backend="pycocotools",
        )
        for ious in (IOUS, [0.95])
    ]
    support = {}
    total_predictions = 0
    for record in records:
        image_id = record["image_id"]
        if image_id not in predictions_by_image:
            raise ValueError(f"추론 기록이 없는 이미지: {image_id}")
        raw = predictions_by_image[image_id]
        key = "detector_score" if threshold_on == "detector" else "final_score"
        kept = [p for p in raw if p[key] >= threshold]
        total_predictions += len(kept)
        boxes = torch.tensor(
            [p["bbox_xyxy"] for p in kept], dtype=torch.float32
        ).reshape(-1, 4)
        pred = {
            "boxes": boxes,
            "labels": torch.tensor(
                [0 if ignore_classes else p["category_id"] for p in kept],
                dtype=torch.int64,
            ),
            "scores": torch.tensor(
                [
                    p["detector_score"] if ignore_classes else p["final_score"]
                    for p in kept
                ],
                dtype=torch.float32,
            ),
        }
        target_boxes = torch.tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = [0 if ignore_classes else c for c in record["category_ids"]]
        for label in labels:
            support[label] = support.get(label, 0) + 1
        target = {
            "boxes": target_boxes,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "area": (target_boxes[:, 2:] - target_boxes[:, :2]).prod(1),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        for metric in metrics:
            metric.update([pred], [target])
    computed = [metric.compute() for metric in metrics]
    per_class = {
        label: {"category_id": label, "support": n} for label, n in support.items()
    }
    means = []
    for name, result in zip(("map_75_95", "ap_95"), computed, strict=True):
        values = dict(
            zip(
                result["classes"].reshape(-1).tolist(),
                result["map_per_class"].reshape(-1).tolist(),
                strict=True,
            )
        )
        selected = []
        for label in sorted(support):
            value = float(values.get(label, -1))
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"GT 존재 클래스의 AP가 계산되지 않았습니다: {label}")
            per_class[label][name] = value
            selected.append(value)
        means.append(sum(selected) / len(selected))
    return {
        "map_75_95": means[0],
        "ap_95": means[1],
        "num_images": len(records),
        "num_ground_truths": sum(support.values()),
        "num_predictions": total_predictions,
        "num_gt_classes": len(support),
        "threshold_on": threshold_on,
        "threshold": threshold,
        "ranking_score": "detector_score"
        if ignore_classes
        else "detector_score * classifier_probability",
        "per_class": [per_class[c] for c in sorted(per_class)],
    }


def classify_image(
    model,
    device,
    class_map,
    record,
    boxes,
    scores,
    detector_name,
    classifier_batch_size=64,
):
    """GT bbox가 아니라 YOLO 예측 bbox로 crop한다."""
    with Image.open(record["image_path"]) as source:
        if source.size != (record["width"], record["height"]):
            raise ValueError(f"이미지 크기 불일치: {record['image_id']}")
        image = (
            torch.from_numpy(np.array(source.convert("RGB"), copy=True))
            .permute(2, 0, 1)
            .float()
            .div_(255)
        )
    detections = []
    for i, (box, score) in enumerate(zip(boxes, scores, strict=True)):
        x1, y1, x2, y2 = map(float, box)
        # 원래 좌표를 유지하며 이미지 경계로 clip한다.
        x1, x2 = max(0.0, x1), min(float(record["width"]), x2)
        y1, y2 = max(0.0, y1), min(float(record["height"]), y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("YOLO가 유효하지 않은 bbox를 반환했습니다.")
        detections.append(
            DetectorPrediction(
                image_id=record["image_id"],
                detection_id=f"{record['image_id']}:{i}",
                bbox_xyxy=(x1, y1, x2, y2),
                bbox_xywh=(x1, y1, x2 - x1, y2 - y1),
                label=1,
                label_name="pill",
                detector_score=float(score),
                detector_name=detector_name,
            )
        )

    def batches():
        for i in range(0, len(detections), classifier_batch_size):
            yield build_classifier_batch(
                image, detections[i : i + classifier_batch_size]
            )

    result = classify_detections(model, batches(), class_map, device)
    if result.errors:
        raise ValueError(f"분류 class map 오류: {result.errors}")
    output = []
    for p in result.predictions:
        row = p.to_dict()
        x, y, w, h = row["bbox_xywh"]
        row["bbox_xyxy"] = [x, y, x + w, y + h]
        output.append(row)
    return output


def export_predictions(path, records, prediction_map, threshold_on):
    columns = [
        "annotation_id",
        "image_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "score",
        "detector_score",
        "classifier_score",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        index = 0
        for record in records:
            for p in prediction_map[record["image_id"]]:
                key = "detector_score" if threshold_on == "detector" else "final_score"
                if p[key] < 0.25:
                    continue
                index += 1
                x, y, w, h = p["bbox_xywh"]
                writer.writerow(
                    dict(
                        annotation_id=index,
                        image_id=record["image_id"],
                        category_id=p["category_id"],
                        bbox_x=x,
                        bbox_y=y,
                        bbox_w=w,
                        bbox_h=h,
                        score=p["final_score"],
                        detector_score=p["detector_score"],
                        classifier_score=p["classifier_score"],
                    )
                )


def write_report(run_dir, config, scores, metadata):
    rows = [
        "# YOLO11s + 공통 ResNet50 · e2e 평가",
        "",
        f"- 실행: `{config['run_name']}`",
        f"- 탐지기 run: `{config['detector_run']}`",
        f"- 분류기: `{Path(config['classifier_checkpoint']).name}` / class map hash `{metadata['class_map_hash']}`",
        "- 두 모델을 eval 모드로 연결했고, 가중치를 재학습하지 않았다.",
        "- GT bbox가 아닌 YOLO 예측 bbox를 crop해 약품을 분류했다.",
        "- 현재 채점은 reference COCO(pyCOCOtools)이다. 팀 공통 채점 코드와의 동일성은 추가 확인이 필요하다.",
        f"- 이 실행의 주 표기 threshold 위치: `{config['threshold_on']}`. detector/final 두 조건을 함께 기록했다.",
        "- 최종 confidence 순위: 탐지 점수 * 분류 확률. GT가 존재하는 클래스의 AP를 동일 가중 평균했다.",
        "- v2 val은 탐지기 모델 선택에 사용한 데이터이며 독립 test 성능이 아니다.",
        "",
        "| 평가 범위 | threshold 위치 | 이미지 수 | 정답 클래스 수 | mAP@[0.75:0.95] | AP@0.95 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for key, value in scores.items():
        if value is not None:
            rows.append(
                f"| {key} | {value['threshold_on']} | {value['num_images']} | {value['num_gt_classes']} | {value['map_75_95']:.6f} | {value['ap_95']:.6f} |"
            )
    rows += [
        "",
        "## 해석",
        "",
        "- 탐지기 단독 점수는 약품명을 무시한다. e2e는 bbox 위치와 약품 종류가 모두 맞아야 한다.",
        "- 분류기 자체의 별도 데이터 평가와 이번 e2e AP는 다른 지표다.",
        "- 참고 평가의 maxDets=[1,10,100], recall grid=0.00~1.00(0.01 간격)이다. 팀 scorer와 확인 후 비교한다.",
        "- per_class CSV에서 AP@0.95가 낮은 약품부터 확인하고 예측 bbox와 category_id를 비교한다.",
        "- detector=탐지 점수에 0.25 적용, final=곱한 최종 점수에 0.25 적용이다. 팀 조건을 확인하기 전 두 결과를 혼용하지 않는다.",
        "- 이미지별 cache가 있어 중단 시 동일 설정·동일 파일의 미완료 이미지부터 재개할 수 있다.",
        "",
        "## 산출물",
        "",
        "- scores.json, per_class_*.csv: 전체 val·조건별 점수",
        "- val_predictions.csv: 주 표기 조건의 검증 예측. 공식 test 제출 파일이 아니다.",
        "- metadata.json: 각 가중치 파일 SHA-256·split_hash·전처리·평가 설정",
        "- cache/: 이미지별 예측 결과; 평가에 사용한 0.25 이상 탐지의 분류 기록",
        "- sessions.jsonl: 재연결 세션별 환경·처리 이미지 수",
        "",
    ]
    (run_dir / "notion_e2e_report.md").write_text("\n".join(rows), encoding="utf-8")


def run_e2e(config):
    if ultralytics.__version__ != "8.4.183":
        raise RuntimeError("ultralytics==8.4.183을 사용하세요.")
    config = dict(config)
    validate_run_name(config["run_name"])
    if config["threshold_on"] not in {"detector", "final"}:
        raise ValueError("threshold_on을 detector 또는 final로 지정하세요.")
    for key, field in (
        ("expected_classifier_sha256", "classifier_checkpoint"),
        ("expected_class_map_sha256", "class_map"),
        ("expected_detector_sha256", "detector_checkpoint"),
    ):
        if config.get(key) and file_hash(config[field]) != config[key]:
            raise ValueError(f"공유/선택된 파일의 SHA-256 불일치: {field}")
    device = torch.device(config["device"])
    class_map = json.loads(Path(config["class_map"]).read_text())
    model, device = load_classifier_inference_model(
        Path(config["classifier_checkpoint"]),
        class_map,
        expected_num_classes=118,
        device=device,
    )
    model.requires_grad_(False)
    records, summary = prepare_records(config["package_dir"], class_map)
    if config.get("limit"):
        if "-smoke-" not in config["run_name"]:
            raise ValueError("부분 실행은 run_name에 -smoke-를 넣어 구분하세요.")
        records = records[: int(config["limit"])]
    detector_checkpoint = Path(config["detector_checkpoint"])
    metadata = {
        "config": config,
        "split_hash": summary["split_hash"],
        "classifier_file_sha256": file_hash(config["classifier_checkpoint"]),
        "class_map_file_sha256": file_hash(config["class_map"]),
        "class_map_hash": hash_class_map(class_map),
        "detector_file_sha256": file_hash(detector_checkpoint),
        "crop": {
            "implementation": "pipeline.pill_recognition.build_classifier_batch",
            "margin": 0.05,
            "padding_rgb": [124, 116, 104],
            "resize": [224, 224],
            "normalization": "ImageNet mean/std",
            "box_for_scoring": "original detector box, not expanded crop",
        },
        "scoring": {
            "backend": "torchmetrics-pycocotools-reference",
            "iou": IOUS,
            "ap95": 0.95,
            "max_dets": [1, 10, 100],
            "gt_classes_only": True,
            "final_score": "detector * classifier",
            "threshold": 0.25,
        },
        "implementation_sha256": {
            p: file_hash(Path(__file__).resolve().parents[1] / p)
            for p in [
                "pipeline/yolo_e2e.py",
                "pipeline/pill_recognition.py",
                "classifier/inference.py",
                "data_loading/image_transforms.py",
            ]
        },
    }
    run_dir = Path(config["output_dir"]) / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "metadata.json").exists():
        if json.loads((run_dir / "metadata.json").read_text()) != metadata:
            raise ValueError(
                "기존 실행의 설정 또는 파일 hash와 다릅니다. 새 run_name을 사용하세요."
            )
    else:
        if any(run_dir.iterdir()):
            raise FileExistsError(
                "metadata 없는 기존 실행 폴더를 재사용할 수 없습니다."
            )
        atomic_json(run_dir / "metadata.json", metadata)
    cache = run_dir / "cache"
    cache.mkdir(exist_ok=True)
    predictions = {}
    missing = []
    for record in records:
        path = cache / f"{record['image_id']}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if (
                saved["image_id"] != record["image_id"]
                or saved["image_path"] != record["relative_path"]
            ):
                raise ValueError("cache 이미지 식별자 불일치")
            predictions[record["image_id"]] = saved["predictions"]
        else:
            missing.append(record)
    started = time.perf_counter()
    if config.get("image_cache_dir"):
        image_cache = Path(config["image_cache_dir"])
        image_cache.mkdir(parents=True, exist_ok=True)
        for record in tqdm(missing, desc="미완료 val 이미지 로컬 복사"):
            source = Path(record["image_path"])
            target = image_cache / f"{record['image_id']}{source.suffix.lower()}"
            if not target.is_file() or target.stat().st_size != source.stat().st_size:
                temporary = target.with_suffix(".copying")
                shutil.copyfile(source, temporary)
                temporary.replace(target)
            record["image_path"] = str(target)
    if missing:
        detector = YOLO(str(detector_checkpoint))
        if detector.names != {0: "pill"}:
            raise ValueError(f"단일 pill YOLO가 아닙니다: {detector.names}")
        total = math.ceil(len(missing) / config["detector_batch"])
        for offset in tqdm(
            range(0, len(missing), config["detector_batch"]),
            total=total,
            desc="YOLO → crop → 공통 분류기",
        ):
            batch = missing[offset : offset + config["detector_batch"]]
            results = detector.predict(
                source=[r["image_path"] for r in batch],
                imgsz=640,
                conf=0.001,
                iou=0.45,
                max_det=300,
                batch=config["detector_batch"],
                device=str(device),
                rect=True,
                verbose=False,
                save=False,
            )
            for record, result in zip(batch, results, strict=True):
                if tuple(result.orig_shape) != (record["height"], record["width"]):
                    raise ValueError("YOLO 원본 이미지 크기 불일치")
                boxes = result.boxes
                keep = boxes.conf >= 0.25
                # final score=detector*probability <= detector이므로 양 조건에서 이 사전 필터는 안전하다.
                values = classify_image(
                    model,
                    device,
                    class_map,
                    record,
                    boxes.xyxy[keep].cpu().tolist(),
                    boxes.conf[keep].cpu().tolist(),
                    config["detector_run"],
                    config["classifier_batch"],
                )
                atomic_json(
                    cache / f"{record['image_id']}.json",
                    {
                        "image_id": record["image_id"],
                        "image_path": record["relative_path"],
                        "predictions": values,
                    },
                )
                predictions[record["image_id"]] = values
    session = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "new_images": len(missing),
        "cached_images": len(records) - len(missing),
        "inference_wall_seconds": time.perf_counter() - started,
    }
    with (run_dir / "sessions.jsonl").open("a") as f:
        f.write(json.dumps(session) + "\n")
    print("전체 추론 완료. 클래스별 AP를 계산합니다.")
    scores = {}
    for scope, subset in [("all_val", records)]:
        for threshold_on in ("detector", "final"):
            key = f"{scope}_{threshold_on}"
            value = (
                score_rows(subset, predictions, threshold_on=threshold_on)
                if subset
                else None
            )
            scores[key] = value
            if value:
                with (run_dir / f"per_class_{key}.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["category_id", "support", "map_75_95", "ap_95"]
                    )
                    writer.writeheader()
                    writer.writerows(value["per_class"])
    scores["detector_only_reference"] = score_rows(
        records, predictions, threshold_on="detector", ignore_classes=True
    )
    atomic_json(run_dir / "scores.json", scores)
    export_predictions(
        run_dir / "val_predictions.csv", records, predictions, config["threshold_on"]
    )
    write_report(run_dir, config, scores, metadata)
    atomic_json(
        run_dir / "status.json",
        {
            "status": "complete",
            "num_images": len(records),
            "partial": bool(config.get("limit")),
        },
    )
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run_e2e(json.loads(args.config.read_text())))
