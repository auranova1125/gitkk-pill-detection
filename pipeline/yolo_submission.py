"""정답 없는 test 이미지로 기존 팀 serializer의 submission.csv를 만든다."""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import torch
import ultralytics
from PIL import Image
from tqdm.auto import tqdm
from ultralytics import YOLO

from classifier.inference import load_classifier_inference_model
from configs.yaml_loading import validate_run_name
from data_loading.recognition import PillRecognitionImageDataset
from pipeline.pill_recognition import (
    SUBMISSION_FIELD_NAMES,
    PillPrediction,
    PillRecognitionResult,
    save_submission_csv,
)
from pipeline.yolo_e2e import atomic_json, classify_image, file_hash


def validate_submission(path, image_sizes, category_ids, sample_path=None):
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != SUBMISSION_FIELD_NAMES:
            raise ValueError("팀 serializer 제출 컬럼 불일치")
        rows = list(reader)
    if sample_path:
        with Path(sample_path).open(encoding="utf-8-sig", newline="") as f:
            template = next(csv.reader(f))
        if tuple(template) != SUBMISSION_FIELD_NAMES:
            raise ValueError(f"Kaggle sample_submission 컬럼과 다릅니다: {template}")
    ids = set()
    predicted_images = set()
    for row in rows:
        annotation_id, image_id, category = map(
            int, (row["annotation_id"], row["image_id"], row["category_id"])
        )
        if (
            annotation_id in ids
            or image_id not in image_sizes
            or category not in category_ids
        ):
            raise ValueError("annotation ID 중복 또는 알 수 없는 이미지/약품 ID")
        ids.add(annotation_id)
        predicted_images.add(image_id)
        x, y, width, height, score = [
            float(row[k]) for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h", "score")
        ]
        image_width, image_height = image_sizes[image_id]
        if not all(math.isfinite(v) for v in (x, y, width, height, score)):
            raise ValueError("NaN/Inf 예측")
        if min(x, y) < 0 or min(width, height) <= 0 or not 0 <= score <= 1:
            raise ValueError("좌표/점수 범위 오류")
        if x + width > image_width + 1e-3 or y + height > image_height + 1e-3:
            raise ValueError("이미지 경계를 벗어난 bbox")
    if not rows:
        raise ValueError("예측 행이 0개입니다. 빈 제출 파일을 만들지 않습니다.")
    return {
        "num_test_images": len(image_sizes),
        "num_predictions": len(rows),
        "num_images_with_predictions": len(predicted_images),
        "num_images_without_predictions": len(image_sizes) - len(predicted_images),
        "sample_columns_verified": bool(sample_path),
        "columns": list(SUBMISSION_FIELD_NAMES),
        "csv_sha256": file_hash(path),
    }


def run_test_submission(config):
    config = dict(config)
    if ultralytics.__version__ != "8.4.183":
        raise RuntimeError("ultralytics==8.4.183을 사용하세요.")
    validate_run_name(config["run_name"])
    if config["threshold_on"] not in {"detector", "final"}:
        raise ValueError("threshold_on은 detector/final 중 하나입니다.")
    for key, field in [
        ("expected_classifier_sha256", "classifier_checkpoint"),
        ("expected_class_map_sha256", "class_map"),
        ("expected_detector_sha256", "detector_checkpoint"),
    ]:
        if config.get(key) and file_hash(config[field]) != config[key]:
            raise ValueError(f"파일 SHA-256 불일치: {field}")
    dataset = PillRecognitionImageDataset(Path(config["input_dir"]))
    samples = dataset.samples
    manifest = [
        {
            "image_id": image_id,
            "relative_path": path.relative_to(dataset.input_dir).as_posix(),
            "bytes": path.stat().st_size,
        }
        for image_id, path in samples
    ]
    signature = {
        "config": config,
        "test_images": manifest,
        "classifier_sha256": file_hash(config["classifier_checkpoint"]),
        "class_map_sha256": file_hash(config["class_map"]),
        "detector_sha256": file_hash(config["detector_checkpoint"]),
        "implementation": {
            f: file_hash(Path(__file__).resolve().parents[1] / f)
            for f in [
                "pipeline/yolo_submission.py",
                "pipeline/yolo_e2e.py",
                "pipeline/pill_recognition.py",
                "data_loading/image_transforms.py",
                "classifier/inference.py",
            ]
        },
    }
    run_dir = Path(config["output_dir"]) / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "submission_metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != signature:
            raise ValueError(
                "기존 제출 실행과 파일/설정/이미지 목록이 다릅니다. 새 실행명을 쓰세요."
            )
    else:
        if any(run_dir.iterdir()):
            raise FileExistsError("metadata 없는 기존 폴더를 재사용하지 않습니다.")
        atomic_json(metadata_path, signature)
    cache = run_dir / "test_cache"
    cache.mkdir(exist_ok=True)
    missing = [
        (image_id, path)
        for image_id, path in samples
        if not (cache / f"{image_id}.json").is_file()
    ]
    class_map = json.loads(Path(config["class_map"]).read_text())
    if missing:
        device = torch.device(config["device"])
        classifier, device = load_classifier_inference_model(
            Path(config["classifier_checkpoint"]),
            class_map,
            expected_num_classes=118,
            device=device,
        )
        classifier.requires_grad_(False)
        detector = YOLO(config["detector_checkpoint"])
        if detector.names != {0: "pill"}:
            raise ValueError("단일 pill YOLO가 아닙니다.")
        local_dir = Path(config["image_cache_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for image_id, source in tqdm(missing, desc="미완료 test 이미지 로컬 복사"):
            target = local_dir / f"{image_id}{source.suffix.lower()}"
            if not target.is_file() or target.stat().st_size != source.stat().st_size:
                temporary = target.with_suffix(".copying")
                shutil.copyfile(source, temporary)
                temporary.replace(target)
            with Image.open(target) as image:
                width, height = image.size
            records.append(
                {
                    "image_id": image_id,
                    "image_path": str(target),
                    "width": width,
                    "height": height,
                    "relative_path": source.relative_to(dataset.input_dir).as_posix(),
                }
            )
        bs = config["detector_batch"]
        for start in tqdm(
            range(0, len(records), bs), desc="test YOLO → crop → 공통 분류기"
        ):
            batch = records[start : start + bs]
            results = detector.predict(
                source=[r["image_path"] for r in batch],
                device=str(device),
                imgsz=640,
                conf=0.001,
                iou=0.45,
                max_det=300,
                batch=bs,
                rect=True,
                verbose=False,
                save=False,
            )
            for record, result in zip(batch, results, strict=True):
                keep = result.boxes.conf >= 0.25
                predictions = classify_image(
                    classifier,
                    device,
                    class_map,
                    record,
                    result.boxes.xyxy[keep].cpu().tolist(),
                    result.boxes.conf[keep].cpu().tolist(),
                    config["detector_run"],
                    config["classifier_batch"],
                )
                atomic_json(
                    cache / f"{record['image_id']}.json",
                    {
                        "image_id": record["image_id"],
                        "relative_path": record["relative_path"],
                        "width": record["width"],
                        "height": record["height"],
                        "predictions": predictions,
                    },
                )
    lookup = {c["class_index"]: c for c in class_map["classes"]}
    predictions = []
    image_sizes = {}
    for entry in manifest:
        image_id = entry["image_id"]
        saved = json.loads((cache / f"{image_id}.json").read_text())
        if (
            saved["image_id"] != image_id
            or saved["relative_path"] != entry["relative_path"]
        ):
            raise ValueError("test cache 이미지 식별 불일치")
        image_sizes[image_id] = (saved["width"], saved["height"])
        for row in saved["predictions"]:
            key = (
                "detector_score"
                if config["threshold_on"] == "detector"
                else "final_score"
            )
            if row[key] < 0.25:
                continue
            cls = lookup[row["class_index"]]
            predictions.append(
                PillPrediction(
                    image_id=str(image_id),
                    detection_id=row["detection_id"],
                    bbox_xywh=tuple(row["bbox_xywh"]),
                    detector_score=row["detector_score"],
                    detector_name=row["detector_name"],
                    class_index=row["class_index"],
                    category_id=row["category_id"],
                    item_seq=row["item_seq"],
                    class_name=cls["name"],
                    classifier_score=row["classifier_score"],
                    final_score=row["final_score"],
                )
            )
    result = PillRecognitionResult(
        predictions=tuple(predictions), detector_errors=(), classifier_errors=()
    )
    # 기존 팀 코드의 제출 파일 저장 기능을 그대로 사용한다.
    submission_path = save_submission_csv(result, run_dir)
    validation = validate_submission(
        submission_path,
        image_sizes,
        {c["category_id"] for c in class_map["classes"]},
        config.get("sample_submission"),
    )
    atomic_json(run_dir / "submission_validation.json", validation)
    note = f"""# Kaggle 제출 기록

- 대회: ai15-level1-project
- 실행: `{config["run_name"]}`
- YOLO run: `{config["detector_run"]}`
- 공통 분류기: `{Path(config["classifier_checkpoint"]).name}`
- test 이미지: {validation["num_test_images"]}장
- 예측 행: {validation["num_predictions"]}개
- 예측 0개인 이미지: {validation["num_images_without_predictions"]}장
- threshold: {config["threshold_on"]} score >= 0.25
- 최종 score: 탐지 점수 * 분류 확률
- submission SHA-256: `{validation["csv_sha256"]}`
- CSV 저장: 기존 pipeline.pill_recognition.save_submission_csv 재사용
- sample_submission 컬럼 대조: {validation["sample_columns_verified"]}

## Kaggle에서 제출한 뒤 직접 기록

- 제출 시각:
- 제출 메시지:
- Public score:
- Private score (공개된 경우만):
- Kaggle 제출 화면 또는 기록:

Test 정답은 공개되지 않아 로컬에서는 점수를 계산하지 않았다. val e2e 점수와 Kaggle 공개 점수는 서로 다른 평가 범위이므로 구분한다. 이 노트북은 CSV만 만들며 Kaggle 업로드는 하지 않는다.
"""
    (run_dir / "kaggle_submission_record.md").write_text(note, encoding="utf-8")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run_test_submission(json.loads(args.config.read_text())))
