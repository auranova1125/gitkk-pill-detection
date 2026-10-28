"""실행: python -m detector_data check 또는 smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from .preflight import create_smoke_package, inspect_package
from .settings import REPOSITORY_ROOT, load_settings, resolve_data_paths


def make_parser():
    parser = argparse.ArgumentParser(description="고정 분할 알약 탐지 데이터 준비와 검사")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("check", "전체 라벨·분할·이미지 파일 존재 검사"),
        ("loaders", "전체 DataLoader와 첫 배치 검증"),
        ("smoke", "실제 이미지 소량으로 입력 또는 모델 연결 검사"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--report", type=Path, default=None, help="검사 결과 JSON 저장 경로")
        if name != "check":
            sub.add_argument("--batch-size", type=int, default=2 if name == "smoke" else 8)
            sub.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu" if name == "smoke" else "auto")
        if name == "smoke":
            sub.add_argument("--samples-per-split", type=int, default=2)
            sub.add_argument("--model-step", action="store_true", help="SSDLite 순전파·역전파·optimizer 1회 및 Validation 순전파")
    return parser


def import_runtime():
    try:
        import torch
        from detector import (
            DetectorConfig,
            select_device,
            validate_detection_batch,
            validate_detector_data_summary,
            validate_detector_label_map,
        )
        from .detector_inputs import _build_detector_inputs
    except ImportError as error:
        raise RuntimeError(
            f"학습 환경의 Python 패키지를 불러오지 못했습니다: {error}. "
            "저장소의 학습 환경을 활성화하고 의존성을 준비하세요. "
            "경로·라벨만 확인하려면 python -m detector_data check를 실행하세요."
        ) from error
    return torch, DetectorConfig, select_device, validate_detection_batch, validate_detector_data_summary, validate_detector_label_map, _build_detector_inputs


def source_digests(settings):
    package, _ = resolve_data_paths(settings)
    names = ["split_manifest.csv", "annotations/train.json", "annotations/validation.json"]
    return {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in names}


def validate_loaders(settings, args):
    torch, Config, select_device, validate_batch, validate_summary, validate_map, build = import_runtime()
    if args.batch_size < 1:
        raise ValueError("배치 크기는 1 이상이어야 합니다.")
    device = select_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA를 사용할 수 없습니다. --device cpu로 실행하세요.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS를 사용할 수 없습니다. --device cpu로 실행하세요.")
    config = Config(
        run_name="detector_data_smoke" if args.command == "smoke" else "detector_data_check",
        output_dir=REPOSITORY_ROOT / "outputs/detector_data",
        label_map_path=REPOSITORY_ROOT / "outputs/detector_data/label_map.json",
        data_summary_path=REPOSITORY_ROOT / "outputs/detector_data/data_summary.json",
        learning_rate=0.001, weight_decay=0.0005, score_threshold=0.25,
        nms_threshold=0.55, early_stopping_patience=5,
        batch_size=args.batch_size, seed=42, show_progress=False,
    )
    loaders = build(config=config, device=device, settings=settings)
    train, val, label_map, summary = loaders
    validate_map(label_map)
    validate_summary(summary, label_map)
    batches = [next(iter(loader)) for loader in (train, val)]
    for batch in batches:
        validate_batch(batch, label_map)
    # 첫 배치 확인이 학습 순서를 바꾸지 않도록 재생성과 재현성을 검증한다.
    again = build(config=config, device=device, settings=settings)
    if again[2:] != loaders[2:]:
        raise ValueError("같은 설정에서 집계 또는 매핑이 달라졌습니다.")
    for old, loader in zip(batches, again[:2]):
        new = next(iter(loader))
        if old["meta"] != new["meta"] or any(not torch.equal(a, b) for a, b in zip(old["images"], new["images"])):
            raise ValueError("같은 설정에서 첫 배치가 재현되지 않았습니다.")
    result = {"device": str(device), "batch_size": args.batch_size, "label_map": label_map,
              "data_summary": summary, "input_validation": "passed", "first_batch_reproducibility": "passed"}
    if args.command == "smoke" and args.model_step:
        if len(batches[0]["images"]) < 2:
            raise ValueError("모델 연결 검사에는 Train 표본과 배치 크기가 각각 2 이상이어야 합니다.")
        from detector import build_detector_model
        torch.manual_seed(config.seed)
        model = build_detector_model(config.model_name, num_classes=2, pretrained_backbone=False).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        model.train()
        images = [image.to(device) for image in batches[0]["images"]]
        targets = [{key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in target.items()} for target in batches[0]["targets"]]
        optimizer.zero_grad(set_to_none=True)
        losses = model(images, targets)
        if not losses:
            raise ValueError("모델이 손실값을 반환하지 않았습니다.")
        loss = sum(losses.values())
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise ValueError("모델 손실값이 유효하지 않습니다.")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all().item() for g in gradients):
            raise ValueError("모델 gradient가 유효하지 않습니다.")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = model([image.to(device) for image in batches[1]["images"]])
        if len(predictions) != len(batches[1]["images"]):
            raise ValueError("Validation 예측 개수가 이미지 수와 다릅니다.")
        for prediction in predictions:
            boxes, labels, scores = (prediction[key] for key in ("boxes", "labels", "scores"))
            if boxes.ndim != 2 or boxes.shape[-1] != 4 or labels.shape != scores.shape or len(boxes) != len(scores):
                raise ValueError("Validation 예측 형식이 올바르지 않습니다.")
            if not torch.isfinite(boxes).all().item() or not torch.isfinite(scores).all().item():
                raise ValueError("Validation 예측 수치가 유효하지 않습니다.")
            if not ((scores >= 0) & (scores <= 1)).all().item() or not (labels == 1).all().item():
                raise ValueError("Validation 예측 점수 또는 라벨이 올바르지 않습니다.")
        result["model_step"] = {"status": "passed", "model_name": config.model_name,
                                "loss": float(loss.detach().cpu()), "pretrained_backbone": False,
                                "optimizer_steps": 1, "checkpoint_saved": False}
    else:
        result["model_step"] = {"status": "not_run"}
    return result


def main(argv=None):
    args = make_parser().parse_args(argv)
    report_path = args.report or REPOSITORY_ROOT / "outputs/detector_data" / f"{args.command}_report.json"
    if not report_path.is_absolute():
        report_path = REPOSITORY_ROOT / report_path
    result = {"command": args.command, "status": "failed"}
    try:
        settings = load_settings()
        before = source_digests(settings)
        if args.command == "check":
            result.update(inspect_package(settings))
        elif args.command == "loaders":
            inspect_package(settings)
            result.update(validate_loaders(settings, args))
        else:
            with TemporaryDirectory(prefix="pill-detector-smoke-") as directory:
                sources = create_smoke_package(settings, directory, args.samples_per_split)
                subset = dict(settings, package_dir=directory)
                result.update(validate_loaders(subset, args))
                result["selected_images"] = sources
                result["scope"] = "fixed_split_subset"
                result["note"] = "표본의 집계와 해시는 전체 데이터의 집계와 다릅니다."
        if source_digests(settings) != before:
            raise ValueError("검사 중 원본 라벨 또는 분할 파일이 변경되었습니다.")
        result.update(status="passed", source_files_unchanged=True, source_file_sha256=before)
        print("[통과] 탐지 데이터 검사 완료")
        if "data_summary" in result:
            summary = result["data_summary"]
            print(f"Train: {summary['num_train_images']}장 / {summary['num_train_instances']}개 박스")
            print(f"Validation: {summary['num_val_images']}장 / {summary['num_val_instances']}개 박스")
            print(f"모델 연결 검사: {result['model_step']['status']}")
        else:
            print(json.dumps(result["counts"], ensure_ascii=False, indent=2))
        code = 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, AssertionError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        print(f"[실패] {error}", file=sys.stderr)
        code = 1
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"검사 기록: {report_path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
