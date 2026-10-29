"""공유 표본 4장으로 모델 연결을 검사한다. 표본을 새로 만들지 않는다."""
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

from .data_records import read_json, read_records
from .settings import DEFAULT_SETTINGS, REPOSITORY_ROOT


def verify_samples(package):
    """기존 ZIP의 파일 해시와 표본 ID를 확인한다. 예전 코드 해시는 실행을 막지 않는다."""
    package = Path(package).resolve()
    manifest = read_json(package / "smoke_package_manifest.json")
    if manifest["version"] != "pill-smoke-package-v1":
        raise ValueError("지원하지 않는 스모크 표본 형식입니다.")
    if (manifest["batch_size"] != 2 or manifest["seed"] != 42
        or manifest["augmentation"] != DEFAULT_SETTINGS["augmentation"]):
        raise ValueError("표본 검사 조건은 배치 2, 시드 42, 증강 없음이어야 합니다.")
    keys = ("version", "files_sha256", "code_sha256", "selected_image_ids",
            "augmentation", "batch_size", "seed")
    encoded = json.dumps({key: manifest[key] for key in keys}, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != manifest["package_hash"]:
        raise ValueError("표본 패키지 식별값이 기록과 다릅니다.")
    required = {"split_manifest.csv", "annotations/train.json", "annotations/validation.json"}
    if not required.issubset(manifest["files_sha256"]):
        raise ValueError("표본 파일 해시 기록이 부족합니다.")
    for name, expected in manifest["files_sha256"].items():
        if name not in required and not name.startswith("images/"):
            continue
        path = (package / name).resolve()
        if not path.is_relative_to(package):
            raise ValueError(f"표본 폴더를 벗어난 경로: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"표본 파일이 변경되었습니다: {name}")
    settings = {
        "package_dir": str(package),
        "image_roots": {"select": str(package / "images/select_aihub"),
                        "codeit": str(package / "images/select_codeit")},
        "augmentation": dict(DEFAULT_SETTINGS["augmentation"]),
    }
    records, _, summary = read_records(settings)
    for split, prefix in (("train", "train"), ("validation", "val")):
        ids = {record["image_id"] for record in records[split]}
        if len(ids) != 2 or ids != set(manifest["selected_image_ids"][split]):
            raise ValueError(f"{split} 표본은 기록된 2장이어야 합니다.")
        if summary[f"num_{prefix}_instances"] != manifest["counts"][split]["boxes"]:
            raise ValueError(f"{split} 표본 박스 수가 기록과 다릅니다.")
        for record in records[split]:
            relative = Path(record["image_path"]).relative_to(package).as_posix()
            if relative not in manifest["files_sha256"]:
                raise ValueError(f"이미지 해시 기록이 없습니다: {relative}")
    return settings, manifest, summary


def run_model_smoke(package, *, device_name="cpu"):
    from .start import make_config, checked_loaders
    import torch
    from detector import build_detector_model, validate_detection_batch

    settings, manifest, _ = verify_samples(package)
    print("[1/3 통과] 공유 표본 파일·ID·박스 수 확인")
    config = make_config(run_name="smoke_test", batch_size=2, seed=42)
    device, train, val, label_map, summary = checked_loaders(
        config=config, device_name=device_name, settings=settings,
    )
    train_batch, val_batch = next(iter(train)), next(iter(val))
    for batch in (train_batch, val_batch):
        validate_detection_batch(batch, label_map)
    print("[2/3 통과] Train·Validation 입력 형식 확인")
    torch.manual_seed(config.seed)
    model = build_detector_model(config.model_name, num_classes=2, pretrained_backbone=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    model.train()
    images = [image.to(device) for image in train_batch["images"]]
    targets = [{key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()} for target in train_batch["targets"]]
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
        raise ValueError("모델 기울기가 유효하지 않습니다.")
    optimizer.step()
    model.eval()
    with torch.no_grad():
        predictions = model([image.to(device) for image in val_batch["images"]])
    if len(predictions) != len(val_batch["images"]):
        raise ValueError("Validation 예측 개수가 이미지 수와 다릅니다.")
    for prediction in predictions:
        boxes, labels, scores = (prediction[key] for key in ("boxes", "labels", "scores"))
        count = len(scores)
        if boxes.shape != (count, 4) or labels.shape != (count,) or scores.shape != (count,):
            raise ValueError("Validation 예측 형식이 올바르지 않습니다.")
        if not torch.isfinite(boxes).all().item() or not torch.isfinite(scores).all().item():
            raise ValueError("Validation 예측 수치가 유효하지 않습니다.")
        if not ((scores >= 0) & (scores <= 1)).all().item() or not (labels == 1).all().item():
            raise ValueError("Validation 예측 점수 또는 라벨 오류입니다.")
    verify_samples(package)
    print("[3/3 통과] 손실 계산·역전파·가중치 갱신 1회·Validation 예측 확인")
    environment = {"python": sys.version.split()[0]}
    for name in ("torch", "torchvision", "torchmetrics", "numpy", "Pillow"):
        try:
            environment[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment[name] = None
    code_hashes = {}
    for folder in ("detector", "detector_data"):
        for path in sorted((REPOSITORY_ROOT / folder).glob("*.py")):
            code_hashes[path.relative_to(REPOSITORY_ROOT).as_posix()] = hashlib.sha256(
                path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    return {
        "scope": "shared_fixed_samples", "device": str(device), "batch_size": 2,
        "package_hash": manifest["package_hash"], "sample_files_verified": True,
        "legacy_code_hash_enforced": False, "current_code_sha256": code_hashes,
        "environment": environment, "label_map": label_map, "data_summary": summary,
        "selected_image_ids": manifest["selected_image_ids"], "input_validation": "passed",
        "model_step": {"status": "passed", "model_name": config.model_name,
                       "loss": float(loss.detach().cpu()), "optimizer_steps": 1,
                       "validation_predictions": len(predictions),
                       "pretrained_backbone": False, "checkpoint_saved": False},
    }
