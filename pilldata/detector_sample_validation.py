"""공유 탐지 표본 패키지의 파일과 라벨 무결성을 검사한다."""

import hashlib
import json
from pathlib import Path

from .detector_records import read_json, read_records
from .detector_settings import DEFAULT_SETTINGS


def verify_detector_sample_package(package):
    """기존 ZIP의 파일 해시와 표본 ID를 확인한다. 예전 코드 해시는 실행을 막지 않는다."""
    package = Path(package).resolve()
    manifest = read_json(package / "smoke_package_manifest.json")
    if manifest["version"] != "pill-smoke-package-v1":
        raise ValueError("지원하지 않는 스모크 표본 형식입니다.")
    if (
        manifest["batch_size"] != 2
        or manifest["seed"] != 42
        or manifest["augmentation"] != DEFAULT_SETTINGS["augmentation"]
    ):
        raise ValueError("표본 검사 조건은 배치 2, 시드 42, 증강 없음이어야 합니다.")
    keys = (
        "version",
        "files_sha256",
        "code_sha256",
        "selected_image_ids",
        "augmentation",
        "batch_size",
        "seed",
    )
    encoded = json.dumps(
        {key: manifest[key] for key in keys}, sort_keys=True, separators=(",", ":")
    ).encode()
    if hashlib.sha256(encoded).hexdigest() != manifest["package_hash"]:
        raise ValueError("표본 패키지 식별값이 기록과 다릅니다.")
    required = {
        "split_manifest.csv",
        "annotations/train.json",
        "annotations/validation.json",
    }
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
        "image_roots": {
            "select": str(package / "images/select_aihub"),
            "codeit": str(package / "images/select_codeit"),
        },
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
