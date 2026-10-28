"""Notebook에서 데이터 준비와 팀원의 학습 함수를 연결하는 시작점."""

from __future__ import annotations

import json
from pathlib import Path

from .settings import REPOSITORY_ROOT, load_settings
from .preflight import inspect_package


def prepare_training_inputs(
    *, run_name="pill_baseline", batch_size=8, seed=42, max_epochs=50,
    learning_rate=0.001, weight_decay=0.0005, device_name="auto",
):
    """전체 입력을 검증하고 첫 배치 검사 전 상태의 로더를 반환한다. 학습은 시작하지 않는다."""
    import torch
    from detector import (
        DetectorConfig, select_device, validate_detection_batch,
        validate_detector_data_summary, validate_detector_label_map,
    )
    from .detector_inputs import build_detector_inputs

    inspect_package(load_settings())
    if device_name not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device_name은 auto, cpu, cuda, mps 중 하나여야 합니다.")
    device = select_device() if device_name == "auto" else torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA를 사용할 수 없습니다.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS를 사용할 수 없습니다.")
    run_dir = REPOSITORY_ROOT / "outputs/detector" / run_name
    config = DetectorConfig(
        run_name=run_name, output_dir=REPOSITORY_ROOT / "outputs/detector",
        label_map_path=run_dir / "label_map.json", data_summary_path=run_dir / "data_summary.json",
        learning_rate=learning_rate, weight_decay=weight_decay, score_threshold=0.25,
        nms_threshold=0.55, early_stopping_patience=5,
        batch_size=batch_size, seed=seed, max_epochs=max_epochs,
    )
    if (run_dir / "metrics.csv").exists() or (run_dir / "last.pt").exists():
        raise FileExistsError("같은 이름의 학습 결과가 있습니다. 새로운 run_name을 지정하세요.")
    train, val, label_map, summary = build_detector_inputs(config=config, device=device)
    validate_detector_label_map(label_map)
    validate_detector_data_summary(summary, label_map)
    for loader in (train, val):
        validate_detection_batch(next(iter(loader)), label_map)
    run_dir.mkdir(parents=True, exist_ok=True)
    for path, payload in ((config.label_map_path, label_map), (config.data_summary_path, summary)):
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    train, val, fresh_map, fresh_summary = build_detector_inputs(config=config, device=device)
    if fresh_map != label_map or fresh_summary != summary:
        raise ValueError("로더 재생성 후 매핑 또는 집계가 달라졌습니다.")
    print(f"데이터 준비 완료: Train {len(train.dataset)}장 / Validation {len(val.dataset)}장")
    print(f"장치: {device} / 배치 크기: {batch_size} / 탐지 클래스: pill=1")
    print("학습은 아직 시작하지 않았습니다.")
    return config, device, train, val, label_map, summary
