"""설정 생성과 입력 검증. 이 파일은 전체 학습을 시작하지 않는다."""

from .settings import REPOSITORY_ROOT


def make_config(
    *,
    run_name,
    batch_size,
    seed=42,
    max_epochs=50,
    learning_rate=0.001,
    weight_decay=0.0005,
):
    from detector import DetectorConfig

    output = REPOSITORY_ROOT / "outputs/detector"
    return DetectorConfig(
        run_name=run_name,
        output_dir=output,
        label_map_path=output / run_name / "label_map.json",
        data_summary_path=output / run_name / "data_summary.json",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        score_threshold=0.25,
        nms_threshold=0.55,
        early_stopping_patience=5,
        batch_size=batch_size,
        seed=seed,
        max_epochs=max_epochs,
    )


def checked_loaders(*, config, device_name="auto", settings=None):
    """첫 배치를 검사한 뒤, 아직 소비하지 않은 로더를 반환한다."""
    import torch

    from detector import (
        select_device,
        validate_detector_training_batch,
        validate_detector_data_summary,
        validate_detector_label_map,
    )

    from .detector_inputs import _build_detector_inputs
    from .settings import load_settings

    if device_name not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("장치는 auto, cpu, cuda, mps 중 하나여야 합니다.")
    device = select_device() if device_name == "auto" else torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA를 사용할 수 없습니다.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS를 사용할 수 없습니다.")
    settings = load_settings() if settings is None else settings
    train, val, label_map, summary = _build_detector_inputs(
        config=config, device=device, settings=settings
    )
    validate_detector_label_map(label_map)
    validate_detector_data_summary(summary, label_map)
    for loader in (train, val):
        validate_detector_training_batch(next(iter(loader)), label_map)
    train, val, fresh_map, fresh_summary = _build_detector_inputs(
        config=config, device=device, settings=settings
    )
    if (fresh_map, fresh_summary) != (label_map, summary):
        raise ValueError("로더 재생성 후 집계가 달라졌습니다.")
    return device, train, val, label_map, summary
