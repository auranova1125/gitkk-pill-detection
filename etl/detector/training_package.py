"""완성된 탐지 학습 패키지를 검증하고 metadata를 생성한다."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from data_loading.detector_package import DEFAULT_SETTINGS, read_records
from utils.io import write_json


def validate_training_package(
    package_dir: Path,
    *,
    image_roots: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """최종 annotation·manifest·이미지 연결을 runtime parser로 검증한다."""

    settings = {
        "package_dir": str(Path(package_dir).expanduser().resolve()),
        "image_roots": {
            name: str(Path(path).expanduser().resolve())
            for name, path in image_roots.items()
        },
        "augmentation": dict(DEFAULT_SETTINGS["augmentation"]),
    }
    _, label_map, data_summary = read_records(settings)
    return label_map, data_summary


def write_training_package_metadata(
    package_dir: Path,
    *,
    image_roots: Mapping[str, Path],
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """검증된 label map과 summary를 최종 패키지에 기록한다."""

    package_dir = Path(package_dir).expanduser().resolve()
    label_map_path = package_dir / "detector_label_map.json"
    data_summary_path = package_dir / "detector_data_summary.json"
    existing = [path for path in (label_map_path, data_summary_path) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"detector package metadata already exists: {names}")

    label_map, data_summary = validate_training_package(
        package_dir,
        image_roots=image_roots,
    )
    write_json(label_map_path, label_map, sort_keys=True)
    write_json(data_summary_path, data_summary, sort_keys=True)
    return label_map_path, data_summary_path
