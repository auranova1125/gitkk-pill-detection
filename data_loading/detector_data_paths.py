"""탐지 학습 패키지와 원본 이미지의 위치를 검증하고 해석한다."""

import copy
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from utils.io import read_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETECTOR_DATA_PATHS = {
    "package_dir": "data/detector/training_package_detector",
    "image_roots": {
        "default": "data/detector/training_package_detector/images",
    },
}


def load_detector_data_paths(path: str | Path | None = None) -> dict[str, Any]:
    """기본값 또는 JSON에서 탐지 패키지와 이미지 루트 위치를 읽는다."""

    if path is None:
        settings = copy.deepcopy(DEFAULT_DETECTOR_DATA_PATHS)
    else:
        settings_path = resolve_repository_path(path)
        if not settings_path.is_file():
            raise FileNotFoundError(
                f"지정한 데이터 설정 파일이 없습니다: {settings_path}"
            )
        settings = read_json(settings_path)
    validate_detector_data_paths(settings)
    return dict(settings)


def validate_detector_data_paths(settings: Any) -> None:
    """위치 설정에 패키지와 이미지 루트만 있는지 확인한다."""

    if not isinstance(settings, Mapping):
        raise TypeError("detector data paths must be a mapping")
    required = set(DEFAULT_DETECTOR_DATA_PATHS)
    missing = required - settings.keys()
    if missing:
        raise ValueError(f"데이터 경로 항목이 없습니다: {sorted(missing)}")
    unknown = settings.keys() - required
    if unknown:
        raise ValueError(f"알 수 없는 데이터 경로 항목입니다: {sorted(unknown)}")

    package_dir = settings["package_dir"]
    if not isinstance(package_dir, str | Path) or not str(package_dir).strip():
        raise ValueError("package_dir must be a non-empty path")
    image_roots = settings["image_roots"]
    if not isinstance(image_roots, Mapping) or not image_roots:
        raise ValueError("image_roots must be a non-empty mapping")
    for dataset, root in image_roots.items():
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("image_roots dataset names must be non-empty strings")
        if not isinstance(root, str | Path) or not str(root).strip():
            raise ValueError(f"image root must be a non-empty path: {dataset}")


def resolve_detector_data_paths(
    settings: Mapping[str, Any],
) -> tuple[Path, dict[str, Path]]:
    """설정의 상대 경로를 저장소 루트 기준 절대 경로로 바꾼다."""

    validate_detector_data_paths(settings)
    return resolve_repository_path(settings["package_dir"]), {
        name: resolve_repository_path(value)
        for name, value in settings["image_roots"].items()
    }


def resolve_detector_image_path(
    relative_name: str | Path,
    *,
    dataset: str,
    roots: Mapping[str, str | Path],
) -> Path:
    """dataset 이미지 루트와 안전한 상대 경로를 결합한다."""

    if dataset not in roots:
        raise ValueError(f"이미지 출처 설정이 없습니다: {dataset}")
    relative = PurePosixPath(str(relative_name).replace("\\", "/"))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or str(relative) in {"", "."}
    ):
        raise ValueError(
            f"이미지 경로는 데이터 루트 기준 상대 경로여야 합니다: {relative}"
        )
    root = Path(roots[dataset]).resolve()
    result = root.joinpath(*relative.parts).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"이미지 기준 폴더를 벗어난 경로입니다: {relative}")
    return result


def resolve_repository_path(value: str | Path) -> Path:
    """상대 경로를 저장소 루트 기준 절대 경로로 바꾼다."""

    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()
