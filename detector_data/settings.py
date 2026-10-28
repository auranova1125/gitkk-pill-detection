"""작업 디렉터리와 관계없이 저장소 기준으로 데이터 경로를 찾는다."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path, PurePosixPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = {
    "package_dir": "datasets",
    "image_roots": {
        "select": "datasets/images/select_aihub",
        "codeit": "datasets/images/select_codeit",
    },
    "augmentation": {
        "horizontal_flip": 0.0,
        "vertical_flip": 0.0,
        "brightness": 0.0,
        "contrast": 0.0,
    },
}


def repository_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def load_settings():
    """기본 설정을 사용한다. 외부 경로는 환경 변수로 명시한 경우에만 읽는다."""
    custom = os.environ.get("DETECTOR_DATA_SETTINGS")
    if not custom:
        return copy.deepcopy(DEFAULT_SETTINGS)
    path = repository_path(custom)
    if not path.is_file():
        raise FileNotFoundError(f"지정한 데이터 설정 파일이 없습니다: {path}")
    with path.open(encoding="utf-8") as file:
        settings = json.load(file)
    for key in DEFAULT_SETTINGS:
        if key not in settings:
            raise ValueError(f"데이터 설정 항목이 없습니다: {key}")
    if set(settings["image_roots"]) != {"select", "codeit"}:
        raise ValueError("image_roots에는 select와 codeit을 지정해야 합니다.")
    return settings


def resolve_data_paths(settings):
    return repository_path(settings["package_dir"]), {
        name: repository_path(value)
        for name, value in settings["image_roots"].items()
    }


def resolve_image_path(portable_name, roots):
    """기존 JSON의 출처별 경로를 이미지 파일만 모은 새 폴더에 연결한다."""
    portable = PurePosixPath(str(portable_name).replace("\\", "/"))
    if (
        portable.is_absolute()
        or ".." in portable.parts
        or len(portable.parts) < 3
        or portable.parts[1] != "images"
    ):
        raise ValueError(f"이미지 경로는 출처/images/파일명 형식이어야 합니다: {portable}")
    dataset = portable.parts[0]
    if dataset not in roots:
        raise ValueError(f"이미지 출처 설정이 없습니다: {dataset}")
    root = Path(roots[dataset]).resolve()
    # roots는 images의 상위 폴더가 아닌 실제 이미지 파일이 있는 폴더다.
    result = root.joinpath(*portable.parts[2:]).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"이미지 기준 폴더를 벗어난 경로입니다: {portable}")
    return result
