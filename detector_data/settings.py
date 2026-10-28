"""작업 디렉터리와 관계없이 저장소 기준으로 데이터 경로를 찾는다."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = {
    "package_dir": "datasets/detector/training_package",
    "image_roots": {
        "select": "datasets/detector/select",
        "codeit": "datasets/detector/codeit_train_validated",
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
