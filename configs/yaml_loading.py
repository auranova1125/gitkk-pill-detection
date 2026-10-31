"""작업별 YAML을 설정 dataclass로 변환할 때 쓰는 공통 검증 함수."""

from collections.abc import Mapping
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

import yaml

DEVICE_NAMES = {"auto", "cpu", "mps", "cuda"}


def load_task_yaml(path: Path, *, expected_task: str) -> dict[str, Any]:
    """YAML 루트와 task 값을 검증한 뒤 일반 dict로 반환한다."""

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("config root must be a mapping")
    if any(not isinstance(key, str) for key in raw):
        raise TypeError("config keys must be strings")
    values = dict(raw)
    if values.get("task") != expected_task:
        raise ValueError(f"config task must be {expected_task!r}")
    return values


def build_dataclass[ConfigT](
    config_type: type[ConfigT],
    value: Any,
    *,
    context: str,
    extra_values: Mapping[str, Any] | None = None,
) -> ConfigT:
    """mapping을 필드 검증 후 지정한 dataclass로 변환한다."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    combined = dict(value)
    if extra_values is not None:
        combined.update(extra_values)
    allowed = {field.name for field in fields(config_type)}
    required = {
        field.name
        for field in fields(config_type)
        if field.default is MISSING and field.default_factory is MISSING
    }
    validate_keys(combined, allowed=allowed, required=required, context=context)
    return config_type(**combined)


def validate_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    context: str,
) -> None:
    """알 수 없는 필드와 필수 필드 누락을 검사한다."""

    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {unknown}")
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}")


def validate_factory_path(value: str | None, *, name: str) -> None:
    """동적 factory 경로가 module:function 형식인지 검사한다."""

    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or null")
    if not value.strip():
        raise ValueError(f"{name} must not be empty; use null when unavailable")
    module_name, separator, function_name = value.partition(":")
    if separator != ":" or not module_name or not function_name or ":" in function_name:
        raise ValueError(f"{name} must use 'module:function' format")


def validate_device_name(value: str) -> None:
    """지원하는 실행 장치 이름인지 검사한다."""

    if not isinstance(value, str):
        raise TypeError("device must be a string")
    if value.strip().lower() not in DEVICE_NAMES:
        choices = ", ".join(sorted(DEVICE_NAMES))
        raise ValueError(f"device must be one of: {choices}")


def validate_positive_integer(value: int, *, name: str) -> None:
    """bool이 아닌 양의 정수인지 검사한다."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def validate_boolean(value: bool, *, name: str) -> None:
    """실제 bool 값인지 검사한다."""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
