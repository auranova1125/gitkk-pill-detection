"""설정 문자열에서 데이터 구성 함수를 안전하게 불러온다."""

from collections.abc import Callable
from importlib import import_module
from typing import Any


def load_data_factory(
    factory_path: str | None,
    *,
    setting_name: str = "data_factory_path",
) -> Callable[..., Any]:
    """`module:function` 경로를 import 가능한 callable로 변환한다."""

    if factory_path is None:
        raise RuntimeError(
            f"{setting_name} is not configured; set a module:function path"
        )
    if not isinstance(factory_path, str):
        raise TypeError(f"{setting_name} must be a string or null")
    module_name, separator, function_name = factory_path.partition(":")
    if separator != ":" or not module_name or not function_name or ":" in function_name:
        raise ValueError(f"{setting_name} must use 'module:function' format")

    module = import_module(module_name)
    try:
        factory = getattr(module, function_name)
    except AttributeError as error:
        raise ImportError(
            f"{factory_path!r} does not reference an existing attribute"
        ) from error
    if not callable(factory):
        raise TypeError(f"{factory_path!r} must reference a callable")
    return factory
