"""설정 문자열에서 데이터 factory를 불러오고 공통 반환 계약을 검증한다."""

from collections.abc import Callable, Iterable, Mapping
from importlib import import_module
from typing import Any

TrainingFactoryResult = tuple[
    Iterable[Mapping[str, Any]],
    Iterable[Mapping[str, Any]],
    Mapping[str, Any],
    Mapping[str, Any],
]


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


def call_training_data_factory(
    factory_path: str | None,
    *,
    config: Any,
    device: Any,
) -> TrainingFactoryResult:
    """학습 factory를 호출하고 공통 4개 반환값 계약을 검증한다."""

    value = load_data_factory(factory_path)(config=config, device=device)
    if not isinstance(value, tuple) or len(value) != 4:
        raise TypeError(
            "training data factory must return "
            "(train_loader, val_loader, metadata, data_summary)"
        )
    train_loader, val_loader, metadata, data_summary = value
    if not isinstance(metadata, Mapping) or not isinstance(data_summary, Mapping):
        raise TypeError("factory metadata and data summary must be mappings")
    return train_loader, val_loader, metadata, data_summary


def first_batch(loader: Iterable[Any], *, name: str) -> Any:
    """DataLoader의 첫 batch를 반환하고 비어 있으면 명확한 오류를 낸다."""

    try:
        return next(iter(loader))
    except StopIteration as error:
        raise ValueError(f"{name} must contain at least one batch") from error
