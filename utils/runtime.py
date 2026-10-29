"""실행 장치, AMP와 random state를 관리하는 공통 함수."""

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

_REQUIRED_RNG_FIELDS = {"python", "numpy", "torch", "cuda", "mps"}


def select_device() -> torch.device:
    """CUDA, MPS, CPU 순서로 사용 가능한 장치를 선택한다."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_grad_scaler(device: torch.device) -> torch.amp.GradScaler:
    """CUDA에서만 활성화되는 AMP GradScaler를 생성한다."""

    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


def set_seed(seed: int) -> None:
    """Python, NumPy와 PyTorch의 random seed를 설정한다."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def capture_rng_state() -> dict[str, Any]:
    """현재 random number generator 상태를 checkpoint용으로 수집한다."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """checkpoint에 저장된 random number generator 상태를 복원한다."""

    missing_fields = sorted(_REQUIRED_RNG_FIELDS - state.keys())
    if missing_fields:
        raise ValueError(f"rng state is missing fields: {missing_fields}")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # checkpoint가 MPS/CUDA로 load되어도 RNG API에는 CPU ByteTensor를 전달한다.
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [device_state.cpu() for device_state in state["cuda"]]
        )
    if state["mps"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())
