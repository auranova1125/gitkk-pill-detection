"""재현 가능한 실행을 위한 random seed 설정."""

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Python, NumPy, PyTorch의 random seed를 설정한다."""

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


def restore_rng_state(state: dict[str, Any]) -> None:
    """checkpoint에 저장된 random number generator 상태를 복원한다."""

    required_fields = {"python", "numpy", "torch", "cuda", "mps"}
    missing_fields = sorted(required_fields - state.keys())
    if missing_fields:
        raise ValueError(f"rng state is missing fields: {missing_fields}")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["mps"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
