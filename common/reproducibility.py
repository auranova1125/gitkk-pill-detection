"""재현 가능한 실행을 위한 random seed 설정."""

import random

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
