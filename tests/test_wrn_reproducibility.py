from __future__ import annotations

import random

import numpy as np
import torch

from src.wrn.training import configure_reproducibility


def test_engine_reproducibility_settings_and_seed_replay() -> None:
    configure_reproducibility(123)
    first = (random.random(), float(np.random.random()), torch.rand(3))
    configure_reproducibility(123)
    second = (random.random(), float(np.random.random()), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    if hasattr(torch.backends.cuda.matmul, "fp32_precision") and hasattr(
        torch.backends.cudnn, "fp32_precision"
    ):
        assert torch.backends.cuda.matmul.fp32_precision == "ieee"
        assert torch.backends.cudnn.fp32_precision == "ieee"
    else:
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
