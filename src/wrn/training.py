"""Auditable one-batch synthetic training-engine semantics."""

from __future__ import annotations

import random
from typing import Protocol

import numpy as np
import torch
from torch import Tensor, nn

from .optimization import NonFiniteGradientError, assert_full_gradient_coverage
from .rng import epoch_data_seed
from .schedule import apply_lr_for_epoch
from .state import TrainingCursor


class EpochDataControl(Protocol):
    def set_epoch(self, zero_based_epoch: int) -> None: ...


class NonFiniteTrainingError(RuntimeError):
    """Raised before optimizer.step when loss or gradients are non-finite."""


def build_classification_criterion() -> nn.CrossEntropyLoss:
    """Historical sizeAverage=true port: unweighted mean CE on logits."""

    return nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)


def configure_reproducibility(run_seed: int) -> None:
    """Apply the frozen deterministic engine-side CPU/CUDA controls."""

    if run_seed < 0:
        raise ValueError("run_seed must be non-negative")
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "fp32_precision") and hasattr(
        torch.backends.cudnn, "fp32_precision"
    ):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def prepare_epoch(
    optimizer: torch.optim.Optimizer,
    data_control: EpochDataControl,
    cursor: TrainingCursor,
) -> tuple[float, int]:
    """Apply LR then the Phase 2 zero-based epoch control before iteration."""

    cursor.validate()
    if cursor.complete or cursor.next_batch_index_0 != 0:
        raise ValueError("prepare_epoch requires a nonterminal epoch-start cursor")
    learning_rate = apply_lr_for_epoch(optimizer, cursor.current_epoch_1)
    zero_based_epoch = cursor.current_epoch_1 - 1
    data_control.set_epoch(zero_based_epoch)
    return learning_rate, epoch_data_seed(zero_based_epoch)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    cursor: TrainingCursor,
    *,
    strict_gradients: bool = True,
) -> float:
    """Perform exactly one successful optimizer update on synthetic inputs."""

    cursor.validate()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, targets)
    if loss.numel() != 1 or not torch.isfinite(loss).all():
        raise NonFiniteTrainingError("loss is NaN or Inf")
    loss.backward()
    try:
        if strict_gradients:
            assert_full_gradient_coverage(model)
        else:
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("non-finite gradients")
    except NonFiniteGradientError as error:
        raise NonFiniteTrainingError(str(error)) from error
    optimizer.step()
    cursor.record_successful_update()
    return float(loss.detach().item())
