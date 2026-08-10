"""Explicit frozen epoch learning-rate schedule and update cross-checks."""

from __future__ import annotations

import torch


EPOCHS = 200
UPDATES_PER_EPOCH = 390
TOTAL_UPDATES = EPOCHS * UPDATES_PER_EPOCH


def lr_for_epoch(epoch_1: int) -> float:
    """Return the LR applied at the start of a one-based epoch."""

    if not 1 <= epoch_1 <= EPOCHS:
        raise ValueError("epoch_1 must be in [1, 200]")
    if epoch_1 < 60:
        return 0.1
    if epoch_1 < 120:
        return 0.02
    if epoch_1 < 160:
        return 0.004
    return 0.0008


def apply_lr_for_epoch(
    optimizer: torch.optim.Optimizer, epoch_1: int
) -> float:
    """Apply exactly one scalar LR to every formal parameter group."""

    learning_rate = lr_for_epoch(epoch_1)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def end_update(epoch_1: int) -> int:
    """Derived number of completed updates at an epoch end."""

    if not 1 <= epoch_1 <= EPOCHS:
        raise ValueError("epoch_1 must be in [1, 200]")
    return epoch_1 * UPDATES_PER_EPOCH


def lr_for_global_update(update_1: int) -> float:
    """Derived cross-check only; the formal schedule source remains epoch."""

    if not 1 <= update_1 <= TOTAL_UPDATES:
        raise ValueError("update_1 must be in [1, 78000]")
    completed_epochs_before_update = (update_1 - 1) // UPDATES_PER_EPOCH
    return lr_for_epoch(completed_epochs_before_update + 1)
