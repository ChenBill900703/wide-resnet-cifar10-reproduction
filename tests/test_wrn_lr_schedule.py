from __future__ import annotations

import pytest
import torch
from torch import nn

from src.wrn.schedule import (
    TOTAL_UPDATES,
    apply_lr_for_epoch,
    end_update,
    lr_for_epoch,
    lr_for_global_update,
)


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [
        (1, 0.1),
        (2, 0.1),
        (59, 0.1),
        (60, 0.02),
        (61, 0.02),
        (119, 0.02),
        (120, 0.004),
        (121, 0.004),
        (159, 0.004),
        (160, 0.0008),
        (161, 0.0008),
        (200, 0.0008),
    ],
)
def test_lr_epoch_boundary_matrix(epoch: int, expected: float) -> None:
    assert lr_for_epoch(epoch) == expected


@pytest.mark.parametrize("epoch", [0, 201])
def test_invalid_epoch_rejected(epoch: int) -> None:
    with pytest.raises(ValueError):
        lr_for_epoch(epoch)


def test_apply_lr_updates_group_at_epoch_start() -> None:
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    assert apply_lr_for_epoch(optimizer, 60) == 0.02
    assert optimizer.param_groups[0]["lr"] == 0.02


def test_derived_epoch_end_update_mapping() -> None:
    expected = {
        1: 390,
        59: 23010,
        60: 23400,
        119: 46410,
        120: 46800,
        159: 62010,
        160: 62400,
        200: 78000,
    }
    assert {epoch: end_update(epoch) for epoch in expected} == expected
    assert TOTAL_UPDATES == 78000


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        (1, 0.1),
        (23010, 0.1),
        (23011, 0.02),
        (46410, 0.02),
        (46411, 0.004),
        (62010, 0.004),
        (62011, 0.0008),
        (78000, 0.0008),
    ],
)
def test_derived_global_update_lr_ranges(update: int, expected: float) -> None:
    assert lr_for_global_update(update) == expected


@pytest.mark.parametrize("update", [0, 78001])
def test_invalid_global_update_rejected(update: int) -> None:
    with pytest.raises(ValueError):
        lr_for_global_update(update)
