from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from src.wrn.checkpoint import load_checkpoint, save_checkpoint_atomic
from src.wrn.optimization import build_formal_sgd
from src.wrn.schedule import apply_lr_for_epoch
from src.wrn.state import TrainingCursor
from src.wrn.training import build_classification_criterion, train_step


def _make(seed: int = 41) -> tuple[nn.Module, torch.optim.SGD]:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3)).double()
    return model, build_formal_sgd(model)


def _batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(8080)
    return [
        (
            torch.randn(3, 4, dtype=torch.float64, generator=generator),
            torch.tensor([(index + offset) % 3 for offset in range(3)]),
        )
        for index in range(5)
    ]


def _run(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cursor: TrainingCursor,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    criterion = build_classification_criterion()
    for inputs, targets in batches:
        train_step(model, optimizer, criterion, inputs, targets, cursor)


def _assert_nested_exact(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_exact(left_item, right_item)
    else:
        assert left == right


@pytest.mark.parametrize(
    ("start_cursor", "split"),
    [
        (TrainingCursor(), 2),
        (TrainingCursor(3, 0, 780), 2),
        (TrainingCursor(60, 0, 23010), 1),
    ],
)
def test_uninterrupted_equals_epoch_or_mid_epoch_resumed_exactly(
    tmp_path: Path, start_cursor: TrainingCursor, split: int
) -> None:
    batches = _batches()
    baseline_model, baseline_optimizer = _make()
    interrupted_model, interrupted_optimizer = _make()
    baseline_cursor = TrainingCursor.from_dict(start_cursor.to_dict())
    interrupted_cursor = TrainingCursor.from_dict(start_cursor.to_dict())
    apply_lr_for_epoch(baseline_optimizer, start_cursor.current_epoch_1)
    apply_lr_for_epoch(interrupted_optimizer, start_cursor.current_epoch_1)

    _run(baseline_model, baseline_optimizer, baseline_cursor, batches)
    _run(
        interrupted_model,
        interrupted_optimizer,
        interrupted_cursor,
        batches[:split],
    )
    path = tmp_path / f"resume-{start_cursor.current_epoch_1}-{split}.pt"
    save_checkpoint_atomic(
        path,
        interrupted_model,
        interrupted_optimizer,
        interrupted_cursor,
        run_seed=4,
    )
    resumed_model, resumed_optimizer = _make(seed=999)
    resumed_cursor, run_seed = load_checkpoint(
        path, resumed_model, resumed_optimizer
    )
    assert run_seed == 4
    _run(resumed_model, resumed_optimizer, resumed_cursor, batches[split:])

    _assert_nested_exact(baseline_model.state_dict(), resumed_model.state_dict())
    _assert_nested_exact(
        baseline_optimizer.state_dict(), resumed_optimizer.state_dict()
    )
    assert baseline_cursor == resumed_cursor
    assert baseline_optimizer.param_groups[0]["lr"] == resumed_optimizer.param_groups[0]["lr"]


@pytest.mark.parametrize(
    ("epoch", "update", "expected_lr"),
    [(59, 22620, 0.1), (60, 23010, 0.02), (61, 23400, 0.02), (120, 46410, 0.004), (160, 62010, 0.0008)],
)
def test_schedule_resume_milestone_cursor_lr(
    tmp_path: Path, epoch: int, update: int, expected_lr: float
) -> None:
    model, optimizer = _make(seed=epoch)
    apply_lr_for_epoch(optimizer, epoch)
    cursor = TrainingCursor(epoch, 0, update)
    path = tmp_path / f"milestone-{epoch}.pt"
    save_checkpoint_atomic(path, model, optimizer, cursor, run_seed=5)
    resumed_model, resumed_optimizer = _make(seed=999)
    restored, _ = load_checkpoint(path, resumed_model, resumed_optimizer)
    assert restored == cursor
    assert resumed_optimizer.param_groups[0]["lr"] == expected_lr
