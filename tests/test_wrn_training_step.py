from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from src.wrn.optimization import build_formal_sgd
from src.wrn.state import TrainingCursor
from src.wrn.training import (
    NonFiniteTrainingError,
    build_classification_criterion,
    prepare_epoch,
    train_step,
)


def _tiny_model() -> nn.Module:
    return nn.Linear(4, 3).double()


def test_mean_cross_entropy_matches_manual_log_softmax_nll() -> None:
    logits = torch.tensor(
        [[1.0, -0.5, 0.25], [-0.2, 0.3, 1.2]], dtype=torch.float64
    )
    targets = torch.tensor([0, 2])
    criterion = build_classification_criterion()
    actual = criterion(logits, targets)
    expected = -torch.log_softmax(logits, dim=1)[range(2), targets].mean()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert criterion.reduction == "mean"
    assert criterion.label_smoothing == 0.0
    assert criterion.weight is None


def test_train_step_clears_gradients_and_increments_once() -> None:
    torch.manual_seed(11)
    model = _tiny_model()
    optimizer = build_formal_sgd(model)
    criterion = build_classification_criterion()
    batch1 = (torch.randn(3, 4, dtype=torch.float64), torch.tensor([0, 1, 2]))
    batch2 = (torch.randn(3, 4, dtype=torch.float64), torch.tensor([2, 0, 1]))
    cursor = TrainingCursor()
    train_step(model, optimizer, criterion, *batch1, cursor)
    after_step1 = copy.deepcopy(model)
    train_step(model, optimizer, criterion, *batch2, cursor)
    actual_step2_grad = model.weight.grad.detach().clone()

    independent = after_step1
    independent.zero_grad(set_to_none=True)
    criterion(independent(batch2[0]), batch2[1]).backward()
    torch.testing.assert_close(
        actual_step2_grad, independent.weight.grad, rtol=0, atol=0
    )
    assert cursor.global_update == 2
    assert cursor.next_batch_index_0 == 2


def test_nonfinite_loss_blocks_step_and_counter() -> None:
    model = _tiny_model()
    optimizer = build_formal_sgd(model)
    cursor = TrainingCursor()
    before = copy.deepcopy(model.state_dict())
    inputs = torch.full((2, 4), float("nan"), dtype=torch.float64)
    with pytest.raises(NonFiniteTrainingError, match="loss"):
        train_step(
            model,
            optimizer,
            build_classification_criterion(),
            inputs,
            torch.tensor([0, 1]),
            cursor,
        )
    assert cursor.global_update == 0
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_forward_failure_does_not_increment_counter() -> None:
    class FailingForward(nn.Linear):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("controlled forward failure")

    model = FailingForward(4, 3).double()
    optimizer = build_formal_sgd(model)
    cursor = TrainingCursor()
    with pytest.raises(RuntimeError, match="controlled forward failure"):
        train_step(
            model,
            optimizer,
            build_classification_criterion(),
            torch.ones(2, 4, dtype=torch.float64),
            torch.tensor([0, 1]),
            cursor,
        )
    assert cursor.global_update == 0


def test_nonfinite_gradient_blocks_step_and_counter() -> None:
    model = _tiny_model()
    optimizer = build_formal_sgd(model)
    cursor = TrainingCursor()
    before = copy.deepcopy(model.state_dict())
    handle = model.weight.register_hook(lambda gradient: gradient * float("nan"))
    with pytest.raises(NonFiniteTrainingError, match="non-finite gradients"):
        train_step(
            model,
            optimizer,
            build_classification_criterion(),
            torch.ones(2, 4, dtype=torch.float64),
            torch.tensor([0, 1]),
            cursor,
        )
    handle.remove()
    assert cursor.global_update == 0
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_missing_gradient_is_a_blocker_and_does_not_step() -> None:
    class ModelWithUnusedParameter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 3).double()
            self.unused = nn.Parameter(torch.ones((), dtype=torch.float64))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.linear(inputs)

    model = ModelWithUnusedParameter()
    optimizer = build_formal_sgd(model)
    cursor = TrainingCursor()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="missing gradients: unused"):
        train_step(
            model,
            optimizer,
            build_classification_criterion(),
            torch.ones(2, 4, dtype=torch.float64),
            torch.tensor([0, 1]),
            cursor,
        )
    assert cursor.global_update == 0
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_optimizer_failure_does_not_increment_counter() -> None:
    model = _tiny_model()
    optimizer = build_formal_sgd(model)
    cursor = TrainingCursor()

    def fail_step(*args: object, **kwargs: object) -> None:
        raise RuntimeError("controlled optimizer failure")

    optimizer.step = fail_step  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="controlled optimizer failure"):
        train_step(
            model,
            optimizer,
            build_classification_criterion(),
            torch.ones(2, 4, dtype=torch.float64),
            torch.tensor([0, 1]),
            cursor,
        )
    assert cursor.global_update == 0


class _EpochSpy:
    def __init__(self) -> None:
        self.zero_based_epoch: int | None = None

    def set_epoch(self, zero_based_epoch: int) -> None:
        self.zero_based_epoch = zero_based_epoch


@pytest.mark.parametrize(
    ("epoch", "update", "expected_lr", "expected_seed"),
    [(60, 23010, 0.02, 60), (120, 46410, 0.004, 120), (160, 62010, 0.0008, 160)],
)
def test_prepare_epoch_integrates_lr_and_phase2_data_seed(
    epoch: int, update: int, expected_lr: float, expected_seed: int
) -> None:
    model = _tiny_model()
    optimizer = build_formal_sgd(model)
    spy = _EpochSpy()
    actual_lr, actual_seed = prepare_epoch(
        optimizer, spy, TrainingCursor(epoch, 0, update)
    )
    assert actual_lr == expected_lr
    assert actual_seed == expected_seed
    assert spy.zero_based_epoch == epoch - 1
