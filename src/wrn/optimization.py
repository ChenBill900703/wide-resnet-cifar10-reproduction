"""Frozen SGD construction and parameter/gradient audits."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn


INITIAL_LR = 0.1
MOMENTUM = 0.9
DAMPENING = 0.0
WEIGHT_DECAY = 0.0005
NESTEROV = True


class MissingGradientError(RuntimeError):
    """A formal trainable parameter did not participate in backward."""


class NonFiniteGradientError(RuntimeError):
    """A formal trainable parameter has a NaN or Inf gradient."""


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return the complete ordered learnable parameter vector."""

    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def build_formal_sgd(model: nn.Module) -> torch.optim.SGD:
    """Build the frozen ordinary, non-foreach, non-fused SGD path."""

    optimizer = torch.optim.SGD(
        trainable_parameters(model),
        lr=INITIAL_LR,
        momentum=MOMENTUM,
        dampening=DAMPENING,
        weight_decay=WEIGHT_DECAY,
        nesterov=NESTEROV,
        maximize=False,
        foreach=False,
        differentiable=False,
        fused=False,
    )
    assert_formal_parameter_group(model, optimizer)
    return optimizer


def _optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> Iterable[nn.Parameter]:
    for group in optimizer.param_groups:
        yield from group["params"]


def assert_formal_parameter_group(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_lr: float | None = INITIAL_LR,
) -> None:
    """Fail unless one group exactly covers every trainable parameter once."""

    if type(optimizer) is not torch.optim.SGD:
        raise TypeError("formal optimizer must be exactly torch.optim.SGD")
    if len(optimizer.param_groups) != 1:
        raise ValueError("formal optimizer must have exactly one parameter group")

    expected = trainable_parameters(model)
    actual = list(_optimizer_parameters(optimizer))
    actual_ids = [id(parameter) for parameter in actual]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("optimizer contains duplicate parameters")
    if set(actual_ids) != {id(parameter) for parameter in expected}:
        raise ValueError("optimizer parameter IDs do not exactly match the model")

    group = optimizer.param_groups[0]
    expected_values = {
        "momentum": MOMENTUM,
        "dampening": DAMPENING,
        "weight_decay": WEIGHT_DECAY,
        "nesterov": NESTEROV,
        "maximize": False,
        "foreach": False,
        "differentiable": False,
        "fused": False,
    }
    for name, expected_value in expected_values.items():
        if group[name] != expected_value:
            raise ValueError(
                f"formal optimizer {name}={group[name]!r}, expected {expected_value!r}"
            )
    if expected_lr is not None and group["lr"] != expected_lr:
        raise ValueError(
            f"formal optimizer lr={group['lr']!r}, expected {expected_lr!r}"
        )


def assert_full_gradient_coverage(model: nn.Module) -> None:
    """Require a present, finite gradient for every trainable parameter."""

    missing: list[str] = []
    nonfinite: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            nonfinite.append(name)
    if missing:
        raise MissingGradientError(f"missing gradients: {', '.join(missing)}")
    if nonfinite:
        raise NonFiniteGradientError(
            f"non-finite gradients: {', '.join(nonfinite)}"
        )


def analytical_sgd_step(
    parameter: Tensor,
    raw_gradient: Tensor,
    momentum_buffer: Tensor | None,
    *,
    lr: float = INITIAL_LR,
    momentum: float = MOMENTUM,
    dampening: float = DAMPENING,
    weight_decay: float = WEIGHT_DECAY,
) -> tuple[Tensor, Tensor, Tensor]:
    """Independent Torch7/PyTorch coupled-decay Nesterov equation."""

    decayed_gradient = raw_gradient + weight_decay * parameter
    if momentum_buffer is None:
        new_buffer = decayed_gradient.clone()
    else:
        new_buffer = momentum * momentum_buffer + (1.0 - dampening) * decayed_gradient
    direction = decayed_gradient + momentum * new_buffer
    updated = parameter - lr * direction
    return updated, new_buffer, direction
