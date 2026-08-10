from __future__ import annotations

import pytest
import torch
from torch import nn

from src.wrn.model import wrn16_8
from src.wrn.optimization import (
    DAMPENING,
    INITIAL_LR,
    MOMENTUM,
    NESTEROV,
    WEIGHT_DECAY,
    analytical_sgd_step,
    assert_formal_parameter_group,
    assert_full_gradient_coverage,
    build_formal_sgd,
)
from src.wrn.training import build_classification_criterion


def test_formal_sgd_hyperparameters_and_exact_parameter_set() -> None:
    model = wrn16_8()
    optimizer = build_formal_sgd(model)
    group = optimizer.param_groups[0]

    assert type(optimizer) is torch.optim.SGD
    assert len(optimizer.param_groups) == 1
    assert group["lr"] == INITIAL_LR
    assert group["momentum"] == MOMENTUM
    assert group["dampening"] == DAMPENING
    assert group["weight_decay"] == WEIGHT_DECAY
    assert group["nesterov"] is NESTEROV
    assert group["foreach"] is False
    assert group["fused"] is False
    assert group["maximize"] is False
    assert group["differentiable"] is False
    assert_formal_parameter_group(model, optimizer)

    parameter_ids = {id(parameter) for parameter in group["params"]}
    named = dict(model.named_parameters())
    for name in (
        "stem.weight",
        "group1.0.bn1.weight",
        "group1.0.bn1.bias",
        "classifier.weight",
        "classifier.bias",
    ):
        assert id(named[name]) in parameter_ids
    assert all(id(buffer) not in parameter_ids for buffer in model.buffers())


def test_parameter_audit_rejects_bn_or_bias_exclusion() -> None:
    model = wrn16_8()
    excluded = [
        parameter
        for name, parameter in model.named_parameters()
        if "bn" not in name and not name.endswith("bias")
    ]
    optimizer = torch.optim.SGD(
        excluded,
        lr=INITIAL_LR,
        momentum=MOMENTUM,
        dampening=DAMPENING,
        weight_decay=WEIGHT_DECAY,
        nesterov=NESTEROV,
        foreach=False,
        fused=False,
    )
    with pytest.raises(ValueError, match="exactly match"):
        assert_formal_parameter_group(model, optimizer)


def test_float64_first_and_second_step_match_independent_equation() -> None:
    parameter = nn.Parameter(torch.tensor([1.25, -0.75], dtype=torch.float64))
    optimizer = torch.optim.SGD(
        [parameter],
        lr=INITIAL_LR,
        momentum=MOMENTUM,
        dampening=DAMPENING,
        weight_decay=WEIGHT_DECAY,
        nesterov=NESTEROV,
        foreach=False,
        fused=False,
    )
    manual_parameter = parameter.detach().clone()
    manual_buffer = None

    for gradient in (
        torch.tensor([0.4, -0.2], dtype=torch.float64),
        torch.tensor([-0.1, 0.3], dtype=torch.float64),
    ):
        expected, manual_buffer, _ = analytical_sgd_step(
            manual_parameter, gradient, manual_buffer
        )
        parameter.grad = gradient.clone()
        optimizer.step()
        torch.testing.assert_close(parameter, expected, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(
            optimizer.state[parameter]["momentum_buffer"],
            manual_buffer,
            rtol=1e-14,
            atol=1e-14,
        )
        manual_parameter = expected


def test_weight_decay_is_coupled_before_momentum_not_decoupled() -> None:
    parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    raw_gradient = torch.tensor([0.3], dtype=torch.float64)
    expected, _, _ = analytical_sgd_step(parameter.detach(), raw_gradient, None)
    optimizer = torch.optim.SGD(
        [parameter],
        lr=INITIAL_LR,
        momentum=MOMENTUM,
        dampening=DAMPENING,
        weight_decay=WEIGHT_DECAY,
        nesterov=NESTEROV,
        foreach=False,
        fused=False,
    )
    parameter.grad = raw_gradient.clone()
    optimizer.step()
    decoupled_candidate = (
        torch.tensor([2.0], dtype=torch.float64) * (1 - INITIAL_LR * WEIGHT_DECAY)
        - INITIAL_LR * (raw_gradient + MOMENTUM * raw_gradient)
    )
    torch.testing.assert_close(parameter, expected, rtol=1e-14, atol=1e-14)
    assert not torch.equal(parameter, decoupled_candidate)


def test_actual_wrn_first_step_matches_manual_conv_bn_and_fc_elements() -> None:
    torch.manual_seed(314159)
    model = wrn16_8()
    optimizer = build_formal_sgd(model)
    criterion = build_classification_criterion()
    inputs = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([2, 7])

    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(inputs), targets)
    loss.backward()
    assert_full_gradient_coverage(model)

    names = (
        "stem.weight",
        "group1.0.bn1.weight",
        "group1.0.bn1.bias",
        "classifier.weight",
        "classifier.bias",
    )
    parameters = dict(model.named_parameters())
    expected: dict[str, torch.Tensor] = {}
    for name in names:
        parameter = parameters[name]
        flat_parameter = parameter.detach().reshape(-1)
        flat_gradient = parameter.grad.detach().reshape(-1)
        updated, _, _ = analytical_sgd_step(
            flat_parameter[0], flat_gradient[0], None
        )
        expected[name] = updated

    optimizer.step()
    for name in names:
        torch.testing.assert_close(
            parameters[name].detach().reshape(-1)[0],
            expected[name],
            rtol=2e-5,
            atol=2e-7,
        )
