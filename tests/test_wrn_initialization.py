from __future__ import annotations

import math

import torch
from torch import nn

from src.wrn import wrn16_8


def test_convolution_initialization_uses_frozen_fan_in_formula(monkeypatch) -> None:
    calls: list[tuple[torch.Size, float, float]] = []
    original = nn.init.normal_

    def recording_normal_(tensor, mean=0.0, std=1.0, generator=None):
        calls.append((tensor.shape, mean, std))
        return original(tensor, mean=mean, std=std, generator=generator)

    monkeypatch.setattr(nn.init, "normal_", recording_normal_)
    model = wrn16_8()
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]

    assert len(calls) == len(convolutions)
    for (shape, mean, observed_std), convolution in zip(calls, convolutions):
        kernel_height, kernel_width = convolution.kernel_size
        expected_std = math.sqrt(
            2.0 / (kernel_width * kernel_height * convolution.in_channels)
        )
        assert shape == convolution.weight.shape
        assert mean == 0.0
        assert observed_std == expected_std
        assert convolution.bias is None


def test_batch_norm_and_classifier_state_match_frozen_policy() -> None:
    model = wrn16_8()
    batch_norms = [
        module for module in model.modules() if isinstance(module, nn.BatchNorm2d)
    ]

    assert batch_norms
    for batch_norm in batch_norms:
        assert batch_norm.affine
        assert batch_norm.eps == 1e-5
        assert batch_norm.momentum == 0.1
        assert torch.all((batch_norm.weight >= 0.0) & (batch_norm.weight < 1.0))
        assert torch.count_nonzero(batch_norm.bias) == 0
        assert torch.count_nonzero(batch_norm.running_mean) == 0
        assert torch.equal(batch_norm.running_var, torch.ones_like(batch_norm.running_var))

    bound = 1.0 / math.sqrt(model.classifier.in_features)
    assert torch.all(model.classifier.weight >= -bound)
    assert torch.all(model.classifier.weight <= bound)
    assert torch.count_nonzero(model.classifier.bias) == 0


def test_initialization_is_replayable_from_model_seed() -> None:
    torch.manual_seed(123)
    first = wrn16_8().state_dict()
    torch.manual_seed(123)
    second = wrn16_8().state_dict()

    assert first.keys() == second.keys()
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0.0, atol=0.0)
