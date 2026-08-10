from __future__ import annotations

import pytest
import torch
from torch import nn

from src.wrn import WideBasicBlock, WideResNet, wrn16_8


def test_frozen_depth_mapping_and_widths() -> None:
    model = wrn16_8()

    assert model.depth == 16
    assert model.widen_factor == 8
    assert model.blocks_per_group == 2
    assert model.stage_channels == (16, 128, 256, 512)
    assert [len(model.group1), len(model.group2), len(model.group3)] == [2, 2, 2]


@pytest.mark.parametrize("depth", [3, 15, 17, 29])
def test_invalid_depth_mapping_is_rejected(depth: int) -> None:
    with pytest.raises(ValueError, match="depth must satisfy"):
        WideResNet(depth=depth, widen_factor=8)


def test_nonzero_dropout_is_rejected_and_no_dropout_module_exists() -> None:
    with pytest.raises(ValueError, match="requires dropout=0"):
        WideResNet(dropout=0.3)

    assert not any(isinstance(module, nn.Dropout) for module in wrn16_8().modules())


def test_projection_locations_and_strides() -> None:
    model = wrn16_8()
    first_blocks = [model.group1[0], model.group2[0], model.group3[0]]
    second_blocks = [model.group1[1], model.group2[1], model.group3[1]]

    assert all(isinstance(block, WideBasicBlock) for block in first_blocks)
    assert [block.conv1.stride for block in first_blocks] == [(1, 1), (2, 2), (2, 2)]
    assert [block.projection.stride for block in first_blocks] == [
        (1, 1),
        (2, 2),
        (2, 2),
    ]
    assert all(block.projection is None for block in second_blocks)


def test_stage_shapes_and_logits_shape() -> None:
    model = wrn16_8().eval()
    observed: dict[str, tuple[int, ...]] = {}

    handles = []
    for name in ("stem", "group1", "group2", "group3"):
        module = getattr(model, name)
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, name=name: observed.__setitem__(
                    name, tuple(output.shape)
                )
            )
        )

    with torch.no_grad():
        logits = model(torch.randn(2, 3, 32, 32))
    for handle in handles:
        handle.remove()

    assert observed == {
        "stem": (2, 16, 32, 32),
        "group1": (2, 128, 32, 32),
        "group2": (2, 256, 16, 16),
        "group3": (2, 512, 8, 8),
    }
    assert logits.shape == (2, 10)


def test_synthetic_forward_backward_is_finite() -> None:
    torch.manual_seed(7)
    model = wrn16_8()
    inputs = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([1, 9])

    loss = nn.CrossEntropyLoss()(model(inputs), targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.stem.weight.grad is not None
    assert torch.isfinite(model.stem.weight.grad).all()
    assert model.classifier.weight.grad is not None
    assert torch.isfinite(model.classifier.weight.grad).all()
