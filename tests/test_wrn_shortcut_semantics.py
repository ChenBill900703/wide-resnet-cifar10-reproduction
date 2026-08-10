from __future__ import annotations

import torch

from src.wrn import WideBasicBlock


def _capture_input(module: torch.nn.Module, captured: dict[str, torch.Tensor], key: str):
    return module.register_forward_pre_hook(
        lambda _module, inputs: captured.__setitem__(key, inputs[0].detach().clone())
    )


def test_dimension_changing_projection_and_residual_share_preactivation() -> None:
    block = WideBasicBlock(3, 5, stride=2).eval()
    captured: dict[str, torch.Tensor] = {}
    handles = [
        _capture_input(block.conv1, captured, "residual_input"),
        _capture_input(block.projection, captured, "projection_input"),
    ]
    x = torch.linspace(-2.0, 2.0, steps=2 * 3 * 8 * 8).reshape(2, 3, 8, 8)

    with torch.no_grad():
        expected = block.relu1(block.bn1(x))
        block(x)
    for handle in handles:
        handle.remove()

    torch.testing.assert_close(captured["residual_input"], expected)
    torch.testing.assert_close(captured["projection_input"], expected)
    assert not torch.equal(captured["projection_input"], x)


def test_identity_uses_raw_x_and_no_post_add_relu() -> None:
    block = WideBasicBlock(3, 3, stride=1).eval()
    with torch.no_grad():
        block.conv1.weight.zero_()
        block.conv2.weight.zero_()
    x = -torch.ones(2, 3, 8, 8)

    with torch.no_grad():
        output = block(x)

    # A raw identity and absent post-add ReLU preserve the negative tensor.
    torch.testing.assert_close(output, x)
    assert (output < 0).all()


def test_residual_branch_starts_from_preactivation_in_identity_block() -> None:
    block = WideBasicBlock(3, 3, stride=1).eval()
    captured: dict[str, torch.Tensor] = {}
    handle = _capture_input(block.conv1, captured, "residual_input")
    x = torch.randn(2, 3, 8, 8) - 1.0

    with torch.no_grad():
        expected = block.relu1(block.bn1(x))
        block(x)
    handle.remove()

    torch.testing.assert_close(captured["residual_input"], expected)


def test_block_execution_order_is_bn_relu_conv_twice() -> None:
    block = WideBasicBlock(3, 3, stride=1).eval()
    events: list[str] = []
    handles = []
    for name in ("bn1", "relu1", "conv1", "bn2", "relu2", "conv2"):
        handles.append(
            getattr(block, name).register_forward_hook(
                lambda _module, _inputs, _output, name=name: events.append(name)
            )
        )

    with torch.no_grad():
        block(torch.randn(2, 3, 8, 8))
    for handle in handles:
        handle.remove()

    assert events == ["bn1", "relu1", "conv1", "bn2", "relu2", "conv2"]
