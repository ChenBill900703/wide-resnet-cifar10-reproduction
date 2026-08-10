from __future__ import annotations

import pytest
import torch

from src.wrn.transforms import (
    CIFAR10_MEAN_255,
    CIFAR10_STD_255,
    normalize_cifar10_255,
)


def _channel_pixel(values: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).view(3, 1, 1)


def test_frozen_statistics_and_zero_difference() -> None:
    assert CIFAR10_MEAN_255 == (125.3, 123.0, 113.9)
    assert CIFAR10_STD_255 == (63.0, 62.1, 66.7)
    output = normalize_cifar10_255(_channel_pixel(CIFAR10_MEAN_255))
    torch.testing.assert_close(output, torch.zeros_like(output), atol=1e-7, rtol=0)


def test_positive_and_negative_one_standard_deviation() -> None:
    plus = tuple(m + s for m, s in zip(CIFAR10_MEAN_255, CIFAR10_STD_255))
    minus = tuple(m - s for m, s in zip(CIFAR10_MEAN_255, CIFAR10_STD_255))
    torch.testing.assert_close(
        normalize_cifar10_255(_channel_pixel(plus)),
        torch.ones(3, 1, 1),
        atol=2e-7,
        rtol=0,
    )
    torch.testing.assert_close(
        normalize_cifar10_255(_channel_pixel(minus)),
        -torch.ones(3, 1, 1),
        atol=2e-7,
        rtol=0,
    )


def test_scale_mismatch_is_detectable_and_unit_scale_equivalence_is_explicit() -> None:
    unit_scale_mean = _channel_pixel(tuple(value / 255 for value in CIFAR10_MEAN_255))
    wrong = normalize_cifar10_255(unit_scale_mean)
    assert torch.max(torch.abs(wrong)) > 1

    mean_unit = _channel_pixel(tuple(value / 255 for value in CIFAR10_MEAN_255))
    std_unit = _channel_pixel(tuple(value / 255 for value in CIFAR10_STD_255))
    mathematically_scaled = (unit_scale_mean - mean_unit) / std_unit
    torch.testing.assert_close(
        mathematically_scaled,
        torch.zeros_like(mathematically_scaled),
        atol=1e-7,
        rtol=0,
    )


def test_dtype_layout_and_rgb_channel_order_are_locked() -> None:
    rgb = torch.tensor([125.3, 123.0, 113.9], dtype=torch.float64).view(3, 1, 1)
    output = normalize_cifar10_255(rgb)
    assert output.dtype == torch.float32
    assert output.shape == (3, 1, 1)

    bgr = rgb.flip(0)
    assert not torch.allclose(normalize_cifar10_255(bgr), torch.zeros_like(output))
    with pytest.raises(ValueError, match="CxHxW"):
        normalize_cifar10_255(torch.zeros(32, 32, 3))


def test_out_of_byte_scale_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="0..255"):
        normalize_cifar10_255(torch.full((3, 1, 1), 256.0))
