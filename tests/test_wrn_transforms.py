from __future__ import annotations

import pytest
import torch

from src.wrn.transforms import (
    AugmentationDecision,
    CIFAR10TrainTransform,
    crop_32,
    horizontal_flip,
    reflection_pad_4,
    sample_augmentation_decision,
)


def test_reflection_padding_matches_manual_edge_excluding_indices() -> None:
    raw = torch.arange(3 * 5 * 6, dtype=torch.float32).reshape(3, 5, 6)
    actual = reflection_pad_4(raw)

    def reflected(position: int, size: int) -> int:
        source = position - 4
        if source < 0:
            return -source
        if source >= size:
            return 2 * size - source - 2
        return source

    expected = torch.empty(3, 13, 14)
    for y in range(13):
        for x in range(14):
            expected[:, y, x] = raw[:, reflected(y, 5), reflected(x, 6)]
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual[:, 3, 4:10], raw[:, 1, :])
    assert not torch.equal(actual[:, 3, 4:10], raw[:, 0, :])


@pytest.mark.parametrize("x_offset,y_offset", [(0, 0), (8, 8), (3, 7)])
def test_crop_uses_exact_zero_based_coordinates(x_offset: int, y_offset: int) -> None:
    coordinate = (
        torch.arange(40).view(1, 40, 1) * 100
        + torch.arange(40).view(1, 1, 40)
    ).expand(3, -1, -1)
    crop = crop_32(coordinate, x_offset=x_offset, y_offset=y_offset)
    assert crop.shape == (3, 32, 32)
    assert int(crop[0, 0, 0]) == y_offset * 100 + x_offset
    assert int(crop[0, -1, -1]) == (y_offset + 31) * 100 + x_offset + 31


def test_crop_rejects_offset_nine() -> None:
    image = torch.zeros(3, 40, 40)
    with pytest.raises(ValueError, match="x_offset"):
        crop_32(image, x_offset=9, y_offset=0)
    with pytest.raises(ValueError, match="y_offset"):
        crop_32(image, x_offset=0, y_offset=9)


def test_historical_flip_branch_zero_flips_and_one_is_unchanged() -> None:
    image = torch.arange(3 * 32 * 32, dtype=torch.uint8).reshape(3, 32, 32)
    transform = CIFAR10TrainTransform()
    flipped = transform.apply_with_decision(
        image, AugmentationDecision(flip=True, x_offset=4, y_offset=4)
    )
    unchanged = transform.apply_with_decision(
        image, AugmentationDecision(flip=False, x_offset=4, y_offset=4)
    )
    assert not torch.equal(flipped, unchanged)
    torch.testing.assert_close(horizontal_flip(horizontal_flip(image)), image)


def test_transform_order_is_proven_by_asymmetric_tensor_output() -> None:
    image = torch.zeros(3, 32, 32, dtype=torch.uint8)
    image[0] = torch.arange(32, dtype=torch.uint8).view(1, 32).expand(32, -1)
    image[1] = torch.arange(32, dtype=torch.uint8).view(32, 1).expand(-1, 32)
    image[2, 3, 5] = 251
    decision = AugmentationDecision(flip=True, x_offset=1, y_offset=7)

    actual = CIFAR10TrainTransform.apply_with_decision(image, decision)
    manual_correct = crop_32(
        reflection_pad_4(horizontal_flip(image)), x_offset=1, y_offset=7
    ).float()
    mean = torch.tensor((125.3, 123.0, 113.9)).view(3, 1, 1)
    std = torch.tensor((63.0, 62.1, 66.7)).view(3, 1, 1)
    manual_correct = (manual_correct - mean) / std
    torch.testing.assert_close(actual, manual_correct)

    wrong_order = horizontal_flip(
        crop_32(reflection_pad_4(image), x_offset=1, y_offset=7)
    ).float()
    wrong_order = (wrong_order - mean) / std
    assert not torch.equal(actual, wrong_order)


def test_sampled_flip_and_independent_offsets_cover_frozen_domains() -> None:
    decisions = [
        sample_augmentation_decision(data_seed=60, sample_index=index)
        for index in range(2000)
    ]
    assert {decision.flip for decision in decisions} == {False, True}
    xs = [decision.x_offset for decision in decisions]
    ys = [decision.y_offset for decision in decisions]
    assert min(xs) == min(ys) == 0
    assert max(xs) == max(ys) == 8
    assert all(0 <= value <= 8 for value in xs + ys)
    assert any(x != y for x, y in zip(xs, ys))
