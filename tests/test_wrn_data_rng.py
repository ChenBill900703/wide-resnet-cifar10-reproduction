from __future__ import annotations

import pytest

from src.wrn.rng import (
    EpochShuffleSampler,
    epoch_data_seed,
    sample_data_seed,
    splitmix64,
)
from src.wrn.transforms import sample_augmentation_decision


def test_splitmix64_reference_vectors_lock_integer_mixing() -> None:
    assert splitmix64(0) == 0xE220A8397B1DCDAF
    assert splitmix64(1) == 0x910A2DEC89025CC1


@pytest.mark.parametrize(
    "zero_based_epoch,expected",
    [(0, 1), (59, 60), (119, 120), (159, 160), (199, 200)],
)
def test_frozen_epoch_seed_mapping(zero_based_epoch: int, expected: int) -> None:
    assert epoch_data_seed(zero_based_epoch) == expected


def test_invalid_epochs_are_rejected() -> None:
    for value in (-1, 200):
        with pytest.raises(ValueError, match=r"\[0, 199\]"):
            epoch_data_seed(value)


def test_same_epoch_and_sample_are_replayable_without_global_rng_state() -> None:
    expected = sample_augmentation_decision(data_seed=120, sample_index=37)
    for unrelated in range(100):
        sample_augmentation_decision(data_seed=1, sample_index=unrelated)
    assert sample_augmentation_decision(data_seed=120, sample_index=37) == expected
    assert sample_data_seed(120, 37) == sample_data_seed(120, 37)


def test_epoch_sampler_replays_full_order_and_changes_between_epochs() -> None:
    sampler = EpochShuffleSampler(range(257))
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(0)
    replay = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert first == replay
    assert first != second
    assert sorted(first) == list(range(257))
