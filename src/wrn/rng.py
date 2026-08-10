"""Deterministic data RNG primitives for the frozen CIFAR-10 pipeline."""

from __future__ import annotations

from collections.abc import Iterator, Sized

import torch
from torch.utils.data import Sampler


_MASK_64 = (1 << 64) - 1
_MASK_63 = (1 << 63) - 1


def epoch_data_seed(zero_based_epoch: int) -> int:
    """Map zero-based epochs 0..199 to the frozen data seeds 1..200."""

    if not 0 <= zero_based_epoch < 200:
        raise ValueError("zero_based_epoch must be in [0, 199]")
    return zero_based_epoch + 1


def splitmix64(value: int) -> int:
    """Return the version-stable SplitMix64 output for an integer input."""

    value = (value + 0x9E3779B97F4A7C15) & _MASK_64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
    return (value ^ (value >> 31)) & _MASK_64


def sample_data_seed(data_seed: int, sample_index: int) -> int:
    """Derive a stable per-sample seed independent of process scheduling."""

    if data_seed < 0:
        raise ValueError("data_seed must be non-negative")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    combined = (
        (data_seed & _MASK_64)
        ^ (((sample_index + 1) * 0xD2B74407B1CE6E93) & _MASK_64)
    )
    return splitmix64(combined) & _MASK_63


class EpochShuffleSampler(Sampler[int]):
    """Shuffle deterministically from the frozen epoch data seed."""

    def __init__(self, data_source: Sized) -> None:
        self._data_source = data_source
        self._zero_based_epoch = 0

    @property
    def zero_based_epoch(self) -> int:
        return self._zero_based_epoch

    def set_epoch(self, zero_based_epoch: int) -> None:
        epoch_data_seed(zero_based_epoch)
        self._zero_based_epoch = zero_based_epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(epoch_data_seed(self._zero_based_epoch))
        yield from torch.randperm(len(self), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self._data_source)
