"""Frozen CIFAR-10 preprocessing and augmentation semantics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .rng import sample_data_seed


CIFAR10_MEAN_255 = (125.3, 123.0, 113.9)
CIFAR10_STD_255 = (63.0, 62.1, 66.7)
CIFAR10_REFLECTION_PADDING = 4
CIFAR10_CROP_SIZE = 32
CIFAR10_MAX_CROP_OFFSET = 8


def _require_chw_rgb(image: Tensor) -> None:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("expected RGB image in CxHxW layout")


def float_255(image: Tensor) -> Tensor:
    """Convert to float32 without changing the frozen 0..255 scale."""

    _require_chw_rgb(image)
    result = image.to(dtype=torch.float32)
    if not torch.isfinite(result).all():
        raise ValueError("image contains non-finite values")
    if result.numel() and (result.min() < 0 or result.max() > 255):
        raise ValueError("pre-normalization image must remain on the 0..255 scale")
    return result


def normalize_cifar10_255(image: Tensor) -> Tensor:
    """Normalize RGB CxHxW input with the frozen byte-scale statistics."""

    result = float_255(image)
    mean = result.new_tensor(CIFAR10_MEAN_255).view(3, 1, 1)
    std = result.new_tensor(CIFAR10_STD_255).view(3, 1, 1)
    return (result - mean) / std


def horizontal_flip(image: Tensor) -> Tensor:
    """Flip a CxHxW image along its width axis."""

    _require_chw_rgb(image)
    return image.flip(-1)


def reflection_pad_4(image: Tensor) -> Tensor:
    """Apply four-sided edge-excluding reflection padding."""

    _require_chw_rgb(image)
    return F.pad(
        image.unsqueeze(0),
        (CIFAR10_REFLECTION_PADDING,) * 4,
        mode="reflect",
    ).squeeze(0)


def crop_32(image: Tensor, *, x_offset: int, y_offset: int) -> Tensor:
    """Crop 32x32 using frozen zero-based width/height offsets 0..8."""

    _require_chw_rgb(image)
    if image.shape[-2:] != (40, 40):
        raise ValueError("crop input must be the 40x40 padded image")
    if not 0 <= x_offset <= CIFAR10_MAX_CROP_OFFSET:
        raise ValueError("x_offset must be in [0, 8]")
    if not 0 <= y_offset <= CIFAR10_MAX_CROP_OFFSET:
        raise ValueError("y_offset must be in [0, 8]")
    return image[
        :,
        y_offset : y_offset + CIFAR10_CROP_SIZE,
        x_offset : x_offset + CIFAR10_CROP_SIZE,
    ]


@dataclass(frozen=True)
class AugmentationDecision:
    """One replayable historical-branch augmentation decision."""

    flip: bool
    x_offset: int
    y_offset: int


def sample_augmentation_decision(
    *, data_seed: int, sample_index: int
) -> AugmentationDecision:
    """Sample flip and independent x/y offsets from a per-sample generator."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(sample_data_seed(data_seed, sample_index))
    flip_draw = int(torch.randint(0, 2, (), generator=generator).item())
    x_offset = int(torch.randint(0, 9, (), generator=generator).item())
    y_offset = int(torch.randint(0, 9, (), generator=generator).item())
    return AugmentationDecision(
        flip=flip_draw == 0,
        x_offset=x_offset,
        y_offset=y_offset,
    )


class CIFAR10TrainTransform:
    """Flip -> reflect-pad -> crop -> float32 byte-scale normalization."""

    def __call__(self, image: Tensor, *, data_seed: int, sample_index: int) -> Tensor:
        decision = sample_augmentation_decision(
            data_seed=data_seed,
            sample_index=sample_index,
        )
        return self.apply_with_decision(image, decision)

    @staticmethod
    def apply_with_decision(
        image: Tensor, decision: AugmentationDecision
    ) -> Tensor:
        if decision.flip:
            image = horizontal_flip(image)
        image = reflection_pad_4(image)
        image = crop_32(
            image,
            x_offset=decision.x_offset,
            y_offset=decision.y_offset,
        )
        return normalize_cifar10_255(image)


class CIFAR10TestTransform:
    """Deterministic frozen mean/std normalization without augmentation."""

    def __call__(self, image: Tensor) -> Tensor:
        return normalize_cifar10_255(image)
