"""CIFAR-10 datasets and loaders for Phase 2 data-pipeline validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

from .rng import EpochShuffleSampler, epoch_data_seed
from .transforms import CIFAR10TestTransform, CIFAR10TrainTransform


CIFAR10_BATCH_SIZE = 128


class FrozenCIFAR10Dataset(Dataset[tuple[Tensor, int]]):
    """Wrap torchvision CIFAR-10 with explicit frozen tensor transforms."""

    def __init__(self, root: str | Path, *, train: bool, download: bool = False) -> None:
        self._base = CIFAR10(
            root=str(root),
            train=train,
            transform=None,
            target_transform=None,
            download=download,
        )
        self.train = train
        self._train_transform = CIFAR10TrainTransform()
        self._test_transform = CIFAR10TestTransform()
        self._zero_based_epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    @property
    def classes(self) -> list[str]:
        return list(self._base.classes)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return dict(self._base.class_to_idx)

    @property
    def targets(self) -> list[int]:
        return self._base.targets

    @property
    def raw_shape(self) -> tuple[int, ...]:
        return tuple(self._base.data.shape)

    @property
    def raw_dtype(self) -> str:
        return str(self._base.data.dtype)

    @property
    def zero_based_epoch(self) -> int:
        return int(self._zero_based_epoch.item())

    def set_epoch(self, zero_based_epoch: int) -> None:
        epoch_data_seed(zero_based_epoch)
        self._zero_based_epoch.fill_(zero_based_epoch)

    def integrity_ok(self) -> bool:
        return bool(self._base._check_integrity())

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        raw_hwc = self._base.data[index]
        raw_chw = torch.from_numpy(raw_hwc).permute(2, 0, 1).contiguous()
        target = int(self._base.targets[index])
        if self.train:
            image = self._train_transform(
                raw_chw,
                data_seed=epoch_data_seed(self.zero_based_epoch),
                sample_index=index,
            )
        else:
            image = self._test_transform(raw_chw)
        return image, target


@dataclass
class CIFAR10DataLoaders:
    """Train/test loaders plus the explicit epoch control surface."""

    train_dataset: FrozenCIFAR10Dataset
    test_dataset: FrozenCIFAR10Dataset
    train_sampler: EpochShuffleSampler
    train: DataLoader[tuple[Tensor, Tensor]]
    test: DataLoader[tuple[Tensor, Tensor]]

    def set_epoch(self, zero_based_epoch: int) -> None:
        self.train_dataset.set_epoch(zero_based_epoch)
        self.train_sampler.set_epoch(zero_based_epoch)


def build_cifar10_loaders(
    root: str | Path,
    *,
    num_workers: int = 0,
    download: bool = False,
) -> CIFAR10DataLoaders:
    """Build the frozen train/test loaders without hidden download defaults."""

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    train_dataset = FrozenCIFAR10Dataset(root, train=True, download=download)
    test_dataset = FrozenCIFAR10Dataset(root, train=False, download=False)
    train_sampler = EpochShuffleSampler(train_dataset)
    worker_generator = torch.Generator(device="cpu").manual_seed(0)
    common = {
        "batch_size": CIFAR10_BATCH_SIZE,
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": False,
        "generator": worker_generator,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=False,
        drop_last=True,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return CIFAR10DataLoaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_sampler=train_sampler,
        train=train_loader,
        test=test_loader,
    )
