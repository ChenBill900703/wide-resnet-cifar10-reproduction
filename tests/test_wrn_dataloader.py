from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torchvision.datasets import CIFAR10

from src.wrn.data import build_cifar10_loaders


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


def _require_data() -> None:
    if not (DATA_ROOT / CIFAR10.base_folder).is_dir():
        pytest.skip("local audited CIFAR-10 artifact is not present")


def test_loader_batch_contract_and_full_test_coverage() -> None:
    _require_data()
    loaders = build_cifar10_loaders(DATA_ROOT, num_workers=0, download=False)
    loaders.set_epoch(0)
    assert loaders.train.batch_size == 128
    assert loaders.train.drop_last is True
    assert loaders.test.batch_size == 128
    assert loaders.test.drop_last is False
    assert len(loaders.train) == 390
    assert len(loaders.test) == 79

    train_samples = sum(images.shape[0] for images, _labels in loaders.train)
    assert train_samples == 49_920
    test_labels = torch.cat([labels for _images, labels in loaders.test])
    assert test_labels.numel() == 10_000
    assert test_labels.tolist() == loaders.test_dataset.targets


def test_loader_epoch_replay_covers_order_and_transformed_tensor() -> None:
    _require_data()
    loaders = build_cifar10_loaders(DATA_ROOT, num_workers=0, download=False)
    loaders.set_epoch(59)
    order = list(loaders.train_sampler)
    images, labels = next(iter(loaders.train))
    loaders.set_epoch(59)
    replay_order = list(loaders.train_sampler)
    replay_images, replay_labels = next(iter(loaders.train))
    assert order == replay_order
    torch.testing.assert_close(images, replay_images, rtol=0, atol=0)
    torch.testing.assert_close(labels, replay_labels, rtol=0, atol=0)
    assert images.shape == (128, 3, 32, 32)
    assert images.dtype == torch.float32
    assert torch.isfinite(images).all()


def test_worker_count_does_not_change_order_or_per_sample_augmentation() -> None:
    _require_data()
    single = build_cifar10_loaders(DATA_ROOT, num_workers=0, download=False)
    multi = build_cifar10_loaders(DATA_ROOT, num_workers=2, download=False)
    single.set_epoch(119)
    multi.set_epoch(119)
    single_images, single_labels = next(iter(single.train))
    multi_images, multi_labels = next(iter(multi.train))
    torch.testing.assert_close(single_labels, multi_labels, rtol=0, atol=0)
    torch.testing.assert_close(single_images, multi_images, rtol=0, atol=0)
