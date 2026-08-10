from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision.datasets import CIFAR10

from src.wrn.data import FrozenCIFAR10Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


def _require_data() -> None:
    if not (DATA_ROOT / CIFAR10.base_folder).is_dir():
        pytest.skip("local audited CIFAR-10 artifact is not present")


@pytest.fixture(scope="module")
def datasets() -> tuple[FrozenCIFAR10Dataset, FrozenCIFAR10Dataset]:
    _require_data()
    return (
        FrozenCIFAR10Dataset(DATA_ROOT, train=True, download=False),
        FrozenCIFAR10Dataset(DATA_ROOT, train=False, download=False),
    )


def test_actual_dataset_integrity_sizes_classes_and_layout(datasets) -> None:
    train, test = datasets
    assert train.integrity_ok() and test.integrity_ok()
    assert len(train) == 50_000
    assert len(test) == 10_000
    assert train.raw_shape == (50_000, 32, 32, 3)
    assert test.raw_shape == (10_000, 32, 32, 3)
    assert train.raw_dtype == test.raw_dtype == "uint8"
    assert train.classes == [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    class_identity = hashlib.sha256(
        json.dumps(train.classes, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert class_identity == "feb528d8ad763e4c89a96910f8ed9ea0113aa59fe89528e3f8804b4ae9b0d2e5"
    assert all(isinstance(label, int) and 0 <= label < 10 for label in train.targets)
    assert all(isinstance(label, int) and 0 <= label < 10 for label in test.targets)


def test_installed_torchvision_split_file_contract() -> None:
    assert CIFAR10.train_list == [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]
    assert CIFAR10.test_list == [
        ["test_batch", "40351d587109b95175f43aff81a1287e"]
    ]
    assert CIFAR10.meta == {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }


def test_test_dataset_is_deterministic_and_unaugmented(datasets) -> None:
    _, test = datasets
    first_image, first_label = test[123]
    second_image, second_label = test[123]
    torch.testing.assert_close(first_image, second_image, rtol=0, atol=0)
    assert first_label == second_label == test.targets[123]
    assert first_image.shape == (3, 32, 32)
    assert first_image.dtype == torch.float32
    assert torch.isfinite(first_image).all()


def test_train_item_replays_within_epoch_and_can_change_across_epochs(datasets) -> None:
    train, _ = datasets
    train.set_epoch(0)
    first = [train[index][0] for index in range(10)]
    replay = [train[index][0] for index in range(10)]
    for actual, expected in zip(replay, first):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    train.set_epoch(1)
    second = [train[index][0] for index in range(10)]
    assert any(not torch.equal(a, b) for a, b in zip(first, second))


def test_raw_train_and_test_arrays_are_separate_objects(datasets) -> None:
    train, test = datasets
    assert train._base.train is True
    assert test._base.train is False
    assert not np.shares_memory(train._base.data, test._base.data)
