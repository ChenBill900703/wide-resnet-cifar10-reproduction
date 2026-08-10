from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.wrn.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    EPOCH_DATA_POLICY,
    FORMAL_TARGET_NAME,
    FROZEN_CONFIG_SHA256,
    CheckpointCompatibilityError,
    load_checkpoint,
    save_checkpoint_atomic,
)
from src.wrn.optimization import build_formal_sgd
from src.wrn.schedule import apply_lr_for_epoch
from src.wrn.state import TrainingCursor
from src.wrn.training import build_classification_criterion, train_step


def _make_model_optimizer(seed: int = 7) -> tuple[nn.Module, torch.optim.SGD]:
    torch.manual_seed(seed)
    model = nn.Linear(4, 3).double()
    return model, build_formal_sgd(model)


def _one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cursor: TrainingCursor,
) -> None:
    train_step(
        model,
        optimizer,
        build_classification_criterion(),
        torch.tensor(
            [[0.1, 0.2, -0.3, 0.4], [-0.2, 0.5, 0.7, -0.1]],
            dtype=torch.float64,
        ),
        torch.tensor([0, 2]),
        cursor,
    )


def test_checkpoint_schema_model_optimizer_cursor_and_rng_restore(
    tmp_path: Path,
) -> None:
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    model, optimizer = _make_model_optimizer(seed=17)
    cursor = TrainingCursor()
    _one_step(model, optimizer, cursor)
    saved_model = copy.deepcopy(model.state_dict())
    saved_buffer = copy.deepcopy(optimizer.state_dict())["state"][0]["momentum_buffer"]
    path = tmp_path / "phase3-test.pt"
    save_checkpoint_atomic(path, model, optimizer, cursor, run_seed=3)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert {
        "format_version",
        "engine_version",
        "formal_target_name",
        "frozen_config_sha256",
        "model_signature",
        "model_state_dict",
        "optimizer_state_dict",
        "optimizer_hyperparameters",
        "training_cursor",
        "run_seed",
        "epoch_data_policy_identifier",
        "current_lr",
        "rng_state",
        "environment",
    } <= payload.keys()
    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert payload["formal_target_name"] == FORMAL_TARGET_NAME
    assert payload["frozen_config_sha256"] == FROZEN_CONFIG_SHA256
    assert payload["epoch_data_policy_identifier"] == EPOCH_DATA_POLICY

    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)
    resumed_model, resumed_optimizer = _make_model_optimizer(seed=999)
    resumed_cursor, run_seed = load_checkpoint(
        path, resumed_model, resumed_optimizer
    )
    assert run_seed == 3
    assert resumed_cursor == cursor
    for name, tensor in resumed_model.state_dict().items():
        assert torch.equal(tensor, saved_model[name])
    resumed_buffer = resumed_optimizer.state_dict()["state"][0]["momentum_buffer"]
    assert torch.equal(resumed_buffer, saved_buffer)
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("formal_target_name", "WRONG", "formal_target_name mismatch"),
        ("frozen_config_sha256", "0" * 64, "frozen_config_sha256 mismatch"),
        ("format_version", 999, "format_version mismatch"),
        ("engine_version", "wrong", "engine_version mismatch"),
    ],
)
def test_checkpoint_identity_tamper_fails_closed(
    tmp_path: Path, field: str, bad_value: object, message: str
) -> None:
    model, optimizer = _make_model_optimizer()
    path = tmp_path / "valid.pt"
    save_checkpoint_atomic(path, model, optimizer, TrainingCursor(), run_seed=1)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload[field] = bad_value
    tampered = tmp_path / f"tampered-{field}.pt"
    torch.save(payload, tampered)
    new_model, new_optimizer = _make_model_optimizer(seed=99)
    with pytest.raises(CheckpointCompatibilityError, match=message):
        load_checkpoint(tampered, new_model, new_optimizer)


def test_wrong_optimizer_metadata_and_model_architecture_rejected(
    tmp_path: Path,
) -> None:
    model, optimizer = _make_model_optimizer()
    path = tmp_path / "valid.pt"
    save_checkpoint_atomic(path, model, optimizer, TrainingCursor(), run_seed=1)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["optimizer_hyperparameters"]["weight_decay"] = 0.0
    tampered = tmp_path / "wrong-optimizer.pt"
    torch.save(payload, tampered)
    new_model, new_optimizer = _make_model_optimizer(seed=99)
    with pytest.raises(CheckpointCompatibilityError, match="optimizer"):
        load_checkpoint(tampered, new_model, new_optimizer)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["optimizer_state_dict"]["param_groups"][0]["weight_decay"] = 0.0
    tampered_state = tmp_path / "wrong-optimizer-state.pt"
    torch.save(payload, tampered_state)
    with pytest.raises(CheckpointCompatibilityError, match="restored optimizer"):
        load_checkpoint(tampered_state, new_model, new_optimizer)

    wrong_model = nn.Linear(4, 4).double()
    wrong_optimizer = build_formal_sgd(wrong_model)
    with pytest.raises(CheckpointCompatibilityError, match="architecture"):
        load_checkpoint(path, wrong_model, wrong_optimizer)


def test_checkpoint_rejects_nonfrozen_run_seed(tmp_path: Path) -> None:
    model, optimizer = _make_model_optimizer()
    with pytest.raises(ValueError, match="frozen seeds"):
        save_checkpoint_atomic(
            tmp_path / "bad-seed.pt",
            model,
            optimizer,
            TrainingCursor(),
            run_seed=99,
        )


def test_inconsistent_cursor_and_stale_schedule_lr_rejected(tmp_path: Path) -> None:
    model, optimizer = _make_model_optimizer()
    path = tmp_path / "valid.pt"
    save_checkpoint_atomic(path, model, optimizer, TrainingCursor(), run_seed=1)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["training_cursor"] = {
        "current_epoch_1": 60,
        "next_batch_index_0": 0,
        "global_update": 23011,
    }
    inconsistent = tmp_path / "inconsistent.pt"
    torch.save(payload, inconsistent)
    new_model, new_optimizer = _make_model_optimizer(seed=99)
    with pytest.raises(CheckpointCompatibilityError, match="cursor"):
        load_checkpoint(inconsistent, new_model, new_optimizer)

    payload["training_cursor"]["global_update"] = 23010
    payload["current_lr"] = 0.1
    stale = tmp_path / "stale-lr.pt"
    torch.save(payload, stale)
    with pytest.raises(CheckpointCompatibilityError, match="stale"):
        load_checkpoint(stale, new_model, new_optimizer)


@pytest.mark.parametrize(
    ("completed_epoch", "next_epoch", "update", "lr"),
    [
        (60, 61, 23400, 0.02),
        (120, 121, 46800, 0.004),
        (160, 161, 62400, 0.0008),
        (200, 201, 78000, 0.0008),
    ],
)
def test_epoch_checkpoint_policy_is_after_completed_epoch(
    tmp_path: Path,
    completed_epoch: int,
    next_epoch: int,
    update: int,
    lr: float,
) -> None:
    model, optimizer = _make_model_optimizer(seed=completed_epoch)
    apply_lr_for_epoch(optimizer, completed_epoch)
    cursor = TrainingCursor(next_epoch, 0, update)
    path = tmp_path / f"after-{completed_epoch}.pt"
    save_checkpoint_atomic(path, model, optimizer, cursor, run_seed=1)
    resumed_model, resumed_optimizer = _make_model_optimizer(seed=999)
    restored, _ = load_checkpoint(path, resumed_model, resumed_optimizer)
    assert restored == cursor
    assert resumed_optimizer.param_groups[0]["lr"] == lr
