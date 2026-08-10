"""Fail-closed, atomic Phase 3 checkpoint serialization and resume."""

from __future__ import annotations

import os
import platform
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .optimization import assert_formal_parameter_group
from .schedule import lr_for_epoch
from .state import TrainingCursor


CHECKPOINT_FORMAT_VERSION = 1
ENGINE_VERSION = "phase3-v1"
FORMAL_TARGET_NAME = "WRN16-8-CIFAR10-ARXIV-V4-MEANSTD-NODROPOUT-V1"
FROZEN_CONFIG_SHA256 = (
    "18EF6815727ADF6816132954721194626950F4B40BD2606A5B7E2297EAB959B3"
)
EPOCH_DATA_POLICY = "zero_based_e0_to_seed_e0_plus_1_v1"


class CheckpointCompatibilityError(RuntimeError):
    """Raised when formal resume metadata does not exactly match."""


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise CheckpointCompatibilityError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def _optimizer_metadata(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    if len(optimizer.param_groups) != 1:
        raise CheckpointCompatibilityError("formal optimizer must have one group")
    group = optimizer.param_groups[0]
    return {
        "type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "momentum": group["momentum"],
        "dampening": group["dampening"],
        "weight_decay": group["weight_decay"],
        "nesterov": group["nesterov"],
        "maximize": group["maximize"],
        "foreach": group["foreach"],
        "differentiable": group["differentiable"],
        "fused": group["fused"],
        "parameter_groups": 1,
    }


def _model_signature(model: nn.Module) -> dict[str, tuple[tuple[int, ...], str]]:
    return {
        name: (tuple(tensor.shape), str(tensor.dtype))
        for name, tensor in model.state_dict().items()
    }


def build_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cursor: TrainingCursor,
    *,
    run_seed: int,
) -> dict[str, Any]:
    """Capture all formal state without selecting or recording test metrics."""

    cursor.validate()
    if run_seed not in (1, 2, 3, 4, 5):
        raise ValueError("run_seed must be one of the frozen seeds 1..5")
    assert_formal_parameter_group(model, optimizer, expected_lr=None)
    current_lrs = {float(group["lr"]) for group in optimizer.param_groups}
    if len(current_lrs) != 1:
        raise ValueError("formal optimizer must have one scalar current LR")
    current_lr = current_lrs.pop()
    expected_epoch = min(cursor.current_epoch_1, 200)
    if current_lr != lr_for_epoch(expected_epoch):
        raise ValueError("optimizer LR is stale for the checkpoint cursor")
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "engine_version": ENGINE_VERSION,
        "formal_target_name": FORMAL_TARGET_NAME,
        "frozen_config_sha256": FROZEN_CONFIG_SHA256,
        "model_signature": _model_signature(model),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_hyperparameters": _optimizer_metadata(optimizer),
        "training_cursor": cursor.to_dict(),
        "run_seed": int(run_seed),
        "epoch_data_policy_identifier": EPOCH_DATA_POLICY,
        "current_lr": current_lr,
        "rng_state": capture_rng_state(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }


def save_checkpoint_atomic(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cursor: TrainingCursor,
    *,
    run_seed: int,
) -> None:
    """Write in the destination directory and atomically replace on success."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_checkpoint(
        model, optimizer, cursor, run_seed=run_seed
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _read_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[TrainingCursor, int]:
    """Validate compatibility, restore objects, then restore RNG last."""

    payload = _read_checkpoint(path)
    required_matches = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "formal_target_name": FORMAL_TARGET_NAME,
        "frozen_config_sha256": FROZEN_CONFIG_SHA256,
        "epoch_data_policy_identifier": EPOCH_DATA_POLICY,
    }
    for name, expected in required_matches.items():
        if payload.get(name) != expected:
            raise CheckpointCompatibilityError(
                f"checkpoint {name} mismatch: {payload.get(name)!r} != {expected!r}"
            )
    if payload.get("model_signature") != _model_signature(model):
        raise CheckpointCompatibilityError("model architecture/state signature mismatch")
    try:
        assert_formal_parameter_group(model, optimizer, expected_lr=None)
    except (TypeError, ValueError) as error:
        raise CheckpointCompatibilityError(str(error)) from error
    if payload.get("optimizer_hyperparameters") != _optimizer_metadata(optimizer):
        raise CheckpointCompatibilityError("optimizer type/hyperparameters mismatch")

    try:
        cursor = TrainingCursor.from_dict(payload["training_cursor"])
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointCompatibilityError("invalid training cursor") from error
    expected_lr = lr_for_epoch(min(cursor.current_epoch_1, 200))
    if payload.get("current_lr") != expected_lr:
        raise CheckpointCompatibilityError("checkpoint LR is stale for its cursor")

    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    except (KeyError, RuntimeError, ValueError) as error:
        raise CheckpointCompatibilityError("state-dict restore failed") from error
    if {float(group["lr"]) for group in optimizer.param_groups} != {expected_lr}:
        raise CheckpointCompatibilityError("restored optimizer LR mismatch")
    try:
        assert_formal_parameter_group(model, optimizer, expected_lr=None)
    except (TypeError, ValueError) as error:
        raise CheckpointCompatibilityError(
            "restored optimizer hyperparameters mismatch"
        ) from error
    if _optimizer_metadata(optimizer) != payload["optimizer_hyperparameters"]:
        raise CheckpointCompatibilityError(
            "restored optimizer metadata mismatch"
        )
    run_seed = int(payload["run_seed"])
    if run_seed not in (1, 2, 3, 4, 5):
        raise CheckpointCompatibilityError("checkpoint run seed is not frozen")
    restore_rng_state(payload["rng_state"])
    return cursor, run_seed
