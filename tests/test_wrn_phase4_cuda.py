from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from src.wrn.checkpoint import load_checkpoint, save_checkpoint_atomic
from src.wrn.cuda_validation import (
    EXPECTED_TRAINABLE_PARAMETERS,
    TARGET_GPU_NAME,
    assert_optimizer_state_cuda,
    audit_model_cuda_device,
    canonical_training_fingerprint,
    cifar_forward_only,
    configure_target_runtime,
    has_nominal_eight_gb_vram,
    numerical_policy_snapshot,
    require_target_cuda_device,
    target_environment_snapshot,
)
from src.wrn.data import build_cifar10_loaders
from src.wrn.model import wrn16_8
from src.wrn.optimization import (
    INITIAL_LR,
    MOMENTUM,
    WEIGHT_DECAY,
    assert_full_gradient_coverage,
    build_formal_sgd,
)
from src.wrn.schedule import TOTAL_UPDATES, end_update, lr_for_epoch
from src.wrn.state import TrainingCursor
from src.wrn.training import build_classification_criterion, train_step


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
CUDA_UNAVAILABLE = not torch.cuda.is_available()
TARGET_CUDA_REQUIRED = os.environ.get("WRN_REQUIRE_TARGET_CUDA") == "1"
pytestmark = pytest.mark.skipif(
    CUDA_UNAVAILABLE and not TARGET_CUDA_REQUIRED,
    reason="Phase 4 requires the actual CUDA-enabled RTX 3070 Ti runtime",
)


@pytest.fixture
def target_device() -> torch.device:
    device = require_target_cuda_device(0)
    torch.cuda.empty_cache()
    return device


def _make_cuda_model(
    seed: int, device: torch.device
) -> tuple[nn.Module, torch.optim.SGD, TrainingCursor]:
    configure_target_runtime(seed)
    model = wrn16_8().to(device)
    optimizer = build_formal_sgd(model)
    return model, optimizer, TrainingCursor()


def _random_cuda_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cursor: TrainingCursor,
    device: torch.device,
) -> None:
    inputs = torch.randn(2, 3, 32, 32, device=device, dtype=torch.float32)
    labels = torch.randint(0, 10, (2,), device=device, dtype=torch.int64)
    train_step(
        model,
        optimizer,
        build_classification_criterion(),
        inputs,
        labels,
        cursor,
    )


def _assert_nested_exact(left: Any, right: Any) -> None:
    if isinstance(left, Tensor):
        assert isinstance(right, Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_exact(left_item, right_item)
    else:
        assert left == right


def test_target_environment_and_numerical_policy_gate(
    target_device: torch.device,
) -> None:
    configure_target_runtime(1)
    environment = target_environment_snapshot(target_device)
    policy = numerical_policy_snapshot()
    model = wrn16_8().to(target_device)

    assert environment["cuda_available"] is True
    assert environment["cuda_device_count"] >= 1
    assert environment["selected_device_index"] == 0
    assert environment["current_device_index"] == 0
    assert environment["gpu_name"] == TARGET_GPU_NAME
    assert environment["compute_capability"] == [8, 6]
    assert has_nominal_eight_gb_vram(environment["gpu_total_vram_bytes"])
    assert environment["torch_cuda_build"] is not None
    assert environment["cudnn"] is not None
    assert environment["nvidia_driver"]
    assert policy["default_dtype"] == "torch.float32"
    assert policy["autocast_cuda_enabled"] is False
    assert policy["tf32_enabled"] is False
    assert policy["cudnn_benchmark"] is False
    assert policy["cudnn_deterministic"] is True
    assert policy["deterministic_algorithms"] is True
    assert policy["deterministic_warn_only"] is False
    assert not hasattr(model, "_orig_mod")
    assert lr_for_epoch(59) == 0.1
    assert lr_for_epoch(60) == 0.02
    assert lr_for_epoch(120) == 0.004
    assert lr_for_epoch(160) == 0.0008
    assert lr_for_epoch(200) == 0.0008
    assert end_update(1) == 390
    assert TOTAL_UPDATES == 78_000
    print("PHASE4_ENVIRONMENT=" + json.dumps(environment, sort_keys=True))
    print("PHASE4_POLICY=" + json.dumps(policy, sort_keys=True))


def test_cuda_model_device_parameter_count_and_seed_replay(
    target_device: torch.device,
) -> None:
    fingerprints: dict[int, str] = {}
    for seed in range(1, 6):
        model, optimizer, cursor = _make_cuda_model(seed, target_device)
        assert audit_model_cuda_device(model, target_device) == (
            EXPECTED_TRAINABLE_PARAMETERS
        )
        fingerprints[seed] = canonical_training_fingerprint(
            model, optimizer, cursor.to_dict()
        )
        del model, optimizer
    assert len(set(fingerprints.values())) == 5

    replay_model, replay_optimizer, replay_cursor = _make_cuda_model(
        1, target_device
    )
    replay = canonical_training_fingerprint(
        replay_model, replay_optimizer, replay_cursor.to_dict()
    )
    assert replay == fingerprints[1]


def test_cuda_synthetic_backward_and_first_step_semantics(
    target_device: torch.device,
) -> None:
    model, optimizer, _cursor = _make_cuda_model(2, target_device)
    criterion = build_classification_criterion()
    inputs = torch.randn(
        2, 3, 32, 32, device=target_device, dtype=torch.float32
    )
    labels = torch.tensor([2, 7], device=target_device, dtype=torch.int64)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, labels)
    assert logits.dtype == torch.float32
    assert inputs.dtype == torch.float32
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    loss.backward()
    assert_full_gradient_coverage(model)
    assert all(
        parameter.grad is not None
        and parameter.grad.device == target_device
        and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    representative_names = (
        "stem.weight",
        "group1.0.bn1.weight",
        "group1.0.bn1.bias",
        "classifier.weight",
        "classifier.bias",
    )
    named_parameters = dict(model.named_parameters())
    expected: dict[str, Tensor] = {}
    for name in representative_names:
        parameter = named_parameters[name]
        value = parameter.detach().reshape(-1)[0].clone()
        raw_gradient = parameter.grad.detach().reshape(-1)[0].clone()
        decayed_gradient = raw_gradient + WEIGHT_DECAY * value
        first_buffer = decayed_gradient
        direction = decayed_gradient + MOMENTUM * first_buffer
        expected[name] = value - INITIAL_LR * direction

    optimizer.step()
    assert_optimizer_state_cuda(optimizer, target_device)
    for name in representative_names:
        torch.testing.assert_close(
            named_parameters[name].detach().reshape(-1)[0],
            expected[name],
            rtol=2e-5,
            atol=2e-7,
        )
        momentum_buffer = optimizer.state[named_parameters[name]][
            "momentum_buffer"
        ]
        assert momentum_buffer.device == target_device


def test_cuda_rng_low_level_and_checkpoint_restore(
    target_device: torch.device, tmp_path: Path
) -> None:
    configure_target_runtime(3)
    torch.cuda.manual_seed(3)
    _first = torch.rand(8, device=target_device)
    low_level_state = torch.cuda.get_rng_state(target_device)
    expected_low_level = torch.rand(8, device=target_device)
    torch.cuda.set_rng_state(low_level_state, target_device)
    actual_low_level = torch.rand(8, device=target_device)
    assert torch.equal(expected_low_level, actual_low_level)
    assert len(torch.cuda.get_rng_state_all()) == torch.cuda.device_count()

    model, optimizer, cursor = _make_cuda_model(3, target_device)
    checkpoint = tmp_path / "cuda-rng.pt"
    save_checkpoint_atomic(checkpoint, model, optimizer, cursor, run_seed=3)
    expected_checkpoint_rng = torch.rand(8, device=target_device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["rng_state"]["torch_cuda"]
    assert payload["environment"]["device_type"] == "cuda"
    assert payload["environment"]["device_index"] == 0
    assert payload["environment"]["gpu_name"] == TARGET_GPU_NAME

    torch.manual_seed(999)
    resumed_model = wrn16_8().to(target_device)
    resumed_optimizer = build_formal_sgd(resumed_model)
    restored_cursor, restored_seed = load_checkpoint(
        checkpoint, resumed_model, resumed_optimizer
    )
    actual_checkpoint_rng = torch.rand(8, device=target_device)
    assert restored_seed == 3
    assert restored_cursor == cursor
    assert torch.equal(expected_checkpoint_rng, actual_checkpoint_rng)


def test_cuda_uninterrupted_resume_is_exact_including_bn_and_rng(
    target_device: torch.device, tmp_path: Path
) -> None:
    baseline_model, baseline_optimizer, baseline_cursor = _make_cuda_model(
        4, target_device
    )
    for _ in range(5):
        _random_cuda_step(
            baseline_model, baseline_optimizer, baseline_cursor, target_device
        )

    interrupted_model, interrupted_optimizer, interrupted_cursor = (
        _make_cuda_model(4, target_device)
    )
    for _ in range(2):
        _random_cuda_step(
            interrupted_model,
            interrupted_optimizer,
            interrupted_cursor,
            target_device,
        )
    checkpoint = tmp_path / "cuda-resume.pt"
    save_checkpoint_atomic(
        checkpoint,
        interrupted_model,
        interrupted_optimizer,
        interrupted_cursor,
        run_seed=4,
    )
    del interrupted_model, interrupted_optimizer
    torch.cuda.empty_cache()

    torch.manual_seed(999)
    resumed_model = wrn16_8().to(target_device)
    resumed_optimizer = build_formal_sgd(resumed_model)
    resumed_cursor, run_seed = load_checkpoint(
        checkpoint, resumed_model, resumed_optimizer
    )
    assert run_seed == 4
    assert audit_model_cuda_device(resumed_model, target_device) == (
        EXPECTED_TRAINABLE_PARAMETERS
    )
    assert_optimizer_state_cuda(resumed_optimizer, target_device)
    for _ in range(3):
        _random_cuda_step(
            resumed_model, resumed_optimizer, resumed_cursor, target_device
        )

    _assert_nested_exact(
        baseline_model.state_dict(), resumed_model.state_dict()
    )
    _assert_nested_exact(
        baseline_optimizer.state_dict(), resumed_optimizer.state_dict()
    )
    assert baseline_cursor == resumed_cursor == TrainingCursor(1, 5, 5)
    bn_buffer_names = [
        name
        for name in baseline_model.state_dict()
        if name.endswith(
            ("running_mean", "running_var", "num_batches_tracked")
        )
    ]
    assert bn_buffer_names
    assert all(
        torch.equal(
            baseline_model.state_dict()[name], resumed_model.state_dict()[name]
        )
        for name in bn_buffer_names
    )
    baseline_fingerprint = canonical_training_fingerprint(
        baseline_model, baseline_optimizer, baseline_cursor.to_dict()
    )
    resumed_fingerprint = canonical_training_fingerprint(
        resumed_model, resumed_optimizer, resumed_cursor.to_dict()
    )
    assert baseline_fingerprint == resumed_fingerprint


def test_fresh_process_cuda_replay_is_exact(target_device: torch.device) -> None:
    del target_device
    command = [
        sys.executable,
        "-m",
        "src.wrn.phase4_cuda_probe",
        "--seed",
        "5",
        "--steps",
        "5",
        "--device-index",
        "0",
    ]
    outputs: list[dict[str, Any]] = []
    for _ in range(2):
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=240,
        )
        outputs.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert outputs[0]["fingerprint"] == outputs[1]["fingerprint"]
    assert outputs[0]["cursor"] == outputs[1]["cursor"]
    assert outputs[0]["environment"]["gpu_name"] == TARGET_GPU_NAME
    assert outputs[0]["cifar_optimizer_steps"] == 0
    assert outputs[1]["cifar_optimizer_steps"] == 0
    print("PHASE4_FRESH_PROCESS=" + json.dumps(outputs[0], sort_keys=True))


def test_cuda_batch128_synthetic_capacity(target_device: torch.device) -> None:
    model, _optimizer, _cursor = _make_cuda_model(1, target_device)
    criterion = build_classification_criterion()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(target_device)
    inputs = torch.randn(
        128, 3, 32, 32, device=target_device, dtype=torch.float32
    )
    labels = torch.randint(0, 10, (128,), device=target_device)
    logits = model(inputs)
    assert logits.shape == (128, 10)
    assert torch.isfinite(logits).all()
    loss = criterion(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert_full_gradient_coverage(model)
    memory = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(target_device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(target_device),
    }
    assert memory["peak_allocated_bytes"] > 0
    assert memory["peak_reserved_bytes"] >= memory["peak_allocated_bytes"]
    print("PHASE4_MEMORY=" + json.dumps(memory, sort_keys=True))


def test_cifar_gpu_forward_only_fidelity_and_worker_replay(
    target_device: torch.device,
) -> None:
    single = build_cifar10_loaders(DATA_ROOT, num_workers=0, download=False)
    multi = build_cifar10_loaders(DATA_ROOT, num_workers=2, download=False)
    single.set_epoch(119)
    multi.set_epoch(119)
    cpu_inputs, cpu_labels = next(iter(single.train))
    multi_inputs, multi_labels = next(iter(multi.train))
    assert torch.equal(cpu_labels, multi_labels)
    assert torch.equal(cpu_inputs, multi_inputs)
    assert cpu_inputs.dtype == torch.float32
    assert cpu_labels.dtype == torch.int64
    assert torch.isfinite(cpu_inputs).all()
    assert -3.0 <= float(cpu_inputs.min()) <= float(cpu_inputs.max()) <= 3.0

    cuda_inputs = cpu_inputs.to(target_device)
    cuda_labels = cpu_labels.to(target_device)
    assert torch.equal(cuda_inputs.cpu(), cpu_inputs)
    assert torch.equal(cuda_labels.cpu(), cpu_labels)
    configure_target_runtime(1)
    model = wrn16_8().to(target_device)
    logits, loss = cifar_forward_only(model, cuda_inputs, cuda_labels)
    assert logits.shape == (128, 10)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    print(
        "PHASE4_CIFAR_FORWARD="
        + json.dumps(
            {
                "batch_shape": list(cuda_inputs.shape),
                "input_dtype": str(cuda_inputs.dtype),
                "label_dtype": str(cuda_labels.dtype),
                "logits_shape": list(logits.shape),
                "backward_count": 0,
                "optimizer_step_count": 0,
            },
            sort_keys=True,
        )
    )
