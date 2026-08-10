"""Target-gated Phase 4 CUDA validation helpers; never launches training."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .training import build_classification_criterion, configure_reproducibility


TARGET_GPU_NAME = "NVIDIA GeForce RTX 3070 Ti"
EXPECTED_TRAINABLE_PARAMETERS = 10_961_370
NOMINAL_EIGHT_GB_BYTES = 8_000_000_000


def require_target_cuda_device(index: int = 0) -> torch.device:
    """Fail unless the explicit device is the approved RTX 3070 Ti target."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python runtime")
    if not 0 <= index < torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {index} is unavailable")
    device = torch.device("cuda", index)
    actual_name = torch.cuda.get_device_name(device)
    if actual_name.casefold() != TARGET_GPU_NAME.casefold():
        raise RuntimeError(
            f"target GPU mismatch: {actual_name!r} != {TARGET_GPU_NAME!r}"
        )
    torch.cuda.set_device(device)
    return device


def configure_target_runtime(run_seed: int) -> None:
    """Apply the frozen deterministic settings without AMP or compilation."""

    configure_reproducibility(run_seed)


def has_nominal_eight_gb_vram(total_bytes: int) -> bool:
    """Use the vendor's decimal GB convention, not an exact 8 GiB boundary."""

    return total_bytes >= NOMINAL_EIGHT_GB_BYTES


def _distribution_version(name: str) -> str:
    return importlib.metadata.version(name)


def _nvidia_driver(index: int) -> str:
    command = [
        "nvidia-smi",
        f"--id={index}",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip().splitlines()[0].strip()


def target_environment_snapshot(device: torch.device) -> dict[str, Any]:
    """Return concise, JSON-safe target environment evidence."""

    if device.type != "cuda" or device.index is None:
        raise ValueError("target environment snapshot requires explicit CUDA device")
    properties = torch.cuda.get_device_properties(device)
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": _distribution_version("torchvision"),
        "numpy": _distribution_version("numpy"),
        "pyyaml": _distribution_version("PyYAML"),
        "pytest": _distribution_version("pytest"),
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "selected_device_index": device.index,
        "current_device_index": torch.cuda.current_device(),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_vram_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
        "nvidia_driver": _nvidia_driver(device.index),
        "default_dtype": str(torch.get_default_dtype()),
    }


def numerical_policy_snapshot() -> dict[str, Any]:
    """Read effective policy through current API, with a legacy fallback."""

    has_precision_api = hasattr(
        torch.backends.cuda.matmul, "fp32_precision"
    ) and hasattr(torch.backends.cudnn, "fp32_precision")
    if has_precision_api:
        matmul_policy = torch.backends.cuda.matmul.fp32_precision
        cudnn_policy = torch.backends.cudnn.fp32_precision
        tf32_enabled = matmul_policy != "ieee" or cudnn_policy != "ieee"
        tf32_evidence = {
            "api": "fp32_precision",
            "matmul": matmul_policy,
            "cudnn": cudnn_policy,
        }
    else:
        matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
        tf32_enabled = matmul_tf32 or cudnn_tf32
        tf32_evidence = {
            "api": "allow_tf32",
            "matmul": matmul_tf32,
            "cudnn": cudnn_tf32,
        }
    try:
        autocast_enabled = bool(torch.is_autocast_enabled("cuda"))
    except TypeError:
        autocast_enabled = bool(torch.is_autocast_enabled())
    return {
        "default_dtype": str(torch.get_default_dtype()),
        "autocast_cuda_enabled": autocast_enabled,
        "tf32_enabled": tf32_enabled,
        "tf32_evidence": tf32_evidence,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
    }


def audit_model_cuda_device(model: nn.Module, device: torch.device) -> int:
    """Require the exact parameter count and no CPU/mixed model state."""

    if device.type != "cuda":
        raise ValueError("model CUDA audit requires a CUDA device")
    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable_count != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            f"trainable parameter count {trainable_count} != "
            f"{EXPECTED_TRAINABLE_PARAMETERS}"
        )
    wrong_parameters = [
        name
        for name, parameter in model.named_parameters()
        if parameter.device != device
    ]
    wrong_buffers = [
        name for name, buffer in model.named_buffers() if buffer.device != device
    ]
    if wrong_parameters or wrong_buffers:
        raise RuntimeError(
            f"mixed-device model: parameters={wrong_parameters}, "
            f"buffers={wrong_buffers}"
        )
    return trainable_count


def assert_optimizer_state_cuda(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    """Require every tensor optimizer state, including momentum, on target."""

    wrong: list[str] = []
    for parameter_index, state in enumerate(optimizer.state.values()):
        for name, value in state.items():
            if isinstance(value, Tensor) and value.device != device:
                wrong.append(f"{parameter_index}:{name}:{value.device}")
    if wrong:
        raise RuntimeError(f"optimizer state not on {device}: {wrong}")


def _update_tensor_hash(digest: Any, name: str, tensor: Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())


def canonical_training_fingerprint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cursor: Mapping[str, int],
) -> str:
    """Hash model buffers/parameters, momentum, metadata and cursor canonically."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        _update_tensor_hash(digest, f"model:{name}", tensor)
    named_parameters = dict(model.named_parameters())
    for name in sorted(named_parameters):
        parameter = named_parameters[name]
        state = optimizer.state.get(parameter, {})
        for state_name, value in sorted(state.items()):
            if isinstance(value, Tensor):
                _update_tensor_hash(
                    digest, f"optimizer:{name}:{state_name}", value
                )
    group_metadata = [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]
    digest.update(
        json.dumps(group_metadata, sort_keys=True, default=str).encode("utf-8")
    )
    digest.update(json.dumps(dict(cursor), sort_keys=True).encode("ascii"))
    return digest.hexdigest()


def cifar_forward_only(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run one CIFAR batch without backward, optimizer or result calculation."""

    if not inputs.is_cuda or inputs.dtype != torch.float32:
        raise ValueError("CIFAR inputs must be CUDA float32")
    if not labels.is_cuda or labels.dtype != torch.int64:
        raise ValueError("CIFAR labels must be CUDA int64")
    if labels.numel() and (labels.min() < 0 or labels.max() > 9):
        raise ValueError("CIFAR labels must be in [0, 9]")
    model.eval()
    criterion = build_classification_criterion()
    with torch.no_grad():
        logits = model(inputs)
        loss = criterion(logits, labels)
    return logits, loss
