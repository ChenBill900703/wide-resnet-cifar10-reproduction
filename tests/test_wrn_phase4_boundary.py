from __future__ import annotations

import ast
from pathlib import Path

from src.wrn.cuda_validation import has_nominal_eight_gb_vram


ROOT = Path(__file__).resolve().parents[1]
CUDA_VALIDATION = ROOT / "src" / "wrn" / "cuda_validation.py"
CUDA_PROBE = ROOT / "src" / "wrn" / "phase4_cuda_probe.py"
CUDA_TEST = ROOT / "tests" / "test_wrn_phase4_cuda.py"


def _function_calls(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    calls: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.add(node.func.id)
    return calls


def test_cifar_phase4_paths_are_forward_only() -> None:
    production_calls = _function_calls(CUDA_VALIDATION, "cifar_forward_only")
    test_calls = _function_calls(
        CUDA_TEST, "test_cifar_gpu_forward_only_fidelity_and_worker_replay"
    )
    assert "backward" not in production_calls | test_calls
    assert "step" not in production_calls | test_calls
    assert "train_step" not in production_calls | test_calls


def test_phase4_has_no_amp_compile_or_formal_launcher() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (CUDA_VALIDATION, CUDA_PROBE)
    )
    prohibited = (
        "GradScaler",
        "torch.amp",
        "torch.cuda.amp",
        "torch.compile(",
        "formal_train",
        "run_5seeds",
        "launch_training",
    )
    for text in prohibited:
        assert text not in combined
    assert "CIFAR10" not in CUDA_PROBE.read_text(encoding="utf-8")


def test_phase4_probe_is_synthetic_and_has_no_repository_artifact_path() -> None:
    source = CUDA_PROBE.read_text(encoding="utf-8")
    assert "torch.randn" in source
    assert "torch.randint" in source
    assert "save_checkpoint" not in source
    assert "runs" not in source
    assert "checkpoints" not in source


def test_nominal_vram_gate_accepts_observed_3070_ti_capacity() -> None:
    assert has_nominal_eight_gb_vram(8_589_410_304)
    assert has_nominal_eight_gb_vram(8_000_000_000)
    assert not has_nominal_eight_gb_vram(7_999_999_999)
