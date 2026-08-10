from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE3_MODULES = (
    ROOT / "src" / "wrn" / "optimization.py",
    ROOT / "src" / "wrn" / "schedule.py",
    ROOT / "src" / "wrn" / "state.py",
    ROOT / "src" / "wrn" / "training.py",
    ROOT / "src" / "wrn" / "checkpoint.py",
)
ARTIFACT_SUFFIXES = frozenset({".pt", ".pth", ".ckpt"})
EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "data",
        "node_modules",
        "references",
    }
)


def _is_under_excluded_directory(root: Path, path: Path) -> bool:
    relative_parts = path.relative_to(root).parts
    return any(
        part.casefold() in EXCLUDED_DIRS for part in relative_parts[:-1]
    )


def find_phase3_training_artifacts(root: Path) -> tuple[Path, ...]:
    """Find checkpoint-like files in repository-managed runtime locations."""

    return tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.casefold() in ARTIFACT_SUFFIXES
            and not _is_under_excluded_directory(root, path)
        )
    )


def test_phase3_has_no_cifar_training_launcher_or_hidden_modern_features() -> None:
    banned_text = (
        "CIFAR10(",
        "from torchvision",
        "DataLoader",
        "autocast",
        "GradScaler",
        "clip_grad",
        "torch.compile",
        "ExponentialMovingAverage",
        "StochasticWeightAveraging",
        "ReduceLROnPlateau",
        "label_smoothing=0.1",
        "mixup",
        "cutmix",
        "best_checkpoint",
        "early_stopping",
    )
    for path in PHASE3_MODULES:
        source = path.read_text(encoding="utf-8")
        for banned in banned_text:
            assert banned not in source, f"{banned!r} found in {path.name}"


def test_optimizer_steps_and_backward_are_confined_to_training_engine() -> None:
    calls: list[tuple[str, str]] = []
    for path in PHASE3_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in {"step", "backward"}:
                calls.append((path.name, node.func.attr))
    assert calls == [("training.py", "backward"), ("training.py", "step")]


def test_repository_has_no_phase3_training_artifacts_or_formal_launcher() -> None:
    prohibited_names = {"formal_train.py", "run_5seeds.py", "launch_training.py"}
    assert not any(
        path.name in prohibited_names
        for path in ROOT.rglob("*.py")
        if not _is_under_excluded_directory(ROOT, path)
    )
    assert not (ROOT / "runs").exists()
    assert not (ROOT / "checkpoints").exists()
    assert find_phase3_training_artifacts(ROOT) == ()


def test_artifact_scan_ignores_virtualenv_but_detects_real_artifacts(
    tmp_path: Path,
) -> None:
    virtualenv_file = (
        tmp_path
        / ".venv"
        / "Lib"
        / "site-packages"
        / "distutils-precedence.pth"
    )
    real_artifacts = {
        tmp_path / "checkpoints" / "model.pt",
        tmp_path / "model.pth",
        tmp_path / "run.ckpt",
    }
    for path in {virtualenv_file, *real_artifacts}:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"boundary-test-placeholder")

    detected = set(find_phase3_training_artifacts(tmp_path))

    assert virtualenv_file not in detected
    assert detected == real_artifacts
