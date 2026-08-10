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
    assert not any(path.name in prohibited_names for path in ROOT.rglob("*.py"))
    assert not (ROOT / "runs").exists()
    assert not (ROOT / "checkpoints").exists()
    artifact_suffixes = {".pt", ".pth", ".ckpt"}
    assert not any(
        path.suffix.lower() in artifact_suffixes
        for path in ROOT.rglob("*")
        if "data" not in path.parts
    )
