from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_import_has_no_dataset_download_side_effect(tmp_path: Path) -> None:
    script = """
from torchvision.datasets import CIFAR10
def forbidden(self):
    raise AssertionError('download called during import')
CIFAR10.download = forbidden
import src.wrn.data
print('IMPORT_PASS')
"""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert completed.stdout.strip() == "IMPORT_PASS"
    assert list(tmp_path.iterdir()) == []


def test_phase2_source_has_no_optimizer_scheduler_training_or_checkpoint_calls() -> None:
    forbidden_attributes = {"backward", "step", "save"}
    forbidden_import_roots = {"torch.optim"}
    phase2_files = [
        ROOT / "src" / "wrn" / "data.py",
        ROOT / "src" / "wrn" / "rng.py",
        ROOT / "src" / "wrn" / "transforms.py",
    ]
    for path in phase2_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name not in forbidden_import_roots for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert node.module not in forbidden_import_roots
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_attributes
