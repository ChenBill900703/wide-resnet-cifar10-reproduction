from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "src" / "wrn" / "formal_launcher.py"
PROTOCOL = ROOT / "src" / "wrn" / "formal_protocol.py"
PHASE5_TESTS = (
    ROOT / "tests" / "test_wrn_phase5_formal.py",
    ROOT / "tests" / "test_wrn_phase5_boundary.py",
)


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _calls(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Attribute):
            names.add(child.func.attr)
        elif isinstance(child.func, ast.Name):
            names.add(child.func.id)
    return names


def test_dry_run_summary_and_preflight_have_no_training_calls() -> None:
    launcher_dry_run = _function_node(LAUNCHER, "dry_run_summary")
    protocol_preflight = _function_node(PROTOCOL, "perform_formal_preflight")
    prohibited = {"train_step", "backward", "step", "execute_formal_training"}
    assert not (_calls(launcher_dry_run) & prohibited)
    assert not (_calls(protocol_preflight) & prohibited)


def test_training_call_is_confined_to_explicit_execute_function() -> None:
    tree = ast.parse(LAUNCHER.read_text(encoding="utf-8"))
    locations: list[str] = []
    for function in (
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    ):
        if "train_step" in _calls(function):
            locations.append(function.name)
    assert locations == ["execute_formal_training"]


def test_phase5_tests_do_not_call_real_training_or_cuda_backward() -> None:
    for path in PHASE5_TESTS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = _calls(tree)
        assert "execute_formal_training" not in calls
        assert "train_step" not in calls
        assert "backward" not in calls
        assert "step" not in calls


def test_formal_launcher_has_no_hidden_modern_training_features() -> None:
    combined = LAUNCHER.read_text(encoding="utf-8") + PROTOCOL.read_text(
        encoding="utf-8"
    )
    prohibited = (
        "GradScaler",
        "torch.amp",
        "torch.cuda.amp",
        "torch.compile(",
        "clip_grad",
        "mixup",
        "cutmix",
        "early_stopping",
        "save_best",
        "select_best_checkpoint",
        "ReduceLROnPlateau",
        "CosineAnnealing",
        "ExponentialMovingAverage",
        "StochasticWeightAveraging",
    )
    for text in prohibited:
        assert text not in combined


def test_epoch200_only_result_has_no_best_metric_selector() -> None:
    evaluation = _function_node(LAUNCHER, "_evaluate_epoch_200")
    assert "min" not in _calls(evaluation)
    assert "max" not in _calls(evaluation)
    source = LAUNCHER.read_text(encoding="utf-8")
    assert '"result_selection": "epoch_200_final_checkpoint_only"' in source
    assert '"best_test_metric": None' in source


def test_candidate_validation_can_never_execute() -> None:
    source = PROTOCOL.read_text(encoding="utf-8")
    assert "freeze-candidate mode can never execute training" in source
    assert "WRN_PHASE5_CANDIDATE_VALIDATION" in source


def test_formal_outputs_are_derived_from_explicit_external_run_directory() -> None:
    launcher_source = LAUNCHER.read_text(encoding="utf-8")
    protocol_source = PROTOCOL.read_text(encoding="utf-8")
    assert "REPOSITORY_ROOT / \"runs\"" not in launcher_source + protocol_source
    assert "REPOSITORY_ROOT / \"checkpoints\"" not in launcher_source + protocol_source
    assert "formal output root must be outside the Git repository" in protocol_source
