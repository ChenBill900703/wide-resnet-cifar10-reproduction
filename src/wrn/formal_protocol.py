"""Fail-closed Phase 5 formal-run identity, preflight, and artifact helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    EPOCH_DATA_POLICY,
    FORMAL_TARGET_NAME,
    FROZEN_CONFIG_SHA256,
)
from .cuda_validation import (
    EXPECTED_TRAINABLE_PARAMETERS,
    TARGET_GPU_NAME,
    audit_model_cuda_device,
    canonical_training_fingerprint,
    configure_target_runtime,
    numerical_policy_snapshot,
    require_target_cuda_device,
    target_environment_snapshot,
)
from .data import CIFAR10DataLoaders, build_cifar10_loaders
from .model import wrn16_8
from .optimization import (
    DAMPENING,
    INITIAL_LR,
    MOMENTUM,
    NESTEROV,
    WEIGHT_DECAY,
    assert_formal_parameter_group,
    build_formal_sgd,
)
from .schedule import EPOCHS, TOTAL_UPDATES, UPDATES_PER_EPOCH, end_update
from .state import TrainingCursor
from .training import build_classification_criterion


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FORMAL_SEEDS = (1, 2, 3, 4, 5)
CURRENT_HUMAN_AUTHORIZED_SEED = 1
DATASET_ARCHIVE_SHA256 = (
    "6D958BE074577803D12ECDEFD02955F39262C83C16FE9348329D7FE0B5C001CE"
)
DATASET_ARCHIVE_NAME = "cifar-10-python.tar.gz"
FORMAL_RUN_SCHEMA_VERSION = 1
FORMAL_RESULT_SCHEMA_VERSION = 1
ARTIFACT_HASH_SCHEMA_VERSION = 1
CHECKPOINT_EPOCHS = (60, 120, 160, 200)
EXPECTED_CHECKPOINT_UPDATES = {epoch: end_update(epoch) for epoch in CHECKPOINT_EPOCHS}
RUN_DIRECTORY_NAME = {
    seed: f"{FORMAL_TARGET_NAME}/seed_{seed:02d}" for seed in FORMAL_SEEDS
}

FROZEN_ARTIFACT_SHA256 = {
    "references/wide_residual_networks.pdf": (
        "AF606AEB02AE3D713A3BA7AD34A5A583D911A5FD33E52B76507DC37A8ED021E9"
    ),
    "AGENTS.md": "2E169C8EA5B65685B5A8254AF6F9DF0C4F407062ECE47893C0BA240BF7D7471F",
    "docs/reproduction_spec.md": (
        "7D471C9F90B6A172D25ABF83E8FBADD7CEC83656EA88EA4058DC3C0A22DC965F"
    ),
    "docs/assumptions.md": (
        "A09A2D497FD0D72A1B14751124D2E9BD00C1F84A617E0CF38F0B56D7213F9794"
    ),
    "docs/evidence_table.md": (
        "99E2B7D82BCA17CCDACC8E1E0D03FE90AB5EB160B80CFCA4A4F4E51C05780FA2"
    ),
    "docs/open_questions.md": (
        "541C07906823843868E76E9458E53DAA89EC5F1AF048BE04D4D31B05E74CD987"
    ),
    "configs/wrn16_8_arxiv_v4_frozen.yaml": FROZEN_CONFIG_SHA256,
}

EXPECTED_TARGET_ENVIRONMENT = {
    "os": "Windows-10-10.0.26100-SP0",
    "python": "3.11.9",
    "torch": "2.13.0+cu126",
    "torchvision": "0.28.0+cu126",
    "numpy": "2.4.4",
    "pyyaml": "6.0.3",
    "pytest": "9.1.1",
    "torch_cuda_build": "12.6",
    "cudnn": 91002,
    "cuda_available": True,
    "cuda_device_count": 1,
    "selected_device_index": 0,
    "current_device_index": 0,
    "gpu_name": TARGET_GPU_NAME,
    "gpu_total_vram_bytes": 8_589_410_304,
    "compute_capability": [8, 6],
    "nvidia_driver": "591.86",
    "default_dtype": "torch.float32",
}

PHASE5_CANDIDATE_PATHS = frozenset(
    {
        "docs/phase5_environment_lock.txt",
        "docs/phase5_formal_run_freeze.md",
        "src/wrn/checkpoint.py",
        "src/wrn/formal_launcher.py",
        "src/wrn/formal_protocol.py",
        "tests/test_wrn_checkpoint.py",
        "tests/test_wrn_phase3_boundary.py",
        "tests/test_wrn_phase5_boundary.py",
        "tests/test_wrn_phase5_formal.py",
    }
)


class FormalPreflightError(RuntimeError):
    """A formal launch gate failed before any CIFAR training update."""


@dataclass(frozen=True)
class FormalLaunchRequest:
    seed: int
    output_root: Path
    data_root: Path
    num_workers: int = 2
    execute_formal: bool = False
    resume_checkpoint: Path | None = None
    expected_code_sha: str | None = None
    validate_freeze_candidate: bool = False


@dataclass
class FormalPreflight:
    request: FormalLaunchRequest
    code_sha: str
    run_id: str
    run_directory: Path
    device: torch.device
    environment: dict[str, Any]
    numerical_policy: dict[str, Any]
    dataset: dict[str, Any]
    loaders: CIFAR10DataLoaders
    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    cursor: TrainingCursor
    initial_training_fingerprint: str
    manifest: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_formal_seed(seed: int) -> int:
    if seed not in FORMAL_SEEDS:
        raise FormalPreflightError(f"formal seed must be one of {FORMAL_SEEDS}")
    return seed


def require_execute_authorization(seed: int, environment: dict[str, str]) -> int:
    validate_formal_seed(seed)
    if environment.get("WRN_FORMAL_TRAINING_AUTHORIZED") != "1":
        raise FormalPreflightError(
            "WRN_FORMAL_TRAINING_AUTHORIZED=1 is required for formal execution"
        )
    authorized_text = environment.get("WRN_FORMAL_AUTHORIZED_SEED")
    try:
        authorized_seed = int(authorized_text) if authorized_text is not None else None
    except ValueError as error:
        raise FormalPreflightError(
            "WRN_FORMAL_AUTHORIZED_SEED must be an integer"
        ) from error
    if authorized_seed not in FORMAL_SEEDS:
        raise FormalPreflightError(
            f"authorized seed must be one of the frozen seeds {FORMAL_SEEDS}"
        )
    if seed != authorized_seed:
        raise FormalPreflightError(
            f"requested seed {seed} does not match authorized seed {authorized_seed}"
        )
    return authorized_seed


def dry_run_authorized_seed(environment: dict[str, str]) -> int:
    """Report the current authorization without using it to start training."""

    text = environment.get("WRN_FORMAL_AUTHORIZED_SEED")
    if text is None:
        return CURRENT_HUMAN_AUTHORIZED_SEED
    try:
        seed = int(text)
    except ValueError as error:
        raise FormalPreflightError(
            "WRN_FORMAL_AUTHORIZED_SEED must be an integer"
        ) from error
    return validate_formal_seed(seed)


def _git(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    return result.stdout.rstrip("\r\n")


def _status_path(status_line: str) -> str:
    path = status_line[3:].strip().replace("\\", "/")
    if " -> " in path:
        path = path.split(" -> ", maxsplit=1)[1]
    return path


def validate_git_status_lines(
    status_lines: list[str], *, allow_freeze_candidate: bool
) -> None:
    tracked_dirty = [line for line in status_lines if not line.startswith("?? ")]
    untracked = [line for line in status_lines if line.startswith("?? ")]

    if allow_freeze_candidate:
        unexpected_tracked = [
            line
            for line in tracked_dirty
            if _status_path(line) not in PHASE5_CANDIDATE_PATHS
        ]
        unexpected_untracked = [
            line
            for line in untracked
            if not _status_path(line).startswith("references/")
            and _status_path(line) not in PHASE5_CANDIDATE_PATHS
        ]
        if unexpected_tracked or unexpected_untracked:
            raise FormalPreflightError(
                "freeze-candidate tree contains unexpected changes: "
                f"{unexpected_tracked + unexpected_untracked}"
            )
        return

    if tracked_dirty:
        raise FormalPreflightError(
            f"tracked Git working tree is not clean: {tracked_dirty}"
        )
    unexpected_untracked = [
        line
        for line in untracked
        if not _status_path(line).startswith("references/")
    ]
    if unexpected_untracked:
        raise FormalPreflightError(
            f"unexpected untracked repository content: {unexpected_untracked}"
        )


def git_preflight(
    repository_root: Path, *, allow_freeze_candidate: bool
) -> dict[str, Any]:
    try:
        code_sha = _git(repository_root, "rev-parse", "HEAD")
        status_text = _git(
            repository_root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FormalPreflightError("Git preflight command failed") from error
    if len(code_sha) != 40 or any(
        character not in "0123456789abcdef" for character in code_sha
    ):
        raise FormalPreflightError("Git HEAD is not a lowercase full commit SHA")
    status_lines = [entry for entry in status_text.split("\0") if entry]
    validate_git_status_lines(
        status_lines, allow_freeze_candidate=allow_freeze_candidate
    )
    return {
        "code_sha": code_sha,
        "working_tree_mode": (
            "PHASE5_FREEZE_CANDIDATE" if allow_freeze_candidate else "CLEAN"
        ),
        "status_lines": status_lines,
    }


def audit_frozen_artifacts(repository_root: Path) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative_path, expected_hash in FROZEN_ARTIFACT_SHA256.items():
        path = repository_root / relative_path
        if not path.is_file():
            raise FormalPreflightError(f"frozen artifact is missing: {relative_path}")
        file_hash = sha256_file(path)
        if file_hash != expected_hash:
            raise FormalPreflightError(
                f"frozen artifact hash mismatch for {relative_path}: "
                f"{file_hash} != {expected_hash}"
            )
        actual[relative_path] = file_hash
    return actual


def resolve_formal_run_directory(
    repository_root: Path, output_root: Path, seed: int
) -> tuple[Path, str]:
    validate_formal_seed(seed)
    if not output_root.is_absolute():
        raise FormalPreflightError("formal output root must be an absolute path")
    repository = repository_root.resolve()
    output = output_root.resolve()
    if output == repository or output.is_relative_to(repository):
        raise FormalPreflightError(
            "formal output root must be outside the Git repository"
        )
    relative_run_id = RUN_DIRECTORY_NAME[seed]
    run_directory = output.joinpath(*relative_run_id.split("/")).resolve()
    if run_directory == repository or run_directory.is_relative_to(repository):
        raise FormalPreflightError("formal run directory resolves inside repository")
    return run_directory, relative_run_id


def validate_run_directory_state(
    run_directory: Path, resume_checkpoint: Path | None
) -> Path | None:
    if resume_checkpoint is None:
        if run_directory.exists():
            raise FormalPreflightError(
                f"formal run directory already exists: {run_directory}"
            )
        return None

    checkpoint = resume_checkpoint.resolve()
    if not run_directory.is_dir():
        raise FormalPreflightError("resume requires the existing formal run directory")
    checkpoint_root = (run_directory / "checkpoints").resolve()
    if not checkpoint.is_file() or not checkpoint.is_relative_to(checkpoint_root):
        raise FormalPreflightError(
            "resume checkpoint must be an existing file under this run's checkpoints"
        )
    completed_result = run_directory / "final_result.json"
    if completed_result.exists():
        try:
            result_status = json.loads(
                completed_result.read_text(encoding="utf-8")
            ).get("status")
        except (AttributeError, json.JSONDecodeError, OSError) as error:
            raise FormalPreflightError("formal result status is unreadable") from error
        if result_status == "COMPLETED":
            raise FormalPreflightError("a completed formal run cannot be resumed")
        if result_status not in {"FAILED", "INTERRUPTED"}:
            raise FormalPreflightError(
                f"formal result has unsupported status: {result_status!r}"
            )
    checkpoint_files = sorted(checkpoint_root.glob("epoch_*.pt"))
    if checkpoint_files and checkpoint != checkpoint_files[-1].resolve():
        raise FormalPreflightError("resume must use the latest formal checkpoint")
    return checkpoint


def dataset_preflight(
    data_root: Path, *, num_workers: int = 0
) -> tuple[dict[str, Any], CIFAR10DataLoaders]:
    archive = data_root / DATASET_ARCHIVE_NAME
    archive_present = archive.is_file()
    archive_hash = sha256_file(archive) if archive_present else None
    if archive_hash is not None and archive_hash != DATASET_ARCHIVE_SHA256:
        raise FormalPreflightError(
            f"CIFAR-10 archive hash mismatch: {archive_hash} != "
            f"{DATASET_ARCHIVE_SHA256}"
        )
    try:
        loaders = build_cifar10_loaders(
            data_root, num_workers=num_workers, download=False
        )
    except Exception as error:
        raise FormalPreflightError(
            "CIFAR-10 extracted data is missing or corrupt; formal mode never downloads"
        ) from error
    if not loaders.train_dataset.integrity_ok() or not loaders.test_dataset.integrity_ok():
        raise FormalPreflightError("CIFAR-10 official split integrity check failed")
    if len(loaders.train_dataset) != 50_000 or len(loaders.test_dataset) != 10_000:
        raise FormalPreflightError("CIFAR-10 split lengths are not 50,000/10,000")
    return (
        {
            "name": "CIFAR-10",
            "archive_name": DATASET_ARCHIVE_NAME,
            "archive_present": archive_present,
            "archive_sha256": DATASET_ARCHIVE_SHA256,
            "archive_actual_sha256": archive_hash,
            "official_split_integrity": True,
            "train_samples": 50_000,
            "test_samples": 10_000,
            "download_performed": False,
        },
        loaders,
    )


def validate_environment_snapshot(environment: dict[str, Any]) -> None:
    mismatches = {
        name: {"actual": environment.get(name), "expected": expected}
        for name, expected in EXPECTED_TARGET_ENVIRONMENT.items()
        if environment.get(name) != expected
    }
    if mismatches:
        raise FormalPreflightError(f"formal environment mismatch: {mismatches}")


def validate_project_interpreter(repository_root: Path, executable: str) -> str:
    expected = (repository_root / ".venv" / "Scripts" / "python.exe").resolve()
    actual = Path(executable).resolve()
    if str(actual).casefold() != str(expected).casefold():
        raise FormalPreflightError(
            "formal interpreter must be the project-local .venv/Scripts/python.exe"
        )
    return ".venv/Scripts/python.exe"


def package_inventory() -> list[str]:
    """Return a credential-free installed name/version inventory."""

    packages = {
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    return sorted(packages, key=str.casefold)


def target_environment_preflight(
    repository_root: Path,
) -> tuple[torch.device, dict[str, Any]]:
    if os.environ.get("WRN_REQUIRE_TARGET_CUDA") != "1":
        raise FormalPreflightError("WRN_REQUIRE_TARGET_CUDA=1 is required")
    interpreter_identity = validate_project_interpreter(repository_root, sys.executable)
    try:
        device = require_target_cuda_device(0)
        environment = target_environment_snapshot(device)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        raise FormalPreflightError("target CUDA environment gate failed") from error
    environment["pip"] = importlib.metadata.version("pip")
    environment["python_executable_identity"] = interpreter_identity
    validate_environment_snapshot(environment)
    return device, environment


def validate_numerical_policy(policy: dict[str, Any]) -> None:
    expected = {
        "default_dtype": "torch.float32",
        "autocast_cuda_enabled": False,
        "tf32_enabled": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
    }
    mismatches = {
        name: {"actual": policy.get(name), "expected": value}
        for name, value in expected.items()
        if policy.get(name) != value
    }
    if mismatches:
        raise FormalPreflightError(f"formal numerical policy mismatch: {mismatches}")


def validate_model_optimizer_pre_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cursor: TrainingCursor,
    *,
    require_initial_cursor: bool,
) -> int:
    count = audit_model_cuda_device(model, device)
    if count != EXPECTED_TRAINABLE_PARAMETERS:
        raise FormalPreflightError("formal trainable parameter count mismatch")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise FormalPreflightError("all formal model parameters must be float32")
    try:
        assert_formal_parameter_group(
            model,
            optimizer,
            expected_lr=INITIAL_LR if require_initial_cursor else None,
        )
    except (TypeError, ValueError) as error:
        raise FormalPreflightError("formal optimizer pre-step audit failed") from error
    cursor.validate()
    if require_initial_cursor and cursor != TrainingCursor(1, 0, 0):
        raise FormalPreflightError("new formal run must start at cursor (1, 0, 0)")
    return count


def _formal_recipe() -> dict[str, Any]:
    return {
        "model": {
            "identity": "WRN-16-8 B(3,3)",
            "depth": 16,
            "widen_factor": 8,
            "blocks_per_group": 2,
            "stage_channels": [16, 128, 256, 512],
            "dropout": 0.0,
            "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        },
        "optimizer": {
            "type": "torch.optim.SGD",
            "parameter_groups": 1,
            "scope": "all_trainable_parameters",
            "lr": INITIAL_LR,
            "momentum": MOMENTUM,
            "dampening": DAMPENING,
            "weight_decay": WEIGHT_DECAY,
            "weight_decay_semantics": "coupled",
            "nesterov": NESTEROV,
            "foreach": False,
            "fused": False,
        },
        "loss": {
            "type": "torch.nn.CrossEntropyLoss",
            "reduction": "mean",
            "label_smoothing": 0.0,
        },
        "lr_schedule": [
            {"start_epoch": 1, "end_epoch": 59, "lr": 0.1},
            {"start_epoch": 60, "end_epoch": 119, "lr": 0.02},
            {"start_epoch": 120, "end_epoch": 159, "lr": 0.004},
            {"start_epoch": 160, "end_epoch": 200, "lr": 0.0008},
        ],
        "batch_size": 128,
        "train_drop_last": True,
        "test_drop_last": False,
        "epochs": EPOCHS,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "total_planned_updates": TOTAL_UPDATES,
        "epoch_data_seed_policy": EPOCH_DATA_POLICY,
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "checkpoint_updates": {
            str(epoch): update
            for epoch, update in EXPECTED_CHECKPOINT_UPDATES.items()
        },
        "result_selection": "epoch_200_final_checkpoint_only",
        "diagnostic_evaluation_cadence": "none_before_epoch_200",
        "best_checkpoint_selection": False,
    }


def build_run_manifest(
    *,
    request: FormalLaunchRequest,
    human_authorized_seed: int,
    code_sha: str,
    code_baseline_status: str,
    run_id: str,
    run_directory: Path,
    frozen_hashes: dict[str, str],
    dataset: dict[str, Any],
    environment: dict[str, Any],
    numerical_policy: dict[str, Any],
    initial_training_fingerprint: str,
    start_timestamp: str,
) -> dict[str, Any]:
    return {
        "schema_version": FORMAL_RUN_SCHEMA_VERSION,
        "formal_target_name": FORMAL_TARGET_NAME,
        "formal_run_id": run_id,
        "seed": request.seed,
        "formal_seed_set": list(FORMAL_SEEDS),
        "human_authorized_seed": human_authorized_seed,
        "code_commit_sha": code_sha,
        "code_baseline_status": code_baseline_status,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "frozen_config_path": "configs/wrn16_8_arxiv_v4_frozen.yaml",
        "frozen_config_sha256": FROZEN_CONFIG_SHA256,
        "frozen_artifact_sha256": frozen_hashes,
        "dataset": dataset,
        "recipe": {**_formal_recipe(), "data_loader_workers": request.num_workers},
        "environment": environment,
        "installed_package_inventory": package_inventory(),
        "numerical_policy": {
            **numerical_policy,
            "fp32": True,
            "amp": False,
            "compile": False,
            "eager_execution": True,
        },
        "initial_training_fingerprint": initial_training_fingerprint,
        "output_directory": str(run_directory),
        "start_timestamp_utc": start_timestamp,
        "no_post_seed1_tuning_rule": (
            "After seed 1 first CIFAR optimizer step, any protocol/source change "
            "invalidates seed 1 and requires a new freeze and restart of all seeds."
        ),
    }


def perform_formal_preflight(request: FormalLaunchRequest) -> FormalPreflight:
    validate_formal_seed(request.seed)
    if request.num_workers < 0:
        raise FormalPreflightError("num_workers must be non-negative")
    if request.validate_freeze_candidate:
        if request.execute_formal:
            raise FormalPreflightError("freeze-candidate mode can never execute training")
        if os.environ.get("WRN_PHASE5_CANDIDATE_VALIDATION") != "1":
            raise FormalPreflightError(
                "WRN_PHASE5_CANDIDATE_VALIDATION=1 is required for candidate dry-run"
            )
    if request.execute_formal:
        human_authorized_seed = require_execute_authorization(
            request.seed, dict(os.environ)
        )
        if request.expected_code_sha is None:
            raise FormalPreflightError("formal execution requires --expected-code-sha")
    else:
        human_authorized_seed = dry_run_authorized_seed(dict(os.environ))

    git = git_preflight(
        REPOSITORY_ROOT,
        allow_freeze_candidate=request.validate_freeze_candidate,
    )
    code_sha = git["code_sha"]
    if request.expected_code_sha is not None and request.expected_code_sha != code_sha:
        raise FormalPreflightError(
            f"Git HEAD {code_sha} != expected code SHA {request.expected_code_sha}"
        )
    code_baseline_status = (
        "PHASE5_FREEZE_CANDIDATE"
        if request.validate_freeze_candidate
        else "FROZEN_COMMIT"
    )

    frozen_hashes = audit_frozen_artifacts(REPOSITORY_ROOT)
    run_directory, run_id = resolve_formal_run_directory(
        REPOSITORY_ROOT, request.output_root, request.seed
    )
    resume_checkpoint = validate_run_directory_state(
        run_directory, request.resume_checkpoint
    )
    if resume_checkpoint != request.resume_checkpoint:
        request = FormalLaunchRequest(
            seed=request.seed,
            output_root=request.output_root,
            data_root=request.data_root,
            num_workers=request.num_workers,
            execute_formal=request.execute_formal,
            resume_checkpoint=resume_checkpoint,
            expected_code_sha=request.expected_code_sha,
            validate_freeze_candidate=request.validate_freeze_candidate,
        )

    dataset, loaders = dataset_preflight(
        request.data_root, num_workers=request.num_workers
    )
    device, environment = target_environment_preflight(REPOSITORY_ROOT)
    configure_target_runtime(request.seed)
    numerical_policy = numerical_policy_snapshot()
    validate_numerical_policy(numerical_policy)

    model = wrn16_8().to(device)
    optimizer = build_formal_sgd(model)
    criterion = build_classification_criterion()
    cursor = TrainingCursor()
    validate_model_optimizer_pre_step(
        model, optimizer, device, cursor, require_initial_cursor=True
    )
    initial_fingerprint = canonical_training_fingerprint(
        model, optimizer, cursor.to_dict()
    )
    manifest = build_run_manifest(
        request=request,
        human_authorized_seed=human_authorized_seed,
        code_sha=code_sha,
        code_baseline_status=code_baseline_status,
        run_id=run_id,
        run_directory=run_directory,
        frozen_hashes=frozen_hashes,
        dataset=dataset,
        environment=environment,
        numerical_policy=numerical_policy,
        initial_training_fingerprint=initial_fingerprint,
        start_timestamp=utc_now(),
    )
    return FormalPreflight(
        request=request,
        code_sha=code_sha,
        run_id=run_id,
        run_directory=run_directory,
        device=device,
        environment=environment,
        numerical_policy=numerical_policy,
        dataset=dataset,
        loaders=loaders,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        cursor=cursor,
        initial_training_fingerprint=initial_fingerprint,
        manifest=manifest,
    )


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(value, temporary, sort_keys=True, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def append_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_artifact_hash_manifest(run_directory: Path) -> dict[str, Any]:
    formal_names = {
        "run_manifest.json",
        "environment.json",
        "training_log.jsonl",
        "run_status.json",
        "final_result.json",
    }
    artifacts = [
        path
        for path in run_directory.rglob("*")
        if path.is_file()
        and (
            path.name in formal_names
            or path.parent.name == "checkpoints" and path.suffix == ".pt"
        )
    ]
    manifest = {
        "schema_version": ARTIFACT_HASH_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "artifacts": {
            path.relative_to(run_directory).as_posix(): sha256_file(path)
            for path in sorted(artifacts)
        },
    }
    atomic_write_json(run_directory / "artifact_hashes.json", manifest)
    return manifest
