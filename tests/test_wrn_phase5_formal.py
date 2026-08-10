from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import src.wrn.formal_protocol as protocol
from src.wrn.checkpoint import (
    CheckpointCompatibilityError,
    load_checkpoint,
    save_checkpoint_atomic,
)
from src.wrn.formal_launcher import _parse_args, main
from src.wrn.formal_protocol import (
    CHECKPOINT_EPOCHS,
    CURRENT_HUMAN_AUTHORIZED_SEED,
    DATASET_ARCHIVE_SHA256,
    EXPECTED_CHECKPOINT_UPDATES,
    EXPECTED_TARGET_ENVIRONMENT,
    FORMAL_SEEDS,
    PHASE5_CANDIDATE_PATHS,
    FormalLaunchRequest,
    FormalPreflightError,
    audit_frozen_artifacts,
    build_run_manifest,
    dataset_preflight,
    require_execute_authorization,
    resolve_formal_run_directory,
    perform_formal_preflight,
    validate_environment_snapshot,
    validate_formal_seed,
    validate_git_status_lines,
    validate_model_optimizer_pre_step,
    validate_numerical_policy,
    validate_run_directory_state,
    write_artifact_hash_manifest,
)
from src.wrn.optimization import build_formal_sgd
from src.wrn.schedule import TOTAL_UPDATES, UPDATES_PER_EPOCH, end_update
from src.wrn.state import TrainingCursor


ROOT = Path(__file__).resolve().parents[1]
FULL_SHA = "a" * 40


def test_formal_seed_set_and_current_authorization_are_exact() -> None:
    assert FORMAL_SEEDS == (1, 2, 3, 4, 5)
    assert CURRENT_HUMAN_AUTHORIZED_SEED == 1
    for seed in FORMAL_SEEDS:
        assert validate_formal_seed(seed) == seed
    for invalid in (-1, 0, 6, 99):
        with pytest.raises(FormalPreflightError, match="formal seed"):
            validate_formal_seed(invalid)


def test_execute_authorization_requires_flag_and_matching_seed() -> None:
    authorized = {
        "WRN_FORMAL_TRAINING_AUTHORIZED": "1",
        "WRN_FORMAL_AUTHORIZED_SEED": "1",
    }
    assert require_execute_authorization(1, authorized) == 1
    with pytest.raises(FormalPreflightError, match="AUTHORIZED=1"):
        require_execute_authorization(1, {})
    with pytest.raises(FormalPreflightError, match="does not match"):
        require_execute_authorization(2, authorized)
    assert require_execute_authorization(
        2,
        {
            "WRN_FORMAL_TRAINING_AUTHORIZED": "1",
            "WRN_FORMAL_AUTHORIZED_SEED": "2",
        },
    ) == 2
    with pytest.raises(FormalPreflightError, match="frozen seeds"):
        require_execute_authorization(
            1,
            {
                "WRN_FORMAL_TRAINING_AUTHORIZED": "1",
                "WRN_FORMAL_AUTHORIZED_SEED": "6",
            },
        )


def test_launcher_parser_defaults_to_dry_run_and_supports_all_seeds(tmp_path: Path) -> None:
    for seed in FORMAL_SEEDS:
        parsed = _parse_args(
            ["--seed", str(seed), "--output-root", str(tmp_path.resolve())]
        )
        assert parsed.seed == seed
        assert parsed.execute_formal is False
        assert parsed.resume_checkpoint is None
    parsed_execute = _parse_args(
        [
            "--seed",
            "1",
            "--output-root",
            str(tmp_path.resolve()),
            "--execute-formal",
        ]
    )
    assert parsed_execute.execute_formal is True


def test_freeze_candidate_mode_can_never_execute(tmp_path: Path) -> None:
    request = FormalLaunchRequest(
        seed=1,
        output_root=tmp_path.resolve(),
        data_root=ROOT / "data",
        execute_formal=True,
        validate_freeze_candidate=True,
    )
    with pytest.raises(FormalPreflightError, match="can never execute"):
        perform_formal_preflight(request)


def test_main_default_dry_run_never_calls_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = {"execute": 0}
    monkeypatch.setattr(
        "src.wrn.formal_launcher.perform_formal_preflight",
        lambda _request: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "src.wrn.formal_launcher.dry_run_summary",
        lambda _preflight: {
            "training_executed": False,
            "cifar_backward_count": 0,
            "cifar_optimizer_step_count": 0,
        },
    )

    def forbidden_execute(_preflight):
        calls["execute"] += 1
        raise AssertionError("dry-run called formal execution")

    monkeypatch.setattr(
        "src.wrn.formal_launcher.execute_formal_training", forbidden_execute
    )
    result = main(
        ["--seed", "1", "--output-root", str(tmp_path.resolve())]
    )
    assert result == 0
    assert calls["execute"] == 0
    output = capsys.readouterr().out
    assert '"training_executed": false' in output


def test_output_root_and_existing_run_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(FormalPreflightError, match="absolute"):
        resolve_formal_run_directory(ROOT, Path("relative"), 1)
    with pytest.raises(FormalPreflightError, match="outside"):
        resolve_formal_run_directory(ROOT, ROOT / "formal-output", 1)

    external = tmp_path.resolve()
    run_directory, run_id = resolve_formal_run_directory(ROOT, external, 1)
    assert run_id.endswith("/seed_01")
    assert not run_directory.is_relative_to(ROOT.resolve())
    run_directory.mkdir(parents=True)
    (run_directory / "existing.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FormalPreflightError, match="already exists"):
        validate_run_directory_state(run_directory, None)


def test_resume_is_separate_and_must_use_latest_checkpoint(tmp_path: Path) -> None:
    run_directory = tmp_path / "target" / "seed_01"
    checkpoints = run_directory / "checkpoints"
    checkpoints.mkdir(parents=True)
    epoch60 = checkpoints / "epoch_060.pt"
    epoch120 = checkpoints / "epoch_120.pt"
    epoch60.write_bytes(b"60")
    epoch120.write_bytes(b"120")
    with pytest.raises(FormalPreflightError, match="latest"):
        validate_run_directory_state(run_directory, epoch60)
    assert validate_run_directory_state(run_directory, epoch120) == epoch120.resolve()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    with pytest.raises(FormalPreflightError, match="under this run"):
        validate_run_directory_state(run_directory, outside)

    result = run_directory / "final_result.json"
    result.write_text(json.dumps({"status": "COMPLETED"}), encoding="utf-8")
    with pytest.raises(FormalPreflightError, match="completed"):
        validate_run_directory_state(run_directory, epoch120)
    result.write_text(json.dumps({"status": "INTERRUPTED"}), encoding="utf-8")
    assert validate_run_directory_state(run_directory, epoch120) == epoch120.resolve()


def test_git_gate_rejects_dirty_or_arbitrary_untracked_content() -> None:
    validate_git_status_lines(["?? references/local.pdf"], allow_freeze_candidate=False)
    with pytest.raises(FormalPreflightError, match="not clean"):
        validate_git_status_lines([" M src/wrn/model.py"], allow_freeze_candidate=False)
    with pytest.raises(FormalPreflightError, match="unexpected untracked"):
        validate_git_status_lines(["?? src/unreviewed.py"], allow_freeze_candidate=False)

    candidate = sorted(PHASE5_CANDIDATE_PATHS)[0]
    validate_git_status_lines(
        [f" M {candidate}", "?? references/local.pdf"],
        allow_freeze_candidate=True,
    )
    with pytest.raises(FormalPreflightError, match="unexpected changes"):
        validate_git_status_lines(
            [" M configs/wrn16_8_arxiv_v4_frozen.yaml"],
            allow_freeze_candidate=True,
        )


def test_frozen_hash_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "frozen.yaml"
    artifact.write_text("tampered", encoding="utf-8")
    monkeypatch.setattr(
        protocol,
        "FROZEN_ARTIFACT_SHA256",
        {"frozen.yaml": "0" * 64},
    )
    with pytest.raises(FormalPreflightError, match="hash mismatch"):
        audit_frozen_artifacts(tmp_path)


def test_dataset_archive_mismatch_rejected_before_loading(tmp_path: Path) -> None:
    archive = tmp_path / protocol.DATASET_ARCHIVE_NAME
    archive.write_bytes(b"wrong archive")
    with pytest.raises(FormalPreflightError, match="archive hash mismatch"):
        dataset_preflight(tmp_path)


def test_formal_dataset_preflight_never_downloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Dataset:
        def __init__(self, length: int) -> None:
            self.length = length

        def __len__(self) -> int:
            return self.length

        def integrity_ok(self) -> bool:
            return True

    calls: list[bool] = []

    def fake_build(_root, *, num_workers: int, download: bool):
        assert num_workers == 0
        calls.append(download)
        return SimpleNamespace(
            train_dataset=Dataset(50_000), test_dataset=Dataset(10_000)
        )

    monkeypatch.setattr(protocol, "build_cifar10_loaders", fake_build)
    snapshot, _loaders = dataset_preflight(tmp_path)
    assert calls == [False]
    assert snapshot["download_performed"] is False
    assert snapshot["archive_sha256"] == DATASET_ARCHIVE_SHA256


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cuda_available", False),
        ("gpu_name", "Wrong GPU"),
        ("torch", "0.0"),
        ("torchvision", "0.0"),
        ("numpy", "0.0"),
        ("pyyaml", "0.0"),
        ("torch_cuda_build", None),
        ("cudnn", None),
        ("nvidia_driver", "wrong"),
        ("compute_capability", [0, 0]),
    ],
)
def test_wrong_formal_environment_rejected(field: str, value: object) -> None:
    environment = dict(EXPECTED_TARGET_ENVIRONMENT)
    environment[field] = value
    with pytest.raises(FormalPreflightError, match="environment mismatch"):
        validate_environment_snapshot(environment)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("autocast_cuda_enabled", True),
        ("tf32_enabled", True),
        ("cudnn_benchmark", True),
        ("cudnn_deterministic", False),
        ("deterministic_algorithms", False),
        ("deterministic_warn_only", True),
        ("default_dtype", "torch.float16"),
    ],
)
def test_wrong_numerical_policy_rejected(field: str, value: object) -> None:
    policy = {
        "default_dtype": "torch.float32",
        "autocast_cuda_enabled": False,
        "tf32_enabled": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
    }
    policy[field] = value
    with pytest.raises(FormalPreflightError, match="policy mismatch"):
        validate_numerical_policy(policy)


def test_model_count_and_optimizer_semantics_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(4, 3)
    optimizer = build_formal_sgd(model)
    device = torch.device("cuda", 0)
    monkeypatch.setattr(
        protocol,
        "audit_model_cuda_device",
        lambda _model, _device: protocol.EXPECTED_TRAINABLE_PARAMETERS,
    )
    assert validate_model_optimizer_pre_step(
        model,
        optimizer,
        device,
        TrainingCursor(),
        require_initial_cursor=True,
    ) == protocol.EXPECTED_TRAINABLE_PARAMETERS

    optimizer.param_groups[0]["weight_decay"] = 0.0
    with pytest.raises(FormalPreflightError, match="optimizer"):
        validate_model_optimizer_pre_step(
            model,
            optimizer,
            device,
            TrainingCursor(),
            require_initial_cursor=True,
        )
    optimizer.param_groups[0]["weight_decay"] = protocol.WEIGHT_DECAY
    monkeypatch.setattr(
        protocol,
        "audit_model_cuda_device",
        lambda _model, _device: protocol.EXPECTED_TRAINABLE_PARAMETERS - 1,
    )
    with pytest.raises(FormalPreflightError, match="parameter count"):
        validate_model_optimizer_pre_step(
            model,
            optimizer,
            device,
            TrainingCursor(),
            require_initial_cursor=True,
        )


def test_update_and_checkpoint_plan_is_exact() -> None:
    assert UPDATES_PER_EPOCH == 390
    assert TOTAL_UPDATES == 78_000
    assert CHECKPOINT_EPOCHS == (60, 120, 160, 200)
    assert EXPECTED_CHECKPOINT_UPDATES == {
        60: 23_400,
        120: 46_800,
        160: 62_400,
        200: 78_000,
    }
    assert end_update(59) == 23_010
    assert end_update(119) == 46_410
    assert end_update(159) == 62_010


def test_manifest_binds_required_formal_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(protocol, "package_inventory", lambda: ["example==1.0"])
    request = FormalLaunchRequest(
        seed=1,
        output_root=tmp_path,
        data_root=ROOT / "data",
    )
    manifest = build_run_manifest(
        request=request,
        human_authorized_seed=1,
        code_sha=FULL_SHA,
        code_baseline_status="FROZEN_COMMIT",
        run_id="target/seed_01",
        run_directory=tmp_path / "target" / "seed_01",
        frozen_hashes={"config": protocol.FROZEN_CONFIG_SHA256},
        dataset={"archive_sha256": DATASET_ARCHIVE_SHA256},
        environment=dict(EXPECTED_TARGET_ENVIRONMENT),
        numerical_policy={
            "default_dtype": "torch.float32",
            "autocast_cuda_enabled": False,
            "tf32_enabled": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
        },
        initial_training_fingerprint="f" * 64,
        start_timestamp="2026-08-10T00:00:00+00:00",
    )
    assert manifest["seed"] == 1
    assert manifest["formal_seed_set"] == [1, 2, 3, 4, 5]
    assert manifest["code_commit_sha"] == FULL_SHA
    assert manifest["frozen_config_sha256"] == protocol.FROZEN_CONFIG_SHA256
    assert manifest["dataset"]["archive_sha256"] == DATASET_ARCHIVE_SHA256
    assert manifest["environment"]["gpu_name"] == protocol.TARGET_GPU_NAME
    assert manifest["initial_training_fingerprint"] == "f" * 64
    recipe = manifest["recipe"]
    assert recipe["updates_per_epoch"] == 390
    assert recipe["total_planned_updates"] == 78_000
    assert recipe["checkpoint_epochs"] == [60, 120, 160, 200]
    assert recipe["data_loader_workers"] == 2
    assert recipe["result_selection"] == "epoch_200_final_checkpoint_only"
    assert recipe["best_checkpoint_selection"] is False
    assert manifest["numerical_policy"]["amp"] is False
    assert manifest["numerical_policy"]["compile"] is False


def test_formal_checkpoint_binds_code_run_and_seed(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = nn.Linear(4, 3)
    optimizer = build_formal_sgd(model)
    checkpoint = tmp_path / "formal.pt"
    save_checkpoint_atomic(
        checkpoint,
        model,
        optimizer,
        TrainingCursor(),
        run_seed=1,
        code_commit_sha=FULL_SHA,
        formal_run_id="target/seed_01",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["code_commit_sha"] == FULL_SHA
    assert payload["formal_run_id"] == "target/seed_01"

    resumed_model = nn.Linear(4, 3)
    resumed_optimizer = build_formal_sgd(resumed_model)
    with pytest.raises(CheckpointCompatibilityError, match="code_commit_sha"):
        load_checkpoint(
            checkpoint,
            resumed_model,
            resumed_optimizer,
            expected_run_seed=1,
            expected_code_commit_sha="b" * 40,
            expected_formal_run_id="target/seed_01",
        )
    with pytest.raises(CheckpointCompatibilityError, match="formal_run_id"):
        load_checkpoint(
            checkpoint,
            resumed_model,
            resumed_optimizer,
            expected_run_seed=1,
            expected_code_commit_sha=FULL_SHA,
            expected_formal_run_id="target/seed_02",
        )
    with pytest.raises(CheckpointCompatibilityError, match="run_seed"):
        load_checkpoint(
            checkpoint,
            resumed_model,
            resumed_optimizer,
            expected_run_seed=2,
            expected_code_commit_sha=FULL_SHA,
            expected_formal_run_id="target/seed_01",
        )


def test_package_inventory_is_plain_name_version_without_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = [
        SimpleNamespace(metadata={"Name": "PackageB"}, version="2.0"),
        SimpleNamespace(metadata={"Name": "package-a"}, version="1.0"),
    ]
    monkeypatch.setattr(protocol.importlib.metadata, "distributions", lambda: fake)
    inventory = protocol.package_inventory()
    assert inventory == ["package-a==1.0", "PackageB==2.0"]
    assert all("@" not in entry and "://" not in entry for entry in inventory)


def test_manifest_round_trip_is_json_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(protocol, "package_inventory", lambda: [])
    manifest = build_run_manifest(
        request=FormalLaunchRequest(1, tmp_path, ROOT / "data"),
        human_authorized_seed=1,
        code_sha=FULL_SHA,
        code_baseline_status="FROZEN_COMMIT",
        run_id="target/seed_01",
        run_directory=tmp_path / "target" / "seed_01",
        frozen_hashes={},
        dataset={"archive_sha256": DATASET_ARCHIVE_SHA256},
        environment=dict(EXPECTED_TARGET_ENVIRONMENT),
        numerical_policy={},
        initial_training_fingerprint="f" * 64,
        start_timestamp="2026-08-10T00:00:00+00:00",
    )
    assert json.loads(json.dumps(manifest)) == manifest


def test_artifact_hash_manifest_selects_only_formal_evidence(tmp_path: Path) -> None:
    for name in (
        "run_manifest.json",
        "environment.json",
        "training_log.jsonl",
        "run_status.json",
        "final_result.json",
    ):
        (tmp_path / name).write_text(name, encoding="utf-8")
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "epoch_200.pt").write_bytes(b"checkpoint")
    (tmp_path / "temporary.bin").write_bytes(b"not formal evidence")

    manifest = write_artifact_hash_manifest(tmp_path)

    assert set(manifest["artifacts"]) == {
        "run_manifest.json",
        "environment.json",
        "training_log.jsonl",
        "run_status.json",
        "final_result.json",
        "checkpoints/epoch_200.pt",
    }
    written = json.loads(
        (tmp_path / "artifact_hashes.json").read_text(encoding="utf-8")
    )
    assert written == manifest
    assert not list(tmp_path.glob("*.tmp"))
