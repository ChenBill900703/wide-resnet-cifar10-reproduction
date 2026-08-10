"""Safe-default Phase 5 formal launcher for the frozen WRN CIFAR-10 runs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_checkpoint, save_checkpoint_atomic
from .cuda_validation import assert_optimizer_state_cuda, numerical_policy_snapshot
from .formal_protocol import (
    CHECKPOINT_EPOCHS,
    DATASET_ARCHIVE_SHA256,
    EXPECTED_CHECKPOINT_UPDATES,
    FORMAL_SEEDS,
    FORMAL_RESULT_SCHEMA_VERSION,
    FORMAL_TARGET_NAME,
    FROZEN_CONFIG_SHA256,
    REPOSITORY_ROOT,
    FormalLaunchRequest,
    FormalPreflight,
    FormalPreflightError,
    append_jsonl,
    atomic_write_json,
    perform_formal_preflight,
    sha256_file,
    utc_now,
    validate_model_optimizer_pre_step,
    validate_numerical_policy,
    write_artifact_hash_manifest,
)
from .schedule import TOTAL_UPDATES, UPDATES_PER_EPOCH, end_update, lr_for_epoch
from .state import TrainingCursor
from .training import prepare_epoch, train_step


def _parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed WRN formal launcher; defaults to target preflight only"
        )
    )
    parser.add_argument("--seed", type=int, required=True, choices=FORMAL_SEEDS)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=REPOSITORY_ROOT / "data"
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--execute-formal",
        action="store_true",
        help="execute the formal run only after all authorization gates pass",
    )
    parser.add_argument(
        "--resume",
        dest="resume_checkpoint",
        type=Path,
        help="resume only from the latest compatible checkpoint in this run",
    )
    parser.add_argument(
        "--expected-code-sha",
        help="require the exact frozen Phase 5 full Git commit SHA",
    )
    parser.add_argument(
        "--validate-freeze-candidate",
        action="store_true",
        help=(
            "pre-commit dry-run for the enumerated Phase 5 candidate files; "
            "this mode can never execute training"
        ),
    )
    return parser.parse_args(arguments)


def _request_from_args(arguments: argparse.Namespace) -> FormalLaunchRequest:
    return FormalLaunchRequest(
        seed=arguments.seed,
        output_root=arguments.output_root,
        data_root=arguments.data_root,
        num_workers=arguments.num_workers,
        execute_formal=arguments.execute_formal,
        resume_checkpoint=arguments.resume_checkpoint,
        expected_code_sha=arguments.expected_code_sha,
        validate_freeze_candidate=arguments.validate_freeze_candidate,
    )


def dry_run_summary(preflight: FormalPreflight) -> dict[str, Any]:
    return {
        "status": "PASS",
        "mode": (
            "PHASE5_FREEZE_CANDIDATE_DRY_RUN"
            if preflight.request.validate_freeze_candidate
            else "FROZEN_BASELINE_DRY_RUN"
        ),
        "formal_target": FORMAL_TARGET_NAME,
        "seed": preflight.request.seed,
        "formal_seed_set": list(FORMAL_SEEDS),
        "currently_human_authorized_seed": preflight.manifest[
            "human_authorized_seed"
        ],
        "code_sha": preflight.code_sha,
        "code_baseline_status": preflight.manifest["code_baseline_status"],
        "frozen_config_sha256": FROZEN_CONFIG_SHA256,
        "frozen_artifact_sha256": preflight.manifest[
            "frozen_artifact_sha256"
        ],
        "dataset_archive_sha256": DATASET_ARCHIVE_SHA256,
        "dataset_archive_actual_sha256": preflight.dataset[
            "archive_actual_sha256"
        ],
        "dataset_integrity": preflight.dataset["official_split_integrity"],
        "environment": preflight.environment,
        "installed_package_inventory": preflight.manifest[
            "installed_package_inventory"
        ],
        "runtime_policy": preflight.manifest["numerical_policy"],
        "parameter_count": preflight.manifest["recipe"]["model"][
            "trainable_parameters"
        ],
        "initial_training_fingerprint": preflight.initial_training_fingerprint,
        "planned_epochs": preflight.manifest["recipe"]["epochs"],
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "planned_updates": TOTAL_UPDATES,
        "checkpoint_plan": EXPECTED_CHECKPOINT_UPDATES,
        "result_selection": "epoch_200_final_checkpoint_only",
        "output_directory": str(preflight.run_directory),
        "training_executed": False,
        "cifar_backward_count": 0,
        "cifar_optimizer_step_count": 0,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise FormalPreflightError(f"expected a JSON object in {path.name}")
    return value


def _validate_existing_manifest(
    existing: dict[str, Any], preflight: FormalPreflight
) -> None:
    expected = {
        "formal_target_name": FORMAL_TARGET_NAME,
        "formal_run_id": preflight.run_id,
        "seed": preflight.request.seed,
        "formal_seed_set": list(FORMAL_SEEDS),
        "code_commit_sha": preflight.code_sha,
        "code_baseline_status": "FROZEN_COMMIT",
        "frozen_config_sha256": FROZEN_CONFIG_SHA256,
        "initial_training_fingerprint": preflight.initial_training_fingerprint,
    }
    mismatches = {
        name: {"actual": existing.get(name), "expected": value}
        for name, value in expected.items()
        if existing.get(name) != value
    }
    if mismatches:
        raise FormalPreflightError(f"existing run manifest mismatch: {mismatches}")
    dataset = existing.get("dataset", {})
    if dataset.get("archive_sha256") != DATASET_ARCHIVE_SHA256:
        raise FormalPreflightError("existing run manifest dataset identity mismatch")


def _initialize_or_resume(preflight: FormalPreflight) -> tuple[TrainingCursor, int]:
    run_directory = preflight.run_directory
    manifest_path = run_directory / "run_manifest.json"
    environment_path = run_directory / "environment.json"
    status_path = run_directory / "run_status.json"
    log_path = run_directory / "training_log.jsonl"

    if preflight.request.resume_checkpoint is None:
        run_directory.mkdir(parents=True, exist_ok=False)
        (run_directory / "checkpoints").mkdir()
        atomic_write_json(manifest_path, preflight.manifest)
        atomic_write_json(
            environment_path,
            {
                "schema_version": 1,
                "environment": preflight.environment,
                "installed_package_inventory": preflight.manifest[
                    "installed_package_inventory"
                ],
                "numerical_policy": preflight.manifest["numerical_policy"],
            },
        )
        status = {
            "schema_version": 1,
            "status": "READY",
            "seed": preflight.request.seed,
            "resume_count": 0,
            "cursor": preflight.cursor.to_dict(),
            "updated_at_utc": utc_now(),
        }
        atomic_write_json(status_path, status)
        append_jsonl(
            log_path,
            {
                "event": "FORMAL_RUN_CREATED",
                "timestamp_utc": utc_now(),
                "seed": preflight.request.seed,
                "code_sha": preflight.code_sha,
                "cursor": preflight.cursor.to_dict(),
            },
        )
        return preflight.cursor, 0

    if not manifest_path.is_file() or not environment_path.is_file():
        raise FormalPreflightError("resume run is missing manifest/environment evidence")
    existing_manifest = _read_json(manifest_path)
    _validate_existing_manifest(existing_manifest, preflight)
    existing_environment = _read_json(environment_path)
    if existing_environment.get("environment") != preflight.environment:
        raise FormalPreflightError("resume environment differs from initial run")
    if existing_environment.get("installed_package_inventory") != preflight.manifest[
        "installed_package_inventory"
    ]:
        raise FormalPreflightError("resume package inventory differs from initial run")
    if existing_environment.get("numerical_policy") != preflight.manifest[
        "numerical_policy"
    ]:
        raise FormalPreflightError("resume numerical policy differs from initial run")
    preflight.manifest = existing_manifest

    cursor, checkpoint_seed = load_checkpoint(
        preflight.request.resume_checkpoint,
        preflight.model,
        preflight.optimizer,
        expected_run_seed=preflight.request.seed,
        expected_code_commit_sha=preflight.code_sha,
        expected_formal_run_id=preflight.run_id,
    )
    if checkpoint_seed != preflight.request.seed:
        raise FormalPreflightError("resume checkpoint seed mismatch")
    assert_optimizer_state_cuda(preflight.optimizer, preflight.device)
    validate_model_optimizer_pre_step(
        preflight.model,
        preflight.optimizer,
        preflight.device,
        cursor,
        require_initial_cursor=False,
    )

    status = _read_json(status_path) if status_path.is_file() else {}
    resume_count = int(status.get("resume_count", 0)) + 1
    atomic_write_json(
        status_path,
        {
            "schema_version": 1,
            "status": "RESUMING",
            "seed": preflight.request.seed,
            "resume_count": resume_count,
            "cursor": cursor.to_dict(),
            "checkpoint": str(
                preflight.request.resume_checkpoint.relative_to(run_directory)
            ).replace("\\", "/"),
            "updated_at_utc": utc_now(),
        },
    )
    append_jsonl(
        log_path,
        {
            "event": "FORMAL_RUN_RESUMED",
            "timestamp_utc": utc_now(),
            "seed": preflight.request.seed,
            "resume_count": resume_count,
            "cursor": cursor.to_dict(),
            "checkpoint": str(
                preflight.request.resume_checkpoint.relative_to(run_directory)
            ).replace("\\", "/"),
        },
    )
    return cursor, resume_count


def _save_formal_checkpoint(
    preflight: FormalPreflight, cursor: TrainingCursor, completed_epoch: int
) -> tuple[Path, str]:
    expected_update = EXPECTED_CHECKPOINT_UPDATES[completed_epoch]
    if cursor.completed_epoch != completed_epoch or cursor.next_batch_index_0 != 0:
        raise RuntimeError("formal checkpoint cursor is not at the epoch boundary")
    if cursor.global_update != expected_update:
        raise RuntimeError(
            f"epoch {completed_epoch} checkpoint update {cursor.global_update} "
            f"!= {expected_update}"
        )
    path = preflight.run_directory / "checkpoints" / f"epoch_{completed_epoch:03d}.pt"
    save_checkpoint_atomic(
        path,
        preflight.model,
        preflight.optimizer,
        cursor,
        run_seed=preflight.request.seed,
        code_commit_sha=preflight.code_sha,
        formal_run_id=preflight.run_id,
    )
    return path, sha256_file(path)


def _evaluate_epoch_200(preflight: FormalPreflight) -> dict[str, Any]:
    preflight.model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for cpu_inputs, cpu_labels in preflight.loaders.test:
            inputs = cpu_inputs.to(preflight.device)
            labels = cpu_labels.to(preflight.device)
            logits = preflight.model(inputs)
            if logits.shape != (inputs.shape[0], 10) or not torch.isfinite(logits).all():
                raise RuntimeError("non-finite or invalid epoch-200 test logits")
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    if total != 10_000:
        raise RuntimeError(f"formal test evaluation covered {total}, expected 10,000")
    incorrect = total - correct
    return {
        "test_sample_count": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": correct / total,
        "test_error": incorrect / total,
        "accuracy_percent": 100.0 * correct / total,
        "test_error_percent": 100.0 * incorrect / total,
    }


def _write_status(
    preflight: FormalPreflight,
    cursor: TrainingCursor,
    resume_count: int,
    status: str,
    **extra: Any,
) -> None:
    atomic_write_json(
        preflight.run_directory / "run_status.json",
        {
            "schema_version": 1,
            "status": status,
            "seed": preflight.request.seed,
            "resume_count": resume_count,
            "cursor": cursor.to_dict(),
            "updated_at_utc": utc_now(),
            **extra,
        },
    )


def _write_incomplete_result(
    preflight: FormalPreflight,
    cursor: TrainingCursor,
    resume_count: int,
    status: str,
    *,
    error: BaseException | None = None,
) -> None:
    checkpoints = sorted((preflight.run_directory / "checkpoints").glob("epoch_*.pt"))
    latest = checkpoints[-1] if checkpoints else None
    atomic_write_json(
        preflight.run_directory / "final_result.json",
        {
            "schema_version": FORMAL_RESULT_SCHEMA_VERSION,
            "status": status,
            "formal_target_name": FORMAL_TARGET_NAME,
            "formal_run_id": preflight.run_id,
            "seed": preflight.request.seed,
            "code_commit_sha": preflight.code_sha,
            "frozen_config_sha256": FROZEN_CONFIG_SHA256,
            "dataset_archive_sha256": DATASET_ARCHIVE_SHA256,
            "epochs_completed": cursor.completed_epoch,
            "global_updates": cursor.global_update,
            "final_lr": float(preflight.optimizer.param_groups[0]["lr"]),
            "final_checkpoint_path": (
                latest.relative_to(preflight.run_directory).as_posix()
                if latest is not None
                else None
            ),
            "final_checkpoint_sha256": (
                sha256_file(latest) if latest is not None else None
            ),
            "test_sample_count": None,
            "correct": None,
            "incorrect": None,
            "accuracy": None,
            "test_error": None,
            "run_start_timestamp_utc": preflight.manifest["start_timestamp_utc"],
            "run_end_timestamp_utc": utc_now(),
            "resume_count": resume_count,
            "result_selection": "epoch_200_final_checkpoint_only",
            "error_type": type(error).__name__ if error is not None else None,
            "error_message": str(error) if error is not None else None,
        },
    )


def execute_formal_training(preflight: FormalPreflight) -> dict[str, Any]:
    if not preflight.request.execute_formal:
        raise FormalPreflightError("formal execution flag is absent")
    if preflight.request.validate_freeze_candidate:
        raise FormalPreflightError("candidate validation mode can never train")

    cursor, resume_count = _initialize_or_resume(preflight)
    preflight.cursor = cursor
    log_path = preflight.run_directory / "training_log.jsonl"
    _write_status(preflight, cursor, resume_count, "RUNNING")

    try:
        while not cursor.complete:
            epoch = cursor.current_epoch_1
            start_batch = cursor.next_batch_index_0
            if start_batch == 0:
                learning_rate, data_seed = prepare_epoch(
                    preflight.optimizer, preflight.loaders, cursor
                )
            else:
                preflight.loaders.set_epoch(epoch - 1)
                learning_rate = lr_for_epoch(epoch)
                data_seed = epoch
                if {group["lr"] for group in preflight.optimizer.param_groups} != {
                    learning_rate
                }:
                    raise RuntimeError("resume LR does not match current epoch")

            validate_numerical_policy(numerical_policy_snapshot())
            validate_model_optimizer_pre_step(
                preflight.model,
                preflight.optimizer,
                preflight.device,
                cursor,
                require_initial_cursor=cursor.global_update == 0,
            )
            epoch_started = time.perf_counter()
            weighted_loss_sum = 0.0
            sample_count = 0
            completed_batches = 0

            for batch_index, (cpu_inputs, cpu_labels) in enumerate(
                preflight.loaders.train
            ):
                if batch_index < start_batch:
                    continue
                if batch_index != cursor.next_batch_index_0:
                    raise RuntimeError("loader batch index and training cursor diverged")
                inputs = cpu_inputs.to(preflight.device)
                labels = cpu_labels.to(preflight.device)
                if inputs.dtype != torch.float32 or labels.dtype != torch.int64:
                    raise RuntimeError("formal CIFAR batch dtype mismatch")
                before_update = cursor.global_update
                batch_size = int(labels.numel())
                loss = train_step(
                    preflight.model,
                    preflight.optimizer,
                    preflight.criterion,
                    inputs,
                    labels,
                    cursor,
                )
                if cursor.global_update != before_update + 1:
                    raise RuntimeError("formal global update did not advance exactly once")
                if before_update == 0:
                    append_jsonl(
                        log_path,
                        {
                            "event": "FORMAL_TRAINING_STARTED",
                            "timestamp_utc": utc_now(),
                            "seed": preflight.request.seed,
                            "epoch": epoch,
                            "batch_index_0": batch_index,
                            "global_update_before": 0,
                            "global_update_after": 1,
                            "code_sha": preflight.code_sha,
                            "frozen_config_sha256": FROZEN_CONFIG_SHA256,
                        },
                    )
                weighted_loss_sum += loss * batch_size
                sample_count += batch_size
                completed_batches += 1

            if cursor.current_epoch_1 != epoch + 1 or cursor.next_batch_index_0 != 0:
                raise RuntimeError("formal epoch did not finish at the next epoch boundary")
            if completed_batches != UPDATES_PER_EPOCH - start_batch:
                raise RuntimeError("formal epoch completed an unexpected batch count")
            if cursor.global_update != end_update(epoch):
                raise RuntimeError("formal epoch-end global update mismatch")
            append_jsonl(
                log_path,
                {
                    "event": "EPOCH_COMPLETED",
                    "timestamp_utc": utc_now(),
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "epoch_data_seed": data_seed,
                    "start_batch_index_0": start_batch,
                    "completed_batches_this_process": completed_batches,
                    "global_update": cursor.global_update,
                    "training_loss_sample_weighted_mean_this_process": (
                        weighted_loss_sum / sample_count
                    ),
                    "samples_this_process": sample_count,
                    "elapsed_seconds_this_process": time.perf_counter()
                    - epoch_started,
                },
            )
            _write_status(preflight, cursor, resume_count, "RUNNING")

            if epoch in CHECKPOINT_EPOCHS:
                checkpoint_path, checkpoint_hash = _save_formal_checkpoint(
                    preflight, cursor, epoch
                )
                append_jsonl(
                    log_path,
                    {
                        "event": "CHECKPOINT_SAVED",
                        "timestamp_utc": utc_now(),
                        "epoch": epoch,
                        "global_update": cursor.global_update,
                        "path": checkpoint_path.relative_to(
                            preflight.run_directory
                        ).as_posix(),
                        "sha256": checkpoint_hash,
                    },
                )

        if cursor.global_update != TOTAL_UPDATES:
            raise RuntimeError("formal run did not finish with 78,000 updates")
        final_checkpoint = (
            preflight.run_directory / "checkpoints" / "epoch_200.pt"
        )
        if not final_checkpoint.is_file():
            raise RuntimeError("formal epoch-200 checkpoint is missing")
        evaluation = _evaluate_epoch_200(preflight)
        result = {
            "schema_version": FORMAL_RESULT_SCHEMA_VERSION,
            "status": "COMPLETED",
            "formal_target_name": FORMAL_TARGET_NAME,
            "formal_run_id": preflight.run_id,
            "seed": preflight.request.seed,
            "code_commit_sha": preflight.code_sha,
            "frozen_config_sha256": FROZEN_CONFIG_SHA256,
            "dataset_archive_sha256": DATASET_ARCHIVE_SHA256,
            "epochs_completed": 200,
            "global_updates": cursor.global_update,
            "final_lr": float(preflight.optimizer.param_groups[0]["lr"]),
            "final_checkpoint_path": final_checkpoint.relative_to(
                preflight.run_directory
            ).as_posix(),
            "final_checkpoint_sha256": sha256_file(final_checkpoint),
            **evaluation,
            "run_start_timestamp_utc": preflight.manifest["start_timestamp_utc"],
            "run_end_timestamp_utc": utc_now(),
            "resume_count": resume_count,
            "result_selection": "epoch_200_final_checkpoint_only",
            "best_test_metric": None,
        }
        atomic_write_json(preflight.run_directory / "final_result.json", result)
        _write_status(preflight, cursor, resume_count, "COMPLETED")
        append_jsonl(
            log_path,
            {
                "event": "FORMAL_RUN_COMPLETED",
                "timestamp_utc": utc_now(),
                "seed": preflight.request.seed,
                "global_update": cursor.global_update,
                "result_selection": "epoch_200_final_checkpoint_only",
            },
        )
        write_artifact_hash_manifest(preflight.run_directory)
        return result
    except KeyboardInterrupt:
        _write_status(preflight, cursor, resume_count, "INTERRUPTED")
        _write_incomplete_result(
            preflight, cursor, resume_count, "INTERRUPTED"
        )
        append_jsonl(
            log_path,
            {
                "event": "FORMAL_RUN_INTERRUPTED",
                "timestamp_utc": utc_now(),
                "cursor": cursor.to_dict(),
            },
        )
        raise
    except Exception as error:
        _write_status(
            preflight,
            cursor,
            resume_count,
            "FAILED",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        _write_incomplete_result(
            preflight,
            cursor,
            resume_count,
            "FAILED",
            error=error,
        )
        append_jsonl(
            log_path,
            {
                "event": "FORMAL_RUN_FAILED",
                "timestamp_utc": utc_now(),
                "cursor": cursor.to_dict(),
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        raise


def main(arguments: list[str] | None = None) -> int:
    parsed = _parse_args(arguments)
    request = _request_from_args(parsed)
    try:
        preflight = perform_formal_preflight(request)
        if not request.execute_formal:
            print("PHASE5_DRY_RUN=" + json.dumps(dry_run_summary(preflight), sort_keys=True))
            return 0
        result = execute_formal_training(preflight)
        print("PHASE5_FORMAL_RESULT=" + json.dumps(result, sort_keys=True))
        return 0
    except FormalPreflightError as error:
        print(f"PHASE5_PREFLIGHT_FAILED={error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
