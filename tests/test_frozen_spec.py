from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "wrn16_8_arxiv_v4_frozen.yaml"


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _lr_for_epoch(config: dict, epoch: int) -> float:
    for segment in config["lr_schedule"]["segments"]:
        if segment["start_epoch"] <= epoch <= segment["end_epoch"]:
            return segment["lr"]
    raise ValueError(f"epoch outside frozen schedule: {epoch}")


def test_frozen_model_and_result_identity() -> None:
    config = _load_config()

    assert config["depth"] == 16
    assert config["widen_factor"] == 8
    assert config["blocks_per_group"] == 2
    assert config["stage_channels"] == {
        "stem": 16,
        "group1": 128,
        "group2": 256,
        "group3": 512,
    }
    assert config["dropout"] == 0.0
    assert config["reference_error"]["value_percent"] == 4.27
    assert config["paper_parameter_count"]["value_millions"] == 11.0


def test_lr_boundaries_are_start_of_epoch() -> None:
    config = _load_config()

    assert _lr_for_epoch(config, 1) == 0.1
    assert _lr_for_epoch(config, 59) == 0.1
    assert _lr_for_epoch(config, 60) == 0.02
    assert _lr_for_epoch(config, 119) == 0.02
    assert _lr_for_epoch(config, 120) == 0.004
    assert _lr_for_epoch(config, 159) == 0.004
    assert _lr_for_epoch(config, 160) == 0.0008
    assert _lr_for_epoch(config, 200) == 0.0008


def test_epoch_seed_and_iterator_derived_cross_checks() -> None:
    config = _load_config()

    epoch_seed = lambda zero_based_epoch: zero_based_epoch + 1
    assert epoch_seed(0) == 1
    assert epoch_seed(199) == 200
    assert config["run_seeds"] == [1, 2, 3, 4, 5]
    assert config["drop_last"] is True
    assert config["test_include_all"] is True
    assert config["derived_iterator_cross_check"] == {
        "train_size": 50000,
        "test_size": 10000,
        "train_batches_per_epoch": 390,
        "train_samples_consumed_per_epoch": 49920,
        "total_optimizer_updates": 78000,
        "classification": "DERIVED FROM OFFICIAL-CODE-SPECIFIED",
    }
