from __future__ import annotations

from src.wrn import wrn16_8


EXPECTED_EXACT_TRAINABLE_PARAMETERS = 10_961_370


def test_exact_trainable_parameter_count_and_paper_rounding() -> None:
    model = wrn16_8()
    exact_count = sum(parameter.numel() for parameter in model.parameters())

    assert exact_count == EXPECTED_EXACT_TRAINABLE_PARAMETERS
    assert round(exact_count / 1_000_000, 1) == 11.0


def test_all_parameters_are_trainable_and_running_stats_are_buffers() -> None:
    model = wrn16_8()

    assert all(parameter.requires_grad for parameter in model.parameters())
    parameter_names = dict(model.named_parameters())
    buffer_names = dict(model.named_buffers())
    assert not any("running_mean" in name or "running_var" in name for name in parameter_names)
    assert any("running_mean" in name for name in buffer_names)
    assert any("running_var" in name for name in buffer_names)
