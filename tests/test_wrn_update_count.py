from __future__ import annotations

import pytest

from src.wrn.state import TrainingCursor


def test_cursor_start_epoch_end_and_final_states() -> None:
    cursor = TrainingCursor()
    assert cursor.to_dict() == {
        "current_epoch_1": 1,
        "next_batch_index_0": 0,
        "global_update": 0,
    }
    for _ in range(390):
        cursor.record_successful_update()
    assert cursor.to_dict() == {
        "current_epoch_1": 2,
        "next_batch_index_0": 0,
        "global_update": 390,
    }

    final = TrainingCursor(200, 389, 77999)
    final.record_successful_update()
    assert final.to_dict() == {
        "current_epoch_1": 201,
        "next_batch_index_0": 0,
        "global_update": 78000,
    }
    assert final.complete
    with pytest.raises(RuntimeError):
        final.record_successful_update()


def test_epoch_completion_invariants_reject_390_as_next_batch() -> None:
    incomplete = TrainingCursor(1, 389, 389)
    assert incomplete.current_epoch_1 == 1
    with pytest.raises(ValueError):
        TrainingCursor(1, 390, 390)
    incomplete.record_successful_update()
    assert incomplete == TrainingCursor(2, 0, 390)


@pytest.mark.parametrize(
    ("epoch", "batch", "update"),
    [(60, 0, 23010), (120, 0, 46410), (160, 0, 62010), (201, 0, 78000)],
)
def test_valid_milestone_cursors(epoch: int, batch: int, update: int) -> None:
    TrainingCursor(epoch, batch, update).validate()


@pytest.mark.parametrize(
    ("epoch", "batch", "update"),
    [(60, 0, 23011), (3, 2, 781), (201, 1, 78001), (1, -1, -1)],
)
def test_inconsistent_cursor_rejected(epoch: int, batch: int, update: int) -> None:
    with pytest.raises(ValueError):
        TrainingCursor(epoch, batch, update)
