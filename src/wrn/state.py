"""Unambiguous formal training cursor and update invariants."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .schedule import EPOCHS, TOTAL_UPDATES, UPDATES_PER_EPOCH


@dataclass
class TrainingCursor:
    """Point to the next batch; global_update counts completed steps."""

    current_epoch_1: int = 1
    next_batch_index_0: int = 0
    global_update: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not 1 <= self.current_epoch_1 <= EPOCHS + 1:
            raise ValueError("current_epoch_1 must be in [1, 201]")
        if self.current_epoch_1 == EPOCHS + 1:
            if self.next_batch_index_0 != 0:
                raise ValueError("the terminal cursor must have next batch 0")
        elif not 0 <= self.next_batch_index_0 < UPDATES_PER_EPOCH:
            raise ValueError("next_batch_index_0 must be in [0, 389]")
        expected = (
            (self.current_epoch_1 - 1) * UPDATES_PER_EPOCH
            + self.next_batch_index_0
        )
        if self.global_update != expected:
            raise ValueError(
                f"global_update={self.global_update}, expected {expected} from cursor"
            )
        if not 0 <= self.global_update <= TOTAL_UPDATES:
            raise ValueError("global_update outside [0, 78000]")

    @property
    def complete(self) -> bool:
        return self.current_epoch_1 == EPOCHS + 1

    @property
    def completed_epoch(self) -> int:
        return self.current_epoch_1 - 1

    def record_successful_update(self) -> None:
        """Advance once and only after a completed optimizer step."""

        self.validate()
        if self.complete:
            raise RuntimeError("all 78,000 formal updates are already complete")
        self.global_update += 1
        self.next_batch_index_0 += 1
        if self.next_batch_index_0 == UPDATES_PER_EPOCH:
            self.current_epoch_1 += 1
            self.next_batch_index_0 = 0
        self.validate()

    def to_dict(self) -> dict[str, int]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, int]) -> "TrainingCursor":
        return cls(
            current_epoch_1=int(values["current_epoch_1"]),
            next_batch_index_0=int(values["next_batch_index_0"]),
            global_update=int(values["global_update"]),
        )
