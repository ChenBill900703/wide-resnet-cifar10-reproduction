"""Fresh-process synthetic CUDA fingerprint probe for Phase 4."""

from __future__ import annotations

import argparse
import json

import torch

from .cuda_validation import (
    canonical_training_fingerprint,
    configure_target_runtime,
    require_target_cuda_device,
    target_environment_snapshot,
)
from .model import wrn16_8
from .optimization import build_formal_sgd
from .state import TrainingCursor
from .training import build_classification_criterion, train_step


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic-only RTX 3070 Ti deterministic fingerprint probe"
    )
    parser.add_argument("--seed", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--steps", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--device-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = require_target_cuda_device(args.device_index)
    configure_target_runtime(args.seed)
    model = wrn16_8().to(device)
    optimizer = build_formal_sgd(model)
    criterion = build_classification_criterion()
    cursor = TrainingCursor()
    for _ in range(args.steps):
        inputs = torch.randn(2, 3, 32, 32, device=device, dtype=torch.float32)
        labels = torch.randint(0, 10, (2,), device=device, dtype=torch.int64)
        train_step(model, optimizer, criterion, inputs, labels, cursor)
    result = {
        "environment": target_environment_snapshot(device),
        "seed": args.seed,
        "steps": args.steps,
        "cursor": cursor.to_dict(),
        "fingerprint": canonical_training_fingerprint(
            model, optimizer, cursor.to_dict()
        ),
        "cifar_optimizer_steps": 0,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
