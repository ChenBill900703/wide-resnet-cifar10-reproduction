"""Initialization for the frozen paper-era-semantics WRN port."""

from __future__ import annotations

import math

from torch import nn


def initialize_wrn(module: nn.Module) -> None:
    """Apply the frozen WRN initialization policy in place.

    Convolution initialization is directly specified by the pinned Torch7 WRN
    utilities. Linear and BatchNorm defaults are the explicit, human-approved
    historical-dependency approximation recorded in the frozen specification.
    """

    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            kernel_height, kernel_width = child.kernel_size
            std = math.sqrt(
                2.0 / (kernel_width * kernel_height * child.in_channels)
            )
            nn.init.normal_(child.weight, mean=0.0, std=std)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.BatchNorm2d):
            if child.affine:
                nn.init.uniform_(child.weight, 0.0, 1.0)
                nn.init.zeros_(child.bias)
            if child.track_running_stats:
                nn.init.zeros_(child.running_mean)
                nn.init.ones_(child.running_var)
                child.num_batches_tracked.zero_()
        elif isinstance(child, nn.Linear):
            bound = 1.0 / math.sqrt(child.in_features)
            nn.init.uniform_(child.weight, -bound, bound)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
