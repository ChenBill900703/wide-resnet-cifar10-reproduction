"""Frozen WRN-16-8 CIFAR-10 architecture.

Only model topology and initialization are implemented in Phase 1. This module
contains no dataset, optimizer, scheduler, checkpoint, or training behavior.
"""

from __future__ import annotations

from torch import Tensor, nn

from .initialization import initialize_wrn


class WideBasicBlock(nn.Module):
    """Pre-activation B(3,3) block with pinned Torch7 shortcut routing."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")

        self.bn1 = nn.BatchNorm2d(
            in_channels, eps=1e-5, momentum=0.1, affine=True
        )
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(
            out_channels, eps=1e-5, momentum=0.1, affine=True
        )
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        dimensions_change = stride != 1 or in_channels != out_channels
        self.projection = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            if dimensions_change
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        preactivated = self.relu1(self.bn1(x))
        residual = self.conv1(preactivated)
        residual = self.conv2(self.relu2(self.bn2(residual)))

        # Torch7 routes raw x through identities, but routes the shared first
        # BN→ReLU tensor through dimension-changing projections.
        shortcut = x if self.projection is None else self.projection(preactivated)
        return residual + shortcut


class WideResNet(nn.Module):
    """CIFAR Wide ResNet with the frozen no-dropout topology."""

    def __init__(
        self,
        *,
        depth: int = 16,
        widen_factor: int = 8,
        num_classes: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"depth must satisfy (depth - 4) % 6 == 0, got {depth}")
        if widen_factor <= 0:
            raise ValueError("widen_factor must be positive")
        if dropout != 0.0:
            raise ValueError("the frozen WRN-16-8 target requires dropout=0")

        self.depth = depth
        self.widen_factor = widen_factor
        self.blocks_per_group = (depth - 4) // 6
        self.stage_channels = (
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
        )

        stem_channels, group1_channels, group2_channels, group3_channels = (
            self.stage_channels
        )
        self.stem = nn.Conv2d(
            3, stem_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.group1 = self._make_group(
            stem_channels, group1_channels, stride=1
        )
        self.group2 = self._make_group(
            group1_channels, group2_channels, stride=2
        )
        self.group3 = self._make_group(
            group2_channels, group3_channels, stride=2
        )
        self.final_bn = nn.BatchNorm2d(
            group3_channels, eps=1e-5, momentum=0.1, affine=True
        )
        self.final_relu = nn.ReLU(inplace=False)
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(group3_channels, num_classes, bias=True)

        initialize_wrn(self)

    def _make_group(
        self, in_channels: int, out_channels: int, *, stride: int
    ) -> nn.Sequential:
        blocks = [WideBasicBlock(in_channels, out_channels, stride)]
        blocks.extend(
            WideBasicBlock(out_channels, out_channels, stride=1)
            for _ in range(1, self.blocks_per_group)
        )
        return nn.Sequential(*blocks)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        return self.final_relu(self.final_bn(x))

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.global_average_pool(x)
        x = x.flatten(1)
        return self.classifier(x)


def wrn16_8(*, num_classes: int = 10) -> WideResNet:
    """Construct the frozen WRN-16-8, B(3,3), dropout-0 target."""

    return WideResNet(
        depth=16, widen_factor=8, num_classes=num_classes, dropout=0.0
    )
