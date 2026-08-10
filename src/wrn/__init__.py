"""Frozen WRN-16-8 architecture exports."""

from .initialization import initialize_wrn
from .model import WideBasicBlock, WideResNet, wrn16_8

__all__ = ["WideBasicBlock", "WideResNet", "initialize_wrn", "wrn16_8"]
