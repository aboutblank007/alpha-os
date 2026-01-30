"""
v4 Position Sizing Module

Provides unified position sizing algorithms:
- Kelly Criterion
- Fixed lot sizing
- Linear interpolation

This is the single source of truth for position sizing in v4.
"""

from alphaos.v4.sizing.kelly import (
    calculate_kelly_lots,
    KellySizingConfig,
)

__all__ = [
    "calculate_kelly_lots",
    "KellySizingConfig",
]
