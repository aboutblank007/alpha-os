"""
AlphaOS Primary Signal Engines (v4.0)

Primary signal generation for Meta-Labeling architecture:
- PivotSuperTrend: Pivot Point + ATR-based SuperTrend trend filter
- FVGSignal: Fair Value Gap detection and entry timing

The Primary engine provides high-recall directional signals.
The Meta-model (LNN+XGBoost) filters for precision.
"""

from alphaos.execution.primary.pivot_supertrend import (
    PivotSuperTrend,
    SuperTrendState,
    TrendDirection as SuperTrendDirection,
)
from alphaos.execution.primary.fvg_signal import (
    FVGDetector,
    FVGSignal,
    FVGType,
    PrimarySignalGenerator,
    PrimarySignal,
)

__all__ = [
    # Pivot SuperTrend
    "PivotSuperTrend",
    "SuperTrendState",
    "SuperTrendDirection",
    # FVG Signal
    "FVGDetector",
    "FVGSignal",
    "FVGType",
    "PrimarySignalGenerator",
    "PrimarySignal",
]
