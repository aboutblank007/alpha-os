"""
AlphaOS Exit Policy Module

Staged exit engines:
- v2.0: Basic staged exit (BE, Partial, Trailing)
- v2.1: Enhanced with bid/ask, cost guard, alignment modulation

Exit logic is owned by OrderManager (single source of truth for positions).
No ML models involved - purely rule-based.
"""

from alphaos.execution.exit.policy_v2 import (
    ExitAction,
    ExitDecision,
    PositionState,
    StagedExitConfig,
    StagedExitPolicyV2,
)
from alphaos.execution.exit.policy_v21 import (
    ExitAction as ExitActionV21,
    ExitDecision as ExitDecisionV21,
    ExitStage,
    PositionStateV21,
    ExitPolicyV21,
    TrendAlignment,
)

__all__ = [
    # v2.0
    "ExitAction",
    "ExitDecision",
    "PositionState",
    "StagedExitConfig",
    "StagedExitPolicyV2",
    # v2.1
    "ExitActionV21",
    "ExitDecisionV21",
    "ExitStage",
    "PositionStateV21",
    "ExitPolicyV21",
    "TrendAlignment",
]
