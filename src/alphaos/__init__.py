"""
AlphaOS v4 - Microstructure LNN Trading System

Core components:
- Volume Bars + Event-driven sampling
- CfC encoder + XGBoost meta head
- Meta-Labeling (Primary direction + Meta confidence)
- Model Guardian for production safety

Copyright (c) 2024-2026 AlphaOS Team. All rights reserved.
"""

__version__ = "4.0.0"
__author__ = "AlphaOS Team"

from alphaos.core.logging import setup_logging, get_logger

__all__ = [
    "__version__",
    "setup_logging",
    "get_logger",
]
