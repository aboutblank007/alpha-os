"""
v4 Kelly Criterion Position Sizing (Single Source of Truth)

Kelly formula: f* = (p * b - q) / b

Where:
- p = probability of winning (calibrated confidence)
- q = 1 - p = probability of losing
- b = win/loss ratio (risk/reward)
- f* = optimal fraction of bankroll to bet

We use fractional Kelly (e.g., 25%) to reduce variance.

This module provides a unified implementation used by:
- v4 CLI serve command
- Legacy order_manager (if enabled)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Logger


@dataclass
class KellySizingConfig:
    """
    Configuration for Kelly position sizing.
    
    All parameters have sensible defaults for XAUUSD scalping.
    """
    # Kelly-specific parameters
    kelly_fraction: float = 0.25   # Fractional Kelly (25% = quarter Kelly)
    kelly_max_fraction: float = 0.5  # Maximum fraction of bankroll
    
    # Risk/Reward ratio (b in Kelly formula)
    # Default 3.0 means we win 3x for every 1x risked (1:3 R:R)
    risk_reward_ratio: float = 3.0
    
    # Lot size constraints
    min_lots: float = 0.01
    max_lots: float = 0.10
    
    # Sizing mode: "kelly", "fixed", "linear"
    mode: str = "kelly"


def calculate_kelly_lots(
    confidence: float,
    min_lots: float = 0.01,
    max_lots: float = 0.10,
    kelly_fraction: float = 0.25,
    kelly_max_fraction: float = 0.5,
    risk_reward_ratio: float = 3.0,
    logger: "Logger | None" = None,
) -> float:
    """
    Calculate position size using Kelly Criterion.
    
    This is the single source of truth for Kelly sizing in v4.
    
    Kelly formula: f* = (p * b - q) / b
    
    Args:
        confidence: Calibrated probability of signal being correct (0.0-1.0)
        min_lots: Minimum position size
        max_lots: Maximum position size
        kelly_fraction: Fractional Kelly multiplier (e.g., 0.25 for quarter Kelly)
        kelly_max_fraction: Maximum Kelly fraction before clamping
        risk_reward_ratio: Win/loss ratio (b in Kelly formula)
        logger: Optional logger for debug output
        
    Returns:
        Calculated lot size, clamped to [min_lots, max_lots]
        
    Example:
        >>> calculate_kelly_lots(confidence=0.65, min_lots=0.01, max_lots=0.10)
        0.05  # Approximately
    """
    # Use confidence as probability of winning (p)
    # Should be calibrated confidence from IsotonicCalibrator
    p = confidence
    q = 1 - p
    b = risk_reward_ratio
    
    # Kelly formula: f* = (p * b - q) / b
    # This is the optimal fraction of bankroll to bet
    kelly_full = (p * b - q) / b
    
    # Apply fractional Kelly
    kelly_bet = kelly_full * kelly_fraction
    
    # Clamp to valid range
    kelly_bet = max(0, min(kelly_bet, kelly_max_fraction))
    
    # If Kelly is negative, don't bet (edge is negative)
    if kelly_bet <= 0:
        if logger is not None:
            logger.debug(
                f"Kelly suggests no bet (negative edge): "
                f"conf={confidence:.3f}, kelly_full={kelly_full:.4f}, p={p:.3f}, b={b:.1f}"
            )
        return min_lots
    
    # Map kelly_bet (0 to kelly_max_fraction) to (min_lots to max_lots)
    if kelly_bet >= kelly_max_fraction:
        lots = max_lots
    else:
        ratio = kelly_bet / kelly_max_fraction
        lots = min_lots + ratio * (max_lots - min_lots)
    
    # Round to 2 decimals
    lots = round(lots, 2)
    
    # Ensure within bounds
    lots = max(min_lots, min(lots, max_lots))
    
    if logger is not None:
        logger.info(
            f"Kelly sizing: conf={confidence:.3f}, p={p:.3f}, b={b:.1f}, "
            f"kelly_full={kelly_full:.4f}, kelly_bet={kelly_bet:.4f}, lots={lots:.2f}"
        )
    
    return lots


def calculate_position_size(
    confidence: float,
    config: KellySizingConfig,
    logger: "Logger | None" = None,
) -> float:
    """
    Calculate position size based on sizing mode in config.
    
    Unified entry point for all position sizing modes:
    - "kelly": Kelly criterion
    - "fixed": Fixed lot size (max_lots)
    - "linear": Linear interpolation between min/max based on confidence
    
    Args:
        confidence: Signal confidence (0.0-1.0)
        config: Sizing configuration
        logger: Optional logger
        
    Returns:
        Calculated lot size
    """
    if config.mode == "fixed":
        return config.max_lots
    
    elif config.mode == "kelly":
        return calculate_kelly_lots(
            confidence=confidence,
            min_lots=config.min_lots,
            max_lots=config.max_lots,
            kelly_fraction=config.kelly_fraction,
            kelly_max_fraction=config.kelly_max_fraction,
            risk_reward_ratio=config.risk_reward_ratio,
            logger=logger,
        )
    
    else:  # "linear" or default
        # Linear interpolation: 0.5 confidence → min_lots, 0.8 confidence → max_lots
        conf_min = 0.50  # Threshold below which we use min_lots
        conf_max = 0.80  # Threshold at which we use max_lots
        
        if confidence <= conf_min:
            lots = config.min_lots
        elif confidence >= conf_max:
            lots = config.max_lots
        else:
            ratio = (confidence - conf_min) / (conf_max - conf_min)
            lots = config.min_lots + ratio * (config.max_lots - config.min_lots)
        
        lots = round(lots, 2)
        
        if logger is not None:
            logger.info(
                f"Linear sizing: conf={confidence:.3f}, lots={lots:.2f}, "
                f"min={config.min_lots}, max={config.max_lots}"
            )
        
        return lots
