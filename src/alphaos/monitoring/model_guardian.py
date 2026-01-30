"""
AlphaOS Model Guardian (v4)

Implements the "Model Kill Switch" for production safety.

The Model Guardian monitors model outputs and internal states for anomalies
that could lead to catastrophic losses. When triggered, it:
1. Sets is_halted = True
2. Returns neutral signals
3. Writes a lock file for human review

Trigger conditions:
- NaN/Inf in predictions or hidden states
- Hidden state saturation (norm > threshold)
- Confidence collapse (sustained low confidence)
- Inference latency exceeded

Reference: AlphaOS v4 Decision Whitepaper (SSOT)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from alphaos.core.logging import get_logger
from alphaos.core.types import SignalType

logger = get_logger(__name__)


# ============================================================================
# Model Failure Types
# ============================================================================

class ModelFailureType(Enum):
    """Types of model failures detected by the Guardian."""
    NAN_INF = auto()              # NaN or Inf in outputs
    STATE_SATURATION = auto()     # Hidden state norm too high
    CONFIDENCE_COLLAPSE = auto()  # Sustained low confidence
    LATENCY_EXCEEDED = auto()     # Inference too slow
    MANUAL_HALT = auto()          # Manual intervention


@dataclass
class GuardianAlert:
    """Alert generated when Guardian detects an anomaly."""
    failure_type: ModelFailureType
    timestamp: float
    details: dict[str, Any]
    severity: str = "critical"  # "warning", "critical"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "failure_type": self.failure_type.name,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "details": self.details,
            "severity": self.severity,
        }


# ============================================================================
# Model Guardian
# ============================================================================

@dataclass
class ModelGuardian:
    """
    Model Guardian - Production safety kill switch (v4).
    
    Monitors model outputs and internal states for anomalies that could
    lead to catastrophic losses. When triggered, halts trading and
    requires human intervention to resume.
    
    Usage:
        guardian = ModelGuardian(config)
        
        # In inference loop:
        result = guardian.check_output(
            prediction=model_output,
            confidence=confidence,
            hidden_state=hidden_state,
            latency_ms=inference_time,
        )
        
        if result.should_halt:
            # Return neutral signal, stop trading
            pass
    """
    
    # Configuration
    enabled: bool = True
    nan_inf_check: bool = True
    state_saturation_threshold: float = 100.0  # Max norm for hidden state
    confidence_collapse_window: int = 10       # Consecutive low confidence ticks
    confidence_collapse_threshold: float = 0.3 # Minimum acceptable confidence
    latency_threshold_ms: float = 200.0        # Max inference latency
    lock_file_path: str = "logs/model_guardian_lock.json"
    
    # State
    is_halted: bool = field(default=False, init=False)
    halt_reason: str = field(default="", init=False)
    alerts: list[GuardianAlert] = field(default_factory=list, init=False)
    
    # Tracking for confidence collapse
    _low_confidence_count: int = field(default=0, init=False)
    _total_checks: int = field(default=0, init=False)
    _halt_timestamp: float = field(default=0.0, init=False)
    
    def __post_init__(self) -> None:
        """Initialize Guardian and check for existing lock file."""
        lock_path = Path(self.lock_file_path)
        if lock_path.exists():
            logger.warning(
                "Model Guardian lock file exists - model was previously halted",
                lock_file=str(lock_path),
            )
            self._load_lock_file(lock_path)
    
    def reset(self) -> None:
        """Reset Guardian state (does NOT clear halt - use unlock for that)."""
        self._low_confidence_count = 0
        self._total_checks = 0
        self.alerts.clear()
        logger.debug("Model Guardian state reset")
    
    def check_output(
        self,
        prediction: int | NDArray[np.float64],
        confidence: float,
        hidden_state: NDArray[np.float64] | None = None,
        latency_ms: float = 0.0,
    ) -> "GuardianCheckResult":
        """
        Check model output for anomalies.
        
        Args:
            prediction: Model prediction (signal type or probability array)
            confidence: Prediction confidence [0, 1]
            hidden_state: LNN hidden state vector (optional)
            latency_ms: Inference latency in milliseconds
            
        Returns:
            GuardianCheckResult with halt decision and safe output
        """
        self._total_checks += 1
        
        # If already halted, return neutral
        if self.is_halted:
            return GuardianCheckResult(
                should_halt=True,
                safe_prediction=SignalType.NEUTRAL.value,
                safe_confidence=0.0,
                alert=None,
                halt_reason=self.halt_reason,
            )
        
        if not self.enabled:
            return GuardianCheckResult(
                should_halt=False,
                safe_prediction=prediction if isinstance(prediction, int) else int(np.argmax(prediction)),
                safe_confidence=confidence,
                alert=None,
                halt_reason="",
            )
        
        # Check for NaN/Inf
        if self.nan_inf_check:
            alert = self._check_nan_inf(prediction, confidence, hidden_state)
            if alert:
                self._trigger_halt(alert)
                return GuardianCheckResult(
                    should_halt=True,
                    safe_prediction=SignalType.NEUTRAL.value,
                    safe_confidence=0.0,
                    alert=alert,
                    halt_reason=self.halt_reason,
                )
        
        # Check hidden state saturation
        if hidden_state is not None:
            alert = self._check_state_saturation(hidden_state)
            if alert:
                self._trigger_halt(alert)
                return GuardianCheckResult(
                    should_halt=True,
                    safe_prediction=SignalType.NEUTRAL.value,
                    safe_confidence=0.0,
                    alert=alert,
                    halt_reason=self.halt_reason,
                )
        
        # Check confidence collapse
        alert = self._check_confidence_collapse(confidence)
        if alert:
            self._trigger_halt(alert)
            return GuardianCheckResult(
                should_halt=True,
                safe_prediction=SignalType.NEUTRAL.value,
                safe_confidence=0.0,
                alert=alert,
                halt_reason=self.halt_reason,
            )
        
        # Check latency
        if latency_ms > 0:
            alert = self._check_latency(latency_ms)
            if alert:
                self._trigger_halt(alert)
                return GuardianCheckResult(
                    should_halt=True,
                    safe_prediction=SignalType.NEUTRAL.value,
                    safe_confidence=0.0,
                    alert=alert,
                    halt_reason=self.halt_reason,
                )
        
        # All checks passed
        safe_pred = prediction if isinstance(prediction, int) else int(np.argmax(prediction))
        return GuardianCheckResult(
            should_halt=False,
            safe_prediction=safe_pred,
            safe_confidence=confidence,
            alert=None,
            halt_reason="",
        )
    
    def _check_nan_inf(
        self,
        prediction: int | NDArray[np.float64],
        confidence: float,
        hidden_state: NDArray[np.float64] | None,
    ) -> GuardianAlert | None:
        """Check for NaN or Inf values."""
        issues = []
        
        # Check prediction
        if isinstance(prediction, np.ndarray):
            if np.any(np.isnan(prediction)) or np.any(np.isinf(prediction)):
                issues.append("prediction array contains NaN/Inf")
        
        # Check confidence
        if np.isnan(confidence) or np.isinf(confidence):
            issues.append(f"confidence is {confidence}")
        
        # Check hidden state
        if hidden_state is not None:
            if np.any(np.isnan(hidden_state)):
                issues.append(f"hidden state contains {np.sum(np.isnan(hidden_state))} NaN values")
            if np.any(np.isinf(hidden_state)):
                issues.append(f"hidden state contains {np.sum(np.isinf(hidden_state))} Inf values")
        
        if issues:
            return GuardianAlert(
                failure_type=ModelFailureType.NAN_INF,
                timestamp=time.time(),
                details={"issues": issues},
                severity="critical",
            )
        
        return None
    
    def _check_state_saturation(
        self,
        hidden_state: NDArray[np.float64],
    ) -> GuardianAlert | None:
        """Check if hidden state norm exceeds threshold."""
        state_norm = float(np.linalg.norm(hidden_state))
        
        if state_norm > self.state_saturation_threshold:
            return GuardianAlert(
                failure_type=ModelFailureType.STATE_SATURATION,
                timestamp=time.time(),
                details={
                    "state_norm": state_norm,
                    "threshold": self.state_saturation_threshold,
                    "state_max": float(np.max(np.abs(hidden_state))),
                    "state_mean": float(np.mean(hidden_state)),
                },
                severity="critical",
            )
        
        return None
    
    def _check_confidence_collapse(self, confidence: float) -> GuardianAlert | None:
        """Check for sustained low confidence."""
        if confidence < self.confidence_collapse_threshold:
            self._low_confidence_count += 1
        else:
            self._low_confidence_count = 0
        
        if self._low_confidence_count >= self.confidence_collapse_window:
            return GuardianAlert(
                failure_type=ModelFailureType.CONFIDENCE_COLLAPSE,
                timestamp=time.time(),
                details={
                    "consecutive_low_confidence": self._low_confidence_count,
                    "threshold": self.confidence_collapse_threshold,
                    "window": self.confidence_collapse_window,
                    "last_confidence": confidence,
                },
                severity="critical",
            )
        
        return None
    
    def _check_latency(self, latency_ms: float) -> GuardianAlert | None:
        """Check if inference latency exceeds threshold."""
        if latency_ms > self.latency_threshold_ms:
            return GuardianAlert(
                failure_type=ModelFailureType.LATENCY_EXCEEDED,
                timestamp=time.time(),
                details={
                    "latency_ms": latency_ms,
                    "threshold_ms": self.latency_threshold_ms,
                },
                severity="critical",
            )
        
        return None
    
    def _trigger_halt(self, alert: GuardianAlert) -> None:
        """Trigger model halt and write lock file."""
        self.is_halted = True
        self._halt_timestamp = time.time()
        self.halt_reason = f"{alert.failure_type.name}: {alert.details}"
        self.alerts.append(alert)
        
        logger.error(
            "MODEL GUARDIAN TRIGGERED - Trading halted",
            failure_type=alert.failure_type.name,
            details=alert.details,
            total_checks=self._total_checks,
        )
        
        # Write lock file
        self._write_lock_file(alert)
    
    def _write_lock_file(self, alert: GuardianAlert) -> None:
        """Write lock file for human review."""
        lock_path = Path(self.lock_file_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        
        lock_data = {
            "halted": True,
            "halt_timestamp": self._halt_timestamp,
            "halt_timestamp_iso": datetime.fromtimestamp(self._halt_timestamp).isoformat(),
            "halt_reason": self.halt_reason,
            "total_checks_before_halt": self._total_checks,
            "alert": alert.to_dict(),
            "all_alerts": [a.to_dict() for a in self.alerts],
            "requires_human_review": True,
        }
        
        with open(lock_path, "w") as f:
            json.dump(lock_data, f, indent=2)
        
        logger.warning(
            "Lock file written - requires human review to unlock",
            lock_file=str(lock_path),
        )
    
    def _load_lock_file(self, lock_path: Path) -> None:
        """Load existing lock file state."""
        try:
            with open(lock_path, "r") as f:
                lock_data = json.load(f)
            
            self.is_halted = lock_data.get("halted", True)
            self._halt_timestamp = lock_data.get("halt_timestamp", 0.0)
            self.halt_reason = lock_data.get("halt_reason", "Loaded from lock file")
            
            logger.info(
                "Loaded Guardian state from lock file",
                is_halted=self.is_halted,
                halt_reason=self.halt_reason,
            )
        except Exception as e:
            logger.error(f"Failed to load lock file: {e}")
            self.is_halted = True
            self.halt_reason = f"Failed to load lock file: {e}"
    
    def unlock(self, reason: str = "Manual unlock") -> bool:
        """
        Unlock the Guardian (requires human intervention).
        
        Args:
            reason: Reason for unlocking
            
        Returns:
            True if successfully unlocked
        """
        if not self.is_halted:
            logger.info("Guardian is not halted, nothing to unlock")
            return True
        
        # Remove lock file
        lock_path = Path(self.lock_file_path)
        if lock_path.exists():
            # Archive the lock file
            archive_path = lock_path.with_suffix(
                f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            lock_path.rename(archive_path)
            logger.info(f"Lock file archived to {archive_path}")
        
        # Reset state
        self.is_halted = False
        self._halt_timestamp = 0.0
        self.halt_reason = ""
        self._low_confidence_count = 0
        
        logger.info(
            "Model Guardian UNLOCKED",
            reason=reason,
            total_alerts=len(self.alerts),
        )
        
        return True
    
    def manual_halt(self, reason: str = "Manual halt") -> None:
        """Manually halt the model."""
        alert = GuardianAlert(
            failure_type=ModelFailureType.MANUAL_HALT,
            timestamp=time.time(),
            details={"reason": reason},
            severity="critical",
        )
        self._trigger_halt(alert)
    
    def get_stats(self) -> dict[str, Any]:
        """Get Guardian statistics."""
        return {
            "enabled": self.enabled,
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason,
            "total_checks": self._total_checks,
            "total_alerts": len(self.alerts),
            "low_confidence_count": self._low_confidence_count,
            "thresholds": {
                "state_saturation": self.state_saturation_threshold,
                "confidence_collapse_window": self.confidence_collapse_window,
                "confidence_collapse_threshold": self.confidence_collapse_threshold,
                "latency_ms": self.latency_threshold_ms,
            },
        }


# ============================================================================
# Guardian Check Result
# ============================================================================

@dataclass
class GuardianCheckResult:
    """Result of a Guardian check."""
    should_halt: bool
    safe_prediction: int
    safe_confidence: float
    alert: GuardianAlert | None
    halt_reason: str


# ============================================================================
# Factory Function
# ============================================================================

def create_model_guardian(
    enabled: bool = True,
    nan_inf_check: bool = True,
    state_saturation_threshold: float = 100.0,
    confidence_collapse_window: int = 10,
    confidence_collapse_threshold: float = 0.3,
    latency_threshold_ms: float = 200.0,
    lock_file_path: str = "logs/model_guardian_lock.json",
) -> ModelGuardian:
    """
    Create a Model Guardian with the specified configuration.
    
    Args:
        enabled: Whether Guardian is active
        nan_inf_check: Check for NaN/Inf values
        state_saturation_threshold: Max hidden state norm
        confidence_collapse_window: Consecutive low confidence ticks to trigger
        confidence_collapse_threshold: Minimum acceptable confidence
        latency_threshold_ms: Maximum inference latency
        lock_file_path: Path for lock file
        
    Returns:
        Configured ModelGuardian instance
    """
    return ModelGuardian(
        enabled=enabled,
        nan_inf_check=nan_inf_check,
        state_saturation_threshold=state_saturation_threshold,
        confidence_collapse_window=confidence_collapse_window,
        confidence_collapse_threshold=confidence_collapse_threshold,
        latency_threshold_ms=latency_threshold_ms,
        lock_file_path=lock_file_path,
    )
