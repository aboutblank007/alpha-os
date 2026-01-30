"""
AlphaOS Monitoring Module (v4)

Provides production monitoring:
- Risk management
- Prometheus metrics
- Health checks
- Model Guardian (production safety kill switch)
"""

from alphaos.monitoring.risk_manager import RiskManager
from alphaos.monitoring.metrics import MetricsCollector, setup_metrics_server
from alphaos.monitoring.health import HealthChecker
from alphaos.monitoring.model_guardian import (
    ModelGuardian,
    ModelFailureType,
    GuardianAlert,
    GuardianCheckResult,
    create_model_guardian,
)

__all__ = [
    "RiskManager",
    "MetricsCollector",
    "setup_metrics_server",
    "HealthChecker",
    "ModelGuardian",
    "ModelFailureType",
    "GuardianAlert",
    "GuardianCheckResult",
    "create_model_guardian",
]
