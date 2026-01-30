"""
AlphaOS Health Checker

Provides health monitoring for production:
- Liveness probes
- Readiness probes
- Component status
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

from alphaos.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Health Status
# ============================================================================

class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    message: str = ""
    last_check: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Health Checker
# ============================================================================

HealthCheck = Callable[[], Awaitable[ComponentHealth]]

@dataclass
class HealthConfig:
    """健康检查配置"""
    check_interval_sec: float = 30.0
    tick_staleness_threshold_sec: int = 60
    warmup_ticks_estimate: int = 200
    critical_components: list[str] = field(default_factory=lambda: ["zmq", "model"])


@dataclass
class HealthChecker:
    """
    System health checker for AlphaOS.
    
    Monitors:
    - ZeroMQ connection
    - Model state
    - Risk manager
    - Data flow
    """
    
    checks: dict[str, HealthCheck] = field(default_factory=dict)
    config: HealthConfig = field(default_factory=HealthConfig)
    _last_results: dict[str, ComponentHealth] = field(default_factory=dict, init=False)
    _check_interval: float = field(default=30.0, init=False)
    _running: bool = field(default=False, init=False)
    
    def register_check(self, name: str, check: HealthCheck) -> None:
        """
        Register a health check.
        
        Args:
            name: Component name
            check: Async function returning ComponentHealth
        """
        self.checks[name] = check
        logger.debug("Registered health check", component=name)
    
    async def check_all(self) -> dict[str, ComponentHealth]:
        """
        Run all health checks.
        
        Returns:
            Dictionary of component name -> health status
        """
        results = {}
        
        for name, check in self.checks.items():
            try:
                health = await check()
                health.last_check = time.time()
                results[name] = health
            except Exception as e:
                results[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                    last_check=time.time(),
                )
        
        self._last_results = results
        return results
    
    async def is_live(self) -> bool:
        """
        Liveness probe: Is the system running?
        
        Returns True if the main process is alive.
        """
        return True  # If we can execute this, we're alive
    
    async def is_ready(self) -> bool:
        """
        Readiness probe: Is the system ready to trade?
        
        Returns True if all critical components are healthy.
        """
        results = await self.check_all()
        
        critical_components = self.config.critical_components
        
        for component in critical_components:
            if component in results:
                if results[component].status == HealthStatus.UNHEALTHY:
                    return False
        
        return True
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self._last_results:
            return HealthStatus.UNKNOWN
        
        statuses = [h.status for h in self._last_results.values()]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED
    
    def get_status_report(self) -> dict[str, Any]:
        """Get full status report."""
        return {
            "overall": self.get_overall_status().value,
            "components": {
                name: {
                    "status": health.status.value,
                    "message": health.message,
                    "last_check": health.last_check,
                    "details": health.details,
                }
                for name, health in self._last_results.items()
            },
            "timestamp": time.time(),
        }
    
    async def start_background_checks(self) -> None:
        """Start periodic health checks in background."""
        self._running = True
        self._check_interval = self.config.check_interval_sec
        
        while self._running:
            await self.check_all()
            await asyncio.sleep(self._check_interval)
    
    def stop(self) -> None:
        """Stop background health checks."""
        self._running = False


# ============================================================================
# Standard Health Checks
# ============================================================================

def create_zmq_health_check(zmq_client, tick_staleness_threshold_sec: int) -> HealthCheck:
    """创建 ZeroMQ 连接健康检查"""
    async def check() -> ComponentHealth:
        if not zmq_client.is_connected:
            return ComponentHealth(
                name="zmq",
                status=HealthStatus.UNHEALTHY,
                message="Not connected to MT5",
            )
        
        stats = zmq_client.get_stats()
        last_tick_age = stats.get("last_tick_age_s")
        
        if last_tick_age is not None and last_tick_age > tick_staleness_threshold_sec:
            return ComponentHealth(
                name="zmq",
                status=HealthStatus.DEGRADED,
                message=f"No ticks for {last_tick_age:.0f}s",
                details=stats,
            )
        
        return ComponentHealth(
            name="zmq",
            status=HealthStatus.HEALTHY,
            message="Connected",
            details=stats,
        )
    
    return check


def create_model_health_check(model, warmup_ticks_estimate: int) -> HealthCheck:
    """创建模型状态健康检查"""
    async def check() -> ComponentHealth:
        if not model.is_warmed_up:
            warmup_progress = model.tick_count / warmup_ticks_estimate
            return ComponentHealth(
                name="model",
                status=HealthStatus.DEGRADED,
                message=f"Warming up ({warmup_progress:.0%})",
                details={"tick_count": model.tick_count},
            )
        
        return ComponentHealth(
            name="model",
            status=HealthStatus.HEALTHY,
            message="Ready",
            details={"tick_count": model.tick_count},
        )
    
    return check


def create_risk_health_check(risk_manager) -> HealthCheck:
    """Create health check for risk manager."""
    async def check() -> ComponentHealth:
        stats = risk_manager.get_stats()
        
        if risk_manager.is_halted:
            return ComponentHealth(
                name="risk",
                status=HealthStatus.DEGRADED,
                message=f"Trading halted: {risk_manager.halt_reason}",
                details=stats,
            )
        
        return ComponentHealth(
            name="risk",
            status=HealthStatus.HEALTHY,
            message="Active",
            details=stats,
        )
    
    return check
