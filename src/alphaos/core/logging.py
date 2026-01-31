"""
AlphaOS Structured Logging Module

Provides production-grade logging using structlog with:
- JSON output for production
- Pretty console output for development
- Contextual logging with bound fields
- Performance metrics integration
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from structlog.typing import Processor


def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    service_name: str = "alphaos",
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format - 'json' for production, 'console' for development
        service_name: Service name to include in all log entries
    """
    # Shared processors for all outputs
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if log_format == "json":
        # Production JSON logging
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
        
        # Configure standard logging to use structlog
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, level.upper()),
        )
    else:
        # Development console logging with colors
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
        
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, level.upper()),
        )
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Log startup
    logger = get_logger(service_name)
    logger.info(
        "Logging initialized",
        level=level,
        format=log_format,
        service=service_name,
    )


class _TeeStream:
    """Write to multiple streams (e.g., console + file)."""

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        total = 0
        for stream in self._streams:
            try:
                if stream is None:
                    continue
                res = stream.write(data)
                if res is not None:
                    total = res
            except Exception:
                # Ignore errors (e.g., broken pipe, closed stream) to prevent crash
                pass
        return total

    def flush(self) -> None:
        for stream in self._streams:
            try:
                if stream is not None:
                    stream.flush()
            except Exception:
                pass


def enable_log_file(
    prefix: str,
    log_dir: Path | None = None,
    log_path: Path | str | None = None,
    tee_console: bool = True,
) -> Path:
    """
    Enable log file output.

    Args:
        prefix: Log file prefix (e.g., 'inference', 'training') - used only if log_path is None
        log_dir: Optional log directory (default: <repo>/logs) - used only if log_path is None
        log_path: Explicit log file path. If provided, prefix/log_dir/timestamp are ignored.
        tee_console: If True, tee output to both console and file (default).
                     If False, only write to file (stdout/stderr replaced entirely).

    Returns:
        Path to the log file.
    """
    if log_path is not None:
        # Use explicit path
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Legacy behavior: generate timestamped path
        root_dir = Path(__file__).resolve().parents[3]
        logs_dir = log_dir or (root_dir / "logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"{prefix}_{timestamp}.log"

    log_file = open(log_path, "a", encoding="utf-8")

    if tee_console:
        # Tee to both console and file
        sys.stdout = _TeeStream(sys.__stdout__, log_file)
        sys.stderr = _TeeStream(sys.__stderr__, log_file)
    else:
        # File only (no console output)
        sys.stdout = log_file  # type: ignore[assignment]
        sys.stderr = log_file  # type: ignore[assignment]

    return log_path


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically module name)
        
    Returns:
        Bound logger instance with context support
    """
    return structlog.get_logger(name)


class LogContext:
    """
    Context manager for adding temporary context to logs.
    
    Usage:
        with LogContext(request_id="123", user="alice"):
            logger.info("Processing request")
    """
    
    def __init__(self, **kwargs: Any) -> None:
        self.context = kwargs
        self._token: Any = None
        
    def __enter__(self) -> "LogContext":
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self
        
    def __exit__(self, *args: Any) -> None:
        structlog.contextvars.unbind_contextvars(*self.context.keys())


def log_performance(
    logger: structlog.stdlib.BoundLogger,
    operation: str,
    duration_ms: float,
    **extra: Any,
) -> None:
    """
    Log a performance metric.
    
    Args:
        logger: Logger instance
        operation: Name of the operation being measured
        duration_ms: Duration in milliseconds
        **extra: Additional context fields
    """
    logger.info(
        "performance_metric",
        operation=operation,
        duration_ms=round(duration_ms, 3),
        **extra,
    )
