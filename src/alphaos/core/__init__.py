"""
AlphaOS Core Module

Contains fundamental utilities used across the system:
- Configuration management
- Logging infrastructure
- Common types and constants
"""

from alphaos.core.logging import setup_logging, get_logger
from alphaos.core.config import load_config, AlphaOSConfig

__all__ = [
    "setup_logging",
    "get_logger",
    "load_config",
    "AlphaOSConfig",
]
