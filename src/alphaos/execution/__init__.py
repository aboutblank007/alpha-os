"""
AlphaOS Execution Layer (v4)

Provides execution primitives used by v4:
- ZeroMQ communication with MT5
- Exit policies (v2.0, v2.1)
"""

from alphaos.execution.zmq_client import ZeroMQClient

__all__ = ["ZeroMQClient"]
