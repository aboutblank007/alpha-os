"""
v4.0 Primary 信号引擎

封装 PivotSuperTrend + FVG 作为 Primary 信号生成器：
- 高召回率：宁可多信号，由 Meta Model 过滤
- 提供趋势方向和入场时机
- 输出标准化的 PrimarySignal 格式

参考：交易模型研究.md Section 4
"""

from alphaos.v4.primary.engine import (
    PrimaryEngineConfig,
    PrimaryEngineV4,
    PrimarySignalV4,
)

__all__ = [
    "PrimaryEngineConfig",
    "PrimaryEngineV4",
    "PrimarySignalV4",
]
