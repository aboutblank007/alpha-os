"""
v4.0 信息驱动采样层

提供统一的采样接口，支持：
- Volume Bars（默认）
- Tick Imbalance Bars
- 可配置的 Volume 来源（real, tick_count, synthetic）
- 时间聚合（1m, 15m 等）

参考：降噪LNN特征提取与信号过滤.md Section 2
"""

from alphaos.v4.sampling.sampler import (
    SamplingConfig,
    SamplingMode,
    VolumeSource,
    UnifiedSampler,
    EventBarStream,
)
from alphaos.v4.sampling.time_aggregator import (
    TimeBarAggregator,
    TimeBarAggregatorConfig,
    TimeBar,
    create_multi_timeframe_bars,
    align_features_to_lower_timeframe,
)

__all__ = [
    "SamplingConfig",
    "SamplingMode",
    "VolumeSource",
    "UnifiedSampler",
    "EventBarStream",
    # 时间聚合
    "TimeBarAggregator",
    "TimeBarAggregatorConfig",
    "TimeBar",
    "create_multi_timeframe_bars",
    "align_features_to_lower_timeframe",
]
