"""
v4.0 微观结构特征模块

提供与 FeatureSchema 对齐的特征计算：
- 价格特征（对数收益率、Z-Score）
- 微观结构（OFI、Kyle Lambda、PDI）
- 波动率（微观波动率、已实现波动率）
- 热力学（温度、熵、相位）
- 订单流（VPIN）
- 趋势（SuperTrend 偏离）
- 降噪输出（Kalman、Wavelet）

ML 友好特征（v4.1）：
- FVG 特征（活跃状态）
- FVG 事件特征（事件型 + 因果跟随）
- 15m SuperTrend 特征
- 时间/Session 特征
- 交叉特征

参考：
- 降噪LNN特征提取与信号过滤.md Section 4
- 交易模型研究.md Section 3
"""

from alphaos.v4.features.pipeline import (
    FeatureConfig,
    FeaturePipelineV4,
    FeatureResult,
    ThermodynamicsConfig,
)
from alphaos.v4.features.vpin import (
    VPINCalculator,
    VPINConfig,
)

# ML 友好特征模块
from alphaos.v4.features.fvg_features import (
    FVGFeatureCalculator,
    FVGFeatureConfig,
    FVGFeaturesResult,
)
from alphaos.v4.features.fvg_event_features import (
    FVGEventCalculator,
    FVGEventConfig,
    FVGEventFeaturesResult,
    ATRRatioCalculator,
    STAlignmentCalculator,
)
from alphaos.v4.features.supertrend_features import (
    SuperTrendFeatureCalculator,
    SuperTrendFeatureConfig,
    SuperTrendFeaturesResult,
)
from alphaos.v4.features.time_features import (
    TimeFeatureCalculator,
    TimeFeatureConfig,
    TimeFeaturesResult,
)
from alphaos.v4.features.interaction_features import (
    InteractionFeatureCalculator,
    InteractionFeaturesResult,
)
from alphaos.v4.features.ml_pipeline import (
    MLFeaturePipeline,
    MLFeatureConfig,
    MLFeatureResult,
)

__all__ = [
    # 原有模块
    "FeatureConfig",
    "FeaturePipelineV4",
    "FeatureResult",
    "ThermodynamicsConfig",
    "VPINCalculator",
    "VPINConfig",
    # ML 友好特征（活跃状态）
    "FVGFeatureCalculator",
    "FVGFeatureConfig",
    "FVGFeaturesResult",
    # FVG 事件特征（事件型 + 因果）
    "FVGEventCalculator",
    "FVGEventConfig",
    "FVGEventFeaturesResult",
    "ATRRatioCalculator",
    "STAlignmentCalculator",
    # SuperTrend
    "SuperTrendFeatureCalculator",
    "SuperTrendFeatureConfig",
    "SuperTrendFeaturesResult",
    # 时间
    "TimeFeatureCalculator",
    "TimeFeatureConfig",
    "TimeFeaturesResult",
    # 交叉
    "InteractionFeatureCalculator",
    "InteractionFeaturesResult",
    # ML 管道
    "MLFeaturePipeline",
    "MLFeatureConfig",
    "MLFeatureResult",
]
