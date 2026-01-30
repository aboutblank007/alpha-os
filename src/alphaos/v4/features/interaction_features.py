"""
交叉特征计算模块

计算不同特征组之间的交互特征：
- 趋势 × FVG 对齐
- 趋势强度 × FVG 大小
- 趋势持续 × 波动率

交叉特征让模型更容易学习"顺趋势 FVG 更值钱"等复杂模式。

特征列表：
- trend_fvg_alignment: st_trend * (bullish_fvg - bearish_fvg)
- trend_strength_x_fvg: |st_distance| * fvg_size_atr
- trend_duration_x_vol: st_bars_since_flip * micro_volatility
- trend_fvg_distance: st_distance * price_to_fvg_mid
- session_trend_interaction: session * st_trend
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from alphaos.core.logging import get_logger
from alphaos.v4.features.fvg_features import FVGFeaturesResult
from alphaos.v4.features.supertrend_features import SuperTrendFeaturesResult
from alphaos.v4.features.time_features import TimeFeaturesResult

logger = get_logger(__name__)


@dataclass
class InteractionFeaturesResult:
    """
    交叉特征计算结果
    
    Attributes:
        trend_fvg_alignment: 趋势与 FVG 对齐度
        trend_strength_x_fvg: 趋势强度 × FVG 大小
        trend_duration_x_vol: 趋势持续 × 波动率
        trend_fvg_distance: 趋势距离 × FVG 距离
        session_trend_interaction: Session × 趋势
    """
    trend_fvg_alignment: float = 0.0
    trend_strength_x_fvg: float = 0.0
    trend_duration_x_vol: float = 0.0
    trend_fvg_distance: float = 0.0
    session_trend_interaction: float = 0.0
    
    def to_array(self) -> NDArray[np.float32]:
        """转换为 numpy 数组"""
        return np.array([
            self.trend_fvg_alignment,
            self.trend_strength_x_fvg,
            self.trend_duration_x_vol,
            self.trend_fvg_distance,
            self.session_trend_interaction,
        ], dtype=np.float32)
    
    @staticmethod
    def feature_names() -> list[str]:
        """特征名称列表"""
        return [
            "trend_fvg_alignment",
            "trend_strength_x_fvg",
            "trend_duration_x_vol",
            "trend_fvg_distance",
            "session_trend_interaction",
        ]
    
    @staticmethod
    def n_features() -> int:
        """特征数量"""
        return 5


def compute_interaction_features(
    st_feat: SuperTrendFeaturesResult,
    fvg_feat: FVGFeaturesResult,
    time_feat: TimeFeaturesResult | None = None,
    volatility: float = 0.0,
) -> InteractionFeaturesResult:
    """
    计算单个样本的交叉特征
    
    Args:
        st_feat: SuperTrend 特征
        fvg_feat: FVG 特征
        time_feat: 时间特征（可选）
        volatility: 波动率（可选）
        
    Returns:
        交叉特征结果
    """
    result = InteractionFeaturesResult()
    
    # 趋势 × FVG 对齐
    # +1 = 多头趋势 + Bullish FVG（顺趋势）
    # -1 = 空头趋势 + Bearish FVG（顺趋势）
    # 0 或负值 = 逆趋势
    fvg_direction = fvg_feat.bullish_fvg - fvg_feat.bearish_fvg
    result.trend_fvg_alignment = st_feat.st_trend * fvg_direction
    
    # 趋势强度 × FVG 大小
    result.trend_strength_x_fvg = abs(st_feat.st_distance) * fvg_feat.fvg_size_atr
    
    # 趋势持续 × 波动率
    result.trend_duration_x_vol = st_feat.st_bars_since_flip * volatility
    
    # 趋势距离 × FVG 距离
    result.trend_fvg_distance = st_feat.st_distance * fvg_feat.price_to_fvg_mid
    
    # Session × 趋势
    if time_feat is not None:
        # 编码：session 0-4, trend -1/0/1
        result.session_trend_interaction = time_feat.session * st_feat.st_trend
    
    return result


def compute_interaction_features_batch(
    st_features: NDArray[np.float32],
    fvg_features: NDArray[np.float32],
    time_features: NDArray[np.float32] | None = None,
    volatility: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """
    批量计算交叉特征
    
    Args:
        st_features: SuperTrend 特征矩阵 (n, 5)
        fvg_features: FVG 特征矩阵 (n, 8)
        time_features: 时间特征矩阵 (n, 9)，可选
        volatility: 波动率数组 (n,)，可选
        
    Returns:
        交叉特征矩阵 (n, 5)
    """
    n_samples = st_features.shape[0]
    result = np.zeros((n_samples, InteractionFeaturesResult.n_features()), dtype=np.float32)
    
    # SuperTrend 特征索引
    ST_TREND_IDX = 0
    ST_DISTANCE_IDX = 1
    ST_BARS_SINCE_FLIP_IDX = 2
    
    # FVG 特征索引
    BULLISH_FVG_IDX = 0
    BEARISH_FVG_IDX = 1
    FVG_SIZE_ATR_IDX = 2
    PRICE_TO_FVG_MID_IDX = 4
    
    # Time 特征索引
    SESSION_IDX = 0
    
    # 趋势 × FVG 对齐
    fvg_direction = fvg_features[:, BULLISH_FVG_IDX] - fvg_features[:, BEARISH_FVG_IDX]
    result[:, 0] = st_features[:, ST_TREND_IDX] * fvg_direction
    
    # 趋势强度 × FVG 大小
    result[:, 1] = np.abs(st_features[:, ST_DISTANCE_IDX]) * fvg_features[:, FVG_SIZE_ATR_IDX]
    
    # 趋势持续 × 波动率
    if volatility is not None:
        result[:, 2] = st_features[:, ST_BARS_SINCE_FLIP_IDX] * volatility
    
    # 趋势距离 × FVG 距离
    result[:, 3] = st_features[:, ST_DISTANCE_IDX] * fvg_features[:, PRICE_TO_FVG_MID_IDX]
    
    # Session × 趋势
    if time_features is not None:
        result[:, 4] = time_features[:, SESSION_IDX] * st_features[:, ST_TREND_IDX]
    
    logger.info(
        "Interaction features computed (batch)",
        n_samples=n_samples,
    )
    
    return result


@dataclass
class InteractionFeatureCalculator:
    """
    交叉特征计算器
    
    批量计算交叉特征的便捷接口。
    
    使用方式：
    ```python
    calc = InteractionFeatureCalculator()
    
    features = calc.compute_batch(
        st_features=st_feat_array,
        fvg_features=fvg_feat_array,
        time_features=time_feat_array,
        volatility=vol_array,
    )
    ```
    """
    
    def compute_batch(
        self,
        st_features: NDArray[np.float32],
        fvg_features: NDArray[np.float32],
        time_features: NDArray[np.float32] | None = None,
        volatility: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """批量计算交叉特征"""
        return compute_interaction_features_batch(
            st_features=st_features,
            fvg_features=fvg_features,
            time_features=time_features,
            volatility=volatility,
        )
    
    @staticmethod
    def feature_names() -> list[str]:
        """特征名称列表"""
        return InteractionFeaturesResult.feature_names()
    
    @staticmethod
    def n_features() -> int:
        """特征数量"""
        return InteractionFeaturesResult.n_features()
