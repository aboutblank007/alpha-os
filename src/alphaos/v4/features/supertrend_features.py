"""
SuperTrend 特征计算模块

将 SuperTrend 从信号过滤器转换为 ML 友好的特征：
- 趋势方向：-1/0/1
- 趋势强度：(Close - ST_line) / ATR（连续值）
- 趋势持续：翻转后 bar 数
- 趋势斜率：ST_line 变化率

支持多时间框架（1m + 15m），输出对齐到低时间框架。

特征列表：
- st_trend: 趋势方向 (-1/0/1)
- st_distance: (Close - ST_line) / ATR
- st_bars_since_flip: 趋势翻转后 bar 数
- st_slope: ST_line 斜率 / ATR
- st_bandwidth: ATR 带宽
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from alphaos.core.logging import get_logger
from alphaos.v4.types import Bar
from alphaos.data.event_bars.tick_imbalance import EventBar
from alphaos.execution.primary.pivot_supertrend import (
    PivotSuperTrend,
    TrendDirection,
    SuperTrendState,
)
from alphaos.v4.sampling.time_aggregator import TimeBar

logger = get_logger(__name__)

EPSILON = 1e-10


@dataclass
class SuperTrendFeaturesResult:
    """
    SuperTrend 特征计算结果
    
    Attributes:
        st_trend: 趋势方向 (-1/0/1)
        st_distance: (Close - ST_line) / ATR
        st_bars_since_flip: 趋势翻转后 bar 数
        st_slope: ST_line 变化率 / ATR
        st_bandwidth: ATR 带宽 (归一化)
    """
    st_trend: int = 0
    st_distance: float = 0.0
    st_bars_since_flip: int = 0
    st_slope: float = 0.0
    st_bandwidth: float = 0.0
    
    def to_array(self) -> NDArray[np.float32]:
        """转换为 numpy 数组"""
        return np.array([
            self.st_trend,
            self.st_distance,
            self.st_bars_since_flip,
            self.st_slope,
            self.st_bandwidth,
        ], dtype=np.float32)
    
    @staticmethod
    def feature_names(suffix: str = "") -> list[str]:
        """特征名称列表"""
        s = f"_{suffix}" if suffix else ""
        return [
            f"st_trend{s}",
            f"st_distance{s}",
            f"st_bars_since_flip{s}",
            f"st_slope{s}",
            f"st_bandwidth{s}",
        ]
    
    @staticmethod
    def n_features() -> int:
        """特征数量"""
        return 5


@dataclass
class SuperTrendFeatureConfig:
    """
    SuperTrend 特征计算配置
    
    Args:
        pivot_lookback: Pivot 确认 lookback
        atr_period: ATR 计算周期
        atr_factor: ATR 乘数
        slope_window: 斜率计算窗口
    """
    pivot_lookback: int = 2
    atr_period: int = 10
    atr_factor: float = 3.0
    slope_window: int = 5


@dataclass
class SuperTrendFeatureCalculator:
    """
    SuperTrend 特征计算器
    
    计算 ML 友好的 SuperTrend 特征。
    
    关键设计：
    - 所有距离用 ATR 归一化
    - 输出连续值（st_distance）而非仅趋势方向
    - 计算趋势斜率作为动量指标
    
    使用方式：
    ```python
    calc = SuperTrendFeatureCalculator(config)
    
    # 批量计算
    features = calc.compute_batch(bars)  # (n_bars, 5)
    
    # 流式计算
    for bar in stream:
        feat = calc.update(bar)
    ```
    """
    config: SuperTrendFeatureConfig = field(default_factory=SuperTrendFeatureConfig)
    
    # 内部状态
    _supertrend: PivotSuperTrend | None = field(default=None, init=False)
    _prev_st_line: float = field(default=0.0, init=False)
    _st_line_history: list[float] = field(default_factory=list, init=False)
    _bar_count: int = field(default=0, init=False)
    
    def __post_init__(self) -> None:
        """初始化 SuperTrend"""
        self._supertrend = PivotSuperTrend(
            pivot_lookback=self.config.pivot_lookback,
            atr_period=self.config.atr_period,
            atr_factor=self.config.atr_factor,
        )
        logger.info(
            "SuperTrendFeatureCalculator initialized",
            config=self.config.__dict__,
        )
    
    def reset(self) -> None:
        """重置状态"""
        self._supertrend = PivotSuperTrend(
            pivot_lookback=self.config.pivot_lookback,
            atr_period=self.config.atr_period,
            atr_factor=self.config.atr_factor,
        )
        self._prev_st_line = 0.0
        self._st_line_history.clear()
        self._bar_count = 0
    
    def compute_batch(
        self,
        bars: Sequence[EventBar] | Sequence[Bar] | Sequence[TimeBar],
    ) -> NDArray[np.float32]:
        """
        批量计算 SuperTrend 特征
        
        Args:
            bars: Bar 序列
            
        Returns:
            特征矩阵 (n_bars, 5)
        """
        self.reset()
        
        n_bars = len(bars)
        features = np.zeros((n_bars, SuperTrendFeaturesResult.n_features()), dtype=np.float32)
        
        for i, bar in enumerate(bars):
            # 转换为 Bar 格式
            converted_bar = self._convert_bar(bar)
            result = self._process_bar(converted_bar)
            features[i] = result.to_array()
        
        logger.info(
            "SuperTrend features computed (batch)",
            n_bars=n_bars,
        )
        
        return features
    
    def update(self, bar: EventBar | Bar | TimeBar) -> SuperTrendFeaturesResult:
        """
        流式更新（单 bar）
        
        Args:
            bar: 当前 bar
            
        Returns:
            SuperTrend 特征结果
        """
        converted_bar = self._convert_bar(bar)
        return self._process_bar(converted_bar)
    
    def _convert_bar(self, bar: EventBar | Bar | TimeBar) -> Bar:
        """将各种 bar 类型转换为 Bar"""
        if isinstance(bar, Bar):
            return bar
        elif isinstance(bar, TimeBar):
            return bar.to_bar()
        elif isinstance(bar, EventBar):
            return Bar(
                time=bar.time,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                tick_volume=bar.tick_count,
            )
        else:
            raise ValueError(f"Unknown bar type: {type(bar)}")
    
    def _process_bar(self, bar: Bar) -> SuperTrendFeaturesResult:
        """处理单个 bar"""
        self._bar_count += 1
        
        # 更新 SuperTrend
        state = self._supertrend.update(bar)
        
        # 记录 ST line 历史（用于计算斜率）
        self._st_line_history.append(state.supertrend_line)
        if len(self._st_line_history) > self.config.slope_window + 1:
            self._st_line_history = self._st_line_history[-self.config.slope_window - 1:]
        
        # 计算特征
        result = self._compute_features(bar, state)
        
        # 更新状态
        self._prev_st_line = state.supertrend_line
        
        return result
    
    def _compute_features(self, bar: Bar, state: SuperTrendState) -> SuperTrendFeaturesResult:
        """计算特征"""
        result = SuperTrendFeaturesResult()
        
        # 趋势方向
        if state.direction == TrendDirection.LONG:
            result.st_trend = 1
        elif state.direction == TrendDirection.SHORT:
            result.st_trend = -1
        else:
            result.st_trend = 0
        
        # 趋势距离（ATR 归一化）
        atr = max(state.atr, EPSILON)
        if state.supertrend_line > 0:
            result.st_distance = (bar.close - state.supertrend_line) / atr
        
        # 趋势持续时间
        result.st_bars_since_flip = state.trend_duration
        
        # ST line 斜率
        if len(self._st_line_history) >= 2:
            slope = self._st_line_history[-1] - self._st_line_history[-2]
            result.st_slope = slope / atr
        
        # 带宽
        result.st_bandwidth = state.bandwidth
        
        return result
    
    @property
    def current_direction(self) -> int:
        """当前趋势方向 (1/-1/0)"""
        if self._supertrend is None:
            return 0
        direction = self._supertrend.current_direction
        if direction == TrendDirection.LONG:
            return 1
        elif direction == TrendDirection.SHORT:
            return -1
        return 0
    
    @property
    def current_state(self) -> SuperTrendState | None:
        """当前 SuperTrend 状态"""
        if self._supertrend is None:
            return None
        return self._supertrend.current_state
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        state = self.current_state
        return {
            "bar_count": self._bar_count,
            "direction": self.current_direction,
            "st_line": state.supertrend_line if state else 0,
            "trend_duration": state.trend_duration if state else 0,
            "atr": state.atr if state else 0,
        }


def compute_aligned_supertrend_features(
    event_bars: Sequence[EventBar],
    time_bars: Sequence[TimeBar],
    alignment_indices: NDArray[np.int64],
    config: SuperTrendFeatureConfig | None = None,
) -> NDArray[np.float32]:
    """
    计算对齐到低时间框架的 SuperTrend 特征
    
    Args:
        event_bars: 低时间框架 bars (用于获取数量)
        time_bars: 高时间框架 bars (计算 SuperTrend)
        alignment_indices: 对齐索引（来自 time_aggregator）
        config: SuperTrend 特征配置
        
    Returns:
        特征矩阵 (len(event_bars), 5)
    """
    if config is None:
        config = SuperTrendFeatureConfig()
    
    # 在高时间框架上计算 SuperTrend 特征
    calc = SuperTrendFeatureCalculator(config=config)
    htf_features = calc.compute_batch(time_bars)
    
    # 对齐到低时间框架
    n_event_bars = len(event_bars)
    aligned_features = np.zeros((n_event_bars, SuperTrendFeaturesResult.n_features()), dtype=np.float32)
    
    for i, htf_idx in enumerate(alignment_indices):
        if 0 <= htf_idx < len(htf_features):
            aligned_features[i] = htf_features[htf_idx]
    
    logger.info(
        "Aligned SuperTrend features computed",
        n_event_bars=n_event_bars,
        n_time_bars=len(time_bars),
    )
    
    return aligned_features
