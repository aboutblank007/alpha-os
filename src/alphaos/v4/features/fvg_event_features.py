"""
FVG 事件特征计算模块 (Event-based, Causal)

与原 fvg_features.py 的区别：
- fvg_event: 仅在"新 FVG 出现"那根 bar 取值 +1/-1/0
- fvg_impulse_atr: 新 FVG 的 gap size / ATR_1m（事件发生时的冲击强度）
- fvg_location_15m: (close_1m - mid_15m_range) / ATR_15m（FVG 位置）
- fvg_follow_up/dn_3: 因果跟随（自最近 FVG 事件以来的已实现反应）

关键设计：
- 所有特征严格因果（不使用未来数据）
- FVG_event 是脉冲信号（只在事件那根 bar 非零）
- follow-through 是滚动更新的累积反应

用于 LNN(CfC) 学习事件发生后的时间演化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from alphaos.core.logging import get_logger
from alphaos.v4.types import Bar
from alphaos.data.event_bars.tick_imbalance import EventBar
from alphaos.v4.sampling.time_aggregator import TimeBar

logger = get_logger(__name__)

EPSILON = 1e-10


@dataclass
class FVGEventState:
    """
    FVG 事件状态（用于跟踪最近一次事件）
    
    Attributes:
        event_bar_idx: 事件发生的 bar 索引
        event_type: +1=Bullish, -1=Bearish
        event_close: 事件时的收盘价
        impulse_atr: 事件时的冲击强度 (gap / ATR)
        max_up_since: 事件后的最大上涨
        max_dn_since: 事件后的最大下跌
    """
    event_bar_idx: int = -1
    event_type: int = 0  # +1=Bullish, -1=Bearish, 0=None
    event_close: float = 0.0
    impulse_atr: float = 0.0
    max_up_since: float = 0.0
    max_dn_since: float = 0.0
    
    def reset(self) -> None:
        """重置状态"""
        self.event_bar_idx = -1
        self.event_type = 0
        self.event_close = 0.0
        self.impulse_atr = 0.0
        self.max_up_since = 0.0
        self.max_dn_since = 0.0


@dataclass
class FVGEventFeaturesResult:
    """
    FVG 事件特征计算结果
    
    Attributes:
        fvg_event: +1=新 Bullish, -1=新 Bearish, 0=无事件
        fvg_impulse_atr: 冲击强度 (仅在事件时非零)
        fvg_location_15m: FVG 位置 (close - mid_15m) / ATR_15m
        fvg_follow_up_3: 事件后最大上涨 / ATR (因果)
        fvg_follow_dn_3: 事件后最大下跌 / ATR (因果)
        fvg_follow_bars: 距离最近事件的 bar 数
        fvg_follow_net: 事件后净收益 / ATR
    """
    fvg_event: int = 0
    fvg_impulse_atr: float = 0.0
    fvg_location_15m: float = 0.0
    fvg_follow_up_3: float = 0.0
    fvg_follow_dn_3: float = 0.0
    fvg_follow_bars: int = 0
    fvg_follow_net: float = 0.0
    
    def to_array(self) -> NDArray[np.float32]:
        """转换为 numpy 数组"""
        return np.array([
            self.fvg_event,
            self.fvg_impulse_atr,
            self.fvg_location_15m,
            self.fvg_follow_up_3,
            self.fvg_follow_dn_3,
            self.fvg_follow_bars,
            self.fvg_follow_net,
        ], dtype=np.float32)
    
    @staticmethod
    def feature_names() -> list[str]:
        return [
            "fvg_event",
            "fvg_impulse_atr",
            "fvg_location_15m",
            "fvg_follow_up_3",
            "fvg_follow_dn_3",
            "fvg_follow_bars",
            "fvg_follow_net",
        ]
    
    @staticmethod
    def n_features() -> int:
        return 7


@dataclass
class FVGEventConfig:
    """
    FVG 事件特征配置
    
    Args:
        min_size_bps: 最小 FVG 大小（基点）
        follow_window: 跟随窗口（bars），超过此窗口后 follow 特征衰减
        atr_period_1m: 1m ATR 计算周期
        cooldown_bars: 冷却期（bars），抑制连续触发使 fvg_event 更"脉冲化"
                       默认 2 = 至少间隔 2 bars 才能触发新事件
    """
    min_size_bps: float = 0.1
    follow_window: int = 3  # 最多跟踪事件后 3 bars
    atr_period_1m: int = 14
    cooldown_bars: int = 2  # 冷却期：抑制连续触发


@dataclass
class FVGEventCalculator:
    """
    FVG 事件特征计算器
    
    关键特性：
    - fvg_event 是严格脉冲信号：仅在 t0 事件 bar 取值 {-1, +1}，其余 bar 为 0
    - 冷却期抑制：连续触发会被抑制（cooldown_bars），确保事件稀疏、不变成 "regime flag"
    - follow-through 是因果的（只使用已发生的数据）
    - 支持批量和流式两种模式
    
    语义冻结（Event-Centered LNN 依赖）：
    - fvg_event ∈ {-1, 0, +1}
    - 只在 t0 触发（follow window 用 temporal recall 覆盖）
    - 默认 cooldown_bars=2：至少间隔 2 bars 才能触发新事件
    
    使用方式：
    ```python
    calc = FVGEventCalculator(config)
    
    # 批量模式
    features = calc.compute_batch(bars_1m, mid_15m, atr_1m, atr_15m)
    
    # 流式模式
    for bar in stream:
        feat = calc.update(bar, mid_15m, atr_1m, atr_15m)
    ```
    """
    config: FVGEventConfig = field(default_factory=FVGEventConfig)
    
    # 内部状态
    _bars: list[Bar] = field(default_factory=list, init=False)
    _event_state: FVGEventState = field(default_factory=FVGEventState, init=False)
    _atr_buffer: list[float] = field(default_factory=list, init=False)
    _current_atr: float = field(default=0.0, init=False)
    
    def __post_init__(self) -> None:
        logger.info(
            "FVGEventCalculator initialized",
            config=self.config.__dict__,
        )
    
    def reset(self) -> None:
        """重置状态"""
        self._bars.clear()
        self._event_state.reset()
        self._atr_buffer.clear()
        self._current_atr = 0.0
    
    def compute_batch(
        self,
        bars: Sequence[EventBar] | Sequence[Bar],
        mid_15m_arr: NDArray[np.float64] | None = None,
        atr_1m_arr: NDArray[np.float64] | None = None,
        atr_15m_arr: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float32]:
        """
        批量计算 FVG 事件特征
        
        Args:
            bars: 1m bar 序列
            mid_15m_arr: 对齐到 1m 的 15m mid-range (可选)
            atr_1m_arr: 1m ATR 序列 (可选，如果不提供则内部计算)
            atr_15m_arr: 对齐到 1m 的 15m ATR (可选)
            
        Returns:
            特征矩阵 (n_bars, 7)
        """
        self.reset()
        
        n_bars = len(bars)
        features = np.zeros((n_bars, FVGEventFeaturesResult.n_features()), dtype=np.float32)
        
        for i, bar in enumerate(bars):
            # 转换 bar 格式
            if isinstance(bar, EventBar):
                bar = self._convert_bar(bar)
            
            mid_15m = mid_15m_arr[i] if mid_15m_arr is not None else 0.0
            atr_1m = atr_1m_arr[i] if atr_1m_arr is not None else None
            atr_15m = atr_15m_arr[i] if atr_15m_arr is not None else 1.0
            
            result = self._process_bar(bar, i, mid_15m, atr_1m, atr_15m)
            features[i] = result.to_array()
        
        logger.info(
            "FVG event features computed (batch)",
            n_bars=n_bars,
        )
        
        return features
    
    def update(
        self,
        bar: EventBar | Bar,
        mid_15m: float = 0.0,
        atr_1m: float | None = None,
        atr_15m: float = 1.0,
    ) -> FVGEventFeaturesResult:
        """
        流式更新（单 bar）
        
        Args:
            bar: 当前 bar
            mid_15m: 当前对齐的 15m mid-range
            atr_1m: 当前 1m ATR (可选)
            atr_15m: 当前 15m ATR
            
        Returns:
            FVG 事件特征结果
        """
        if isinstance(bar, EventBar):
            bar = self._convert_bar(bar)
        
        bar_idx = len(self._bars)
        return self._process_bar(bar, bar_idx, mid_15m, atr_1m, atr_15m)
    
    def _convert_bar(self, event_bar: EventBar) -> Bar:
        """将 EventBar 转换为 Bar"""
        return Bar(
            time=event_bar.time,
            open=event_bar.open,
            high=event_bar.high,
            low=event_bar.low,
            close=event_bar.close,
        )
    
    def _process_bar(
        self,
        bar: Bar,
        bar_idx: int,
        mid_15m: float,
        atr_1m: float | None,
        atr_15m: float,
    ) -> FVGEventFeaturesResult:
        """处理单个 bar"""
        self._bars.append(bar)
        
        # 更新 ATR (如果没有提供外部 ATR)
        if atr_1m is None:
            self._update_atr(bar)
            atr_1m = max(self._current_atr, EPSILON)
        
        atr_15m = max(atr_15m, EPSILON)
        
        result = FVGEventFeaturesResult()
        
        # 1. 检测 FVG 事件（需要至少 3 bars）
        fvg_detected = False
        fvg_type = 0
        fvg_size = 0.0
        
        if len(self._bars) >= 3:
            bar_0 = self._bars[-3]  # 2 bars ago
            bar_1 = self._bars[-2]  # 1 bar ago (impulse)
            bar_2 = self._bars[-1]  # Current bar
            
            mid_price = (bar_1.high + bar_1.low) / 2
            
            # Bullish FVG: bar_0.high < bar_2.low
            if bar_0.high < bar_2.low:
                fvg_size = bar_2.low - bar_0.high
                size_bps = fvg_size / mid_price * 10000
                if size_bps >= self.config.min_size_bps:
                    fvg_detected = True
                    fvg_type = 1
            
            # Bearish FVG: bar_0.low > bar_2.high
            elif bar_0.low > bar_2.high:
                fvg_size = bar_0.low - bar_2.high
                size_bps = fvg_size / mid_price * 10000
                if size_bps >= self.config.min_size_bps:
                    fvg_detected = True
                    fvg_type = -1
        
        # 2. 如果检测到新 FVG，检查冷却期并更新事件状态
        # 冷却期逻辑：抑制连续触发，使 fvg_event 更"脉冲化"
        # fvg_event ∈ {−1, 0, +1}，只在 t0 触发
        if fvg_detected:
            # 检查是否在冷却期内（距离上一个事件太近）
            bars_since_last_event = bar_idx - self._event_state.event_bar_idx
            in_cooldown = (
                self._event_state.event_bar_idx >= 0 and  # 有过事件
                bars_since_last_event <= self.config.cooldown_bars  # 在冷却期内
            )
            
            if not in_cooldown:
                # 冷却期已过，允许触发新事件
                result.fvg_event = fvg_type
                result.fvg_impulse_atr = fvg_size / atr_1m
                
                # 重置事件跟踪状态
                self._event_state.event_bar_idx = bar_idx
                self._event_state.event_type = fvg_type
                self._event_state.event_close = bar.close
                self._event_state.impulse_atr = fvg_size / atr_1m
                self._event_state.max_up_since = 0.0
                self._event_state.max_dn_since = 0.0
            # else: 在冷却期内，不触发新事件（fvg_event 保持默认 0）
        
        # 3. 计算 FVG 位置（相对于 15m mid-range）
        if mid_15m > 0:
            result.fvg_location_15m = (bar.close - mid_15m) / atr_15m
        
        # 4. 计算因果 follow-through（仅在有事件状态时）
        if self._event_state.event_bar_idx >= 0:
            bars_since_event = bar_idx - self._event_state.event_bar_idx
            
            # 只在事件后的窗口内更新
            if 0 < bars_since_event <= self.config.follow_window:
                # 更新最大上涨/下跌
                price_change = bar.close - self._event_state.event_close
                
                if price_change > self._event_state.max_up_since:
                    self._event_state.max_up_since = price_change
                if price_change < self._event_state.max_dn_since:
                    self._event_state.max_dn_since = price_change
            
            # 设置 follow-through 特征
            result.fvg_follow_bars = bars_since_event
            result.fvg_follow_up_3 = self._event_state.max_up_since / atr_1m
            result.fvg_follow_dn_3 = self._event_state.max_dn_since / atr_1m
            result.fvg_follow_net = (bar.close - self._event_state.event_close) / atr_1m
        
        return result
    
    def _update_atr(self, bar: Bar) -> None:
        """更新内部 ATR"""
        if len(self._bars) < 2:
            self._current_atr = bar.high - bar.low
            return
        
        prev_bar = self._bars[-2]
        tr = max(
            bar.high - bar.low,
            abs(bar.high - prev_bar.close),
            abs(bar.low - prev_bar.close),
        )
        
        self._atr_buffer.append(tr)
        
        if len(self._atr_buffer) > self.config.atr_period_1m:
            self._atr_buffer = self._atr_buffer[-self.config.atr_period_1m:]
        
        self._current_atr = sum(self._atr_buffer) / len(self._atr_buffer)
    
    @property
    def last_event_state(self) -> FVGEventState:
        """获取最近的事件状态"""
        return self._event_state
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "n_bars": len(self._bars),
            "current_atr": round(self._current_atr, 4),
            "has_event": self._event_state.event_bar_idx >= 0,
            "event_type": self._event_state.event_type,
            "bars_since_event": len(self._bars) - 1 - self._event_state.event_bar_idx if self._event_state.event_bar_idx >= 0 else -1,
        }


@dataclass
class ATRRatioCalculator:
    """
    ATR 比率计算器
    
    计算 ATR_1m / ATR_15m 比率，用于 XGB 筛选。
    """
    atr_period: int = 14
    
    _atr_1m_buffer: list[float] = field(default_factory=list, init=False)
    _atr_15m_buffer: list[float] = field(default_factory=list, init=False)
    
    def compute_batch(
        self,
        atr_1m_arr: NDArray[np.float64],
        atr_15m_arr: NDArray[np.float64],
    ) -> NDArray[np.float32]:
        """计算 ATR 比率"""
        atr_15m_safe = np.maximum(atr_15m_arr, EPSILON)
        return (atr_1m_arr / atr_15m_safe).astype(np.float32)
    
    def update(self, atr_1m: float, atr_15m: float) -> float:
        """流式更新"""
        atr_15m = max(atr_15m, EPSILON)
        return atr_1m / atr_15m


@dataclass
class STAlignmentCalculator:
    """
    SuperTrend 对齐计算器
    
    计算 st_alignment = st_trend_15m * fvg_event
    """
    
    @staticmethod
    def compute_batch(
        st_trend_15m: NDArray[np.int32],
        fvg_event: NDArray[np.int32],
    ) -> NDArray[np.float32]:
        """批量计算 ST 对齐"""
        return (st_trend_15m * fvg_event).astype(np.float32)
    
    @staticmethod
    def compute(st_trend_15m: int, fvg_event: int) -> float:
        """单次计算"""
        return float(st_trend_15m * fvg_event)
