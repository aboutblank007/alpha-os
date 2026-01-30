"""
v4.0 Primary Engine

统一的 Primary 信号引擎，封装 PivotSuperTrend + FVG：
- PivotSuperTrend: 趋势过滤（方向确认）
- FVG: 入场时机（价值回撤）

信号生成逻辑：
1. SuperTrend 确定趋势方向（LONG/SHORT）
2. FVG 检测价值缺口
3. 当价格进入 FVG 的 CE（Consequent Encroachment）点时触发信号
4. 输出标准化的 PrimarySignalV4

参考：交易模型研究.md Section 4

⚠️ SEMANTIC CONSTRAINT (架构约束)
=====================================
PrimaryEngine outputs are treated as EVENT TRIGGERS, NOT predictive features.

✅ 允许的用途：
  - 作为 event gate（决定是否进入 meta-model 流程）
  - 确定事件发生的 bar index（训练时的采样点）
  - 推理时作为 filter 条件之一（has_signal）

❌ 禁止的用途：
  - 将 PrimarySignalV4 的任何字段喂入 XGB 作为特征
  - 将 PrimarySignalV4 当作 alpha source 或 prediction target
  - 对 PrimarySignalV4 做 SHAP / 特征重要性分析

原因：PrimaryEngine 的设计是"高召回、低精度"的事件检测器，
     让 XGB 学习其输出会导致模型过拟合于触发条件本身，
     而非学习"触发时机的好坏"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from alphaos.core.logging import get_logger
from alphaos.execution.primary.pivot_supertrend import (
    PivotSuperTrend,
    TrendDirection,
    SuperTrendState,
)
from alphaos.execution.primary.fvg_signal import (
    FVGDetector,
    FVGSignal,
    FVGType,
    PrimarySignal,
    PrimarySignalGenerator,
)
from alphaos.v4.types import Bar
from alphaos.data.event_bars.tick_imbalance import EventBar

logger = get_logger(__name__)


@dataclass
class PrimarySignalV4:
    """
    v4.0 标准化 Primary 信号
    
    ⚠️ 语义约束：此对象是 EVENT TRIGGER，不是 FEATURE SOURCE。
    - 用于标记"事件发生"（gate），不用于"预测未来"（alpha）
    - direction 用于确定交易方向，但不应作为 XGB 特征
    
    Attributes:
        bar_idx: 信号触发的 Bar 索引
        direction: 信号方向 (1=LONG, -1=SHORT)
        entry_price: 建议入场价格
        stop_loss: 结构止损价格
        trend_duration: 趋势持续 Bar 数
        fvg_size_bps: FVG 大小（基点）
        supertrend_line: SuperTrend 线价格
        atr: 当前 ATR 值
        confidence_factors: 置信度因子字典
    """
    bar_idx: int
    direction: int  # 1 = LONG, -1 = SHORT
    entry_price: float
    stop_loss: float
    trend_duration: int
    fvg_size_bps: float
    supertrend_line: float
    atr: float
    confidence_factors: dict = field(default_factory=dict)
    
    def to_tuple(self) -> tuple[int, int]:
        """转换为 (bar_idx, direction) 元组（用于 meta-labeling）"""
        return (self.bar_idx, self.direction)
    
    def to_dict(self) -> dict:
        return {
            "bar_idx": self.bar_idx,
            "direction": self.direction,
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "trend_duration": self.trend_duration,
            "fvg_size_bps": round(self.fvg_size_bps, 2),
            "supertrend_line": round(self.supertrend_line, 2),
            "atr": round(self.atr, 4),
            "confidence_factors": self.confidence_factors,
        }


@dataclass
class PrimaryEngineConfig:
    """
    Primary Engine 配置
    
    Args:
        # PivotSuperTrend 参数（TradingView 原版默认值）
        pivot_lookback: Pivot 确认 lookback (原版 prd=2)
        atr_period: ATR 计算周期 (原版 Pd=10)
        atr_factor: ATR 乘数（带宽）(原版 Factor=3)
        
        # FVG 参数（数据驱动优化）
        min_fvg_size_bps: 最小 FVG 大小（基点）
        max_fvg_age_bars: FVG 最大有效期（Bar 数）
        ce_tolerance_bps: CE 入场容差（基点）
        
        # 信号参数
        min_trend_duration: 最小趋势持续 Bar 数
        cooldown_bars: 信号冷却 Bar 数
        sl_buffer_bps: 止损缓冲（基点）
        
        # 模式
        require_fvg: 是否必须有 FVG 才触发信号
        fvg_entry_mode: FVG 入场模式
            - "immediate": FVG 形成即入场（不等回踩）
            - "ce_retracement": 等价格回踩到 CE 中点再入场
    """
    # PivotSuperTrend 参数（TradingView 原版默认值）
    pivot_lookback: int = 2      # 原版 prd=2
    atr_period: int = 10         # 原版 Pd=10
    atr_factor: float = 3.0      # 原版 Factor=3
    
    # FVG 参数（数据驱动优化：P50 FVG = 0.66 bps）
    min_fvg_size_bps: float = 0.5   # 捕获 ~60% FVG
    max_fvg_age_bars: int = 30      # 延长有效期
    ce_tolerance_bps: float = 1.0   # ≈半个点差
    
    # 信号参数（数据驱动：P75 趋势持续 = 2 bars）
    min_trend_duration: int = 2
    cooldown_bars: int = 3
    sl_buffer_bps: float = 5.0
    
    # 模式
    require_fvg: bool = True    # 必须有 FVG 才触发信号
    fvg_entry_mode: str = "immediate"  # "immediate" | "ce_retracement"
    
    def to_dict(self) -> dict:
        return {
            "pivot_lookback": self.pivot_lookback,
            "atr_period": self.atr_period,
            "atr_factor": self.atr_factor,
            "min_fvg_size_bps": self.min_fvg_size_bps,
            "max_fvg_age_bars": self.max_fvg_age_bars,
            "ce_tolerance_bps": self.ce_tolerance_bps,
            "min_trend_duration": self.min_trend_duration,
            "cooldown_bars": self.cooldown_bars,
            "sl_buffer_bps": self.sl_buffer_bps,
            "require_fvg": self.require_fvg,
            "fvg_entry_mode": self.fvg_entry_mode,
        }


@dataclass
class PrimaryEngineV4:
    """
    v4.0 Primary Signal Engine
    
    封装 PivotSuperTrend + FVG，提供统一的信号接口。
    
    ⚠️ SEMANTIC CONSTRAINT (架构约束)
    ================================
    输出的 PrimarySignalV4 是 EVENT TRIGGER，不是 predictive feature。
    
    - 训练时：用于确定 event_indices（采样哪些 bar 做 meta-labeling）
    - 推理时：用于 filter 条件（has_signal 才进入后续流程）
    - 禁止：将 signal 的任何字段喂入 XGB 作为特征
    
    批量模式（训练）：
    ```python
    engine = PrimaryEngineV4(config)
    signals = engine.generate_signals_batch(bars)
    
    # 用于 meta-labeling（仅用 bar_idx 和 direction，不喂入 XGB）
    signal_tuples = [(s.bar_idx, s.direction) for s in signals]
    ```
    
    流式模式（推理）：
    ```python
    engine = PrimaryEngineV4(config)
    for bar in bar_stream:
        signal = engine.update(bar)
        if signal:
            # 作为 event gate，不是 feature
            pass
    ```
    """
    config: PrimaryEngineConfig = field(default_factory=PrimaryEngineConfig)
    
    # 内部组件
    _supertrend: PivotSuperTrend | None = field(default=None, init=False)
    _fvg_detector: FVGDetector | None = field(default=None, init=False)
    
    # 状态
    _bar_count: int = field(default=0, init=False)
    _signal_count: int = field(default=0, init=False)
    _cooldown_until: int = field(default=0, init=False)
    _bars_buffer: list = field(default_factory=list, init=False)
    
    def __post_init__(self) -> None:
        """初始化组件"""
        self._init_components()
    
    def _init_components(self) -> None:
        """初始化内部组件"""
        self._supertrend = PivotSuperTrend(
            pivot_lookback=self.config.pivot_lookback,
            atr_period=self.config.atr_period,
            atr_factor=self.config.atr_factor,
        )
        
        self._fvg_detector = FVGDetector(
            min_size_bps=self.config.min_fvg_size_bps,
            max_age_bars=self.config.max_fvg_age_bars,
            ce_tolerance_bps=self.config.ce_tolerance_bps,
        )
        
        logger.info(
            "PrimaryEngineV4 initialized",
            config=self.config.to_dict(),
        )
    
    def _convert_to_bar(self, event_bar: EventBar) -> Bar:
        """将 EventBar 转换为 Bar（用于 SuperTrend/FVG）"""
        return Bar(
            time=event_bar.time,
            open=event_bar.open,
            high=event_bar.high,
            low=event_bar.low,
            close=event_bar.close,
        )
    
    def generate_signals_batch(
        self,
        bars: Sequence[EventBar] | Sequence[Bar],
    ) -> list[PrimarySignalV4]:
        """
        批量生成信号（训练模式）
        
        Args:
            bars: Bar 序列
            
        Returns:
            PrimarySignalV4 列表
        """
        signals = []
        
        # 转换为 Bar 格式
        converted_bars = []
        for bar in bars:
            if isinstance(bar, EventBar):
                converted_bars.append(self._convert_to_bar(bar))
            else:
                converted_bars.append(bar)
        
        # 逐 bar 处理
        for i, bar in enumerate(converted_bars):
            signal = self._process_bar(bar, i)
            if signal:
                signals.append(signal)
        
        logger.info(
            "Generated primary signals (batch)",
            n_bars=len(bars),
            n_signals=len(signals),
            long_signals=sum(1 for s in signals if s.direction == 1),
            short_signals=sum(1 for s in signals if s.direction == -1),
        )
        
        return signals
    
    def update(
        self,
        bar: EventBar | Bar,
        current_price: float | None = None,
    ) -> PrimarySignalV4 | None:
        """
        流式更新（单 bar）
        
        Args:
            bar: 当前完成的 Bar
            current_price: 可选的当前价格（用于实时检查）
            
        Returns:
            PrimarySignalV4 如果触发信号，否则 None
        """
        # 转换格式
        if isinstance(bar, EventBar):
            bar = self._convert_to_bar(bar)
        
        return self._process_bar(bar, self._bar_count)
    
    def _process_bar(
        self,
        bar: Bar,
        bar_idx: int,
    ) -> PrimarySignalV4 | None:
        """处理单个 Bar"""
        self._bar_count = bar_idx + 1
        self._bars_buffer.append(bar)
        
        # 保持缓冲区大小
        max_buffer = max(100, self.config.pivot_lookback * 3)
        if len(self._bars_buffer) > max_buffer:
            self._bars_buffer = self._bars_buffer[-max_buffer:]
        
        # 更新 SuperTrend
        st_state = self._supertrend.update(bar)
        
        # 更新 FVG 并获取新检测到的 FVG
        new_fvgs = self._fvg_detector.update(bar)
        
        # 检查信号条件
        signal = self._check_signal(bar, bar_idx, st_state, new_fvgs)
        
        if signal:
            self._signal_count += 1
            self._cooldown_until = bar_idx + self.config.cooldown_bars
        
        return signal
    
    def _check_signal(
        self,
        bar: Bar,
        bar_idx: int,
        st_state: SuperTrendState,
        new_fvgs: list[FVGSignal] | None = None,
    ) -> PrimarySignalV4 | None:
        """
        检查信号条件
        
        信号逻辑：
        1. SuperTrend 确定趋势方向（多/空）
        2. 根据 fvg_entry_mode 决定入场时机：
           - "immediate": FVG 形成即入场（new_fvgs 不为空）
           - "ce_retracement": 等价格回踩到 FVG CE 中点
        3. 如果 require_fvg=False，则只根据趋势方向生成信号
        """
        # 检查冷却期
        if bar_idx < self._cooldown_until:
            return None
        
        # 需要明确的趋势方向
        if st_state.direction == TrendDirection.NONE:
            return None
        
        # 需要最小趋势持续时间
        if st_state.trend_duration < self.config.min_trend_duration:
            return None
        
        price = bar.close
        direction = 1 if st_state.direction == TrendDirection.LONG else -1
        
        # 检查 FVG
        entry_fvg = None
        
        if self.config.require_fvg:
            if self.config.fvg_entry_mode == "immediate":
                # 立即入场模式：检查本 bar 是否有新 FVG 与趋势方向一致
                if new_fvgs:
                    for fvg in new_fvgs:
                        # Bullish FVG + LONG trend = LONG signal
                        # Bearish FVG + SHORT trend = SHORT signal
                        if (fvg.fvg_type == FVGType.BULLISH and 
                            st_state.direction == TrendDirection.LONG):
                            entry_fvg = fvg
                            break
                        elif (fvg.fvg_type == FVGType.BEARISH and 
                              st_state.direction == TrendDirection.SHORT):
                            entry_fvg = fvg
                            break
                
                if entry_fvg is None:
                    return None
            
            else:  # ce_retracement 模式
                # 等待价格回踩到 FVG CE 中点
                entry_fvg = self._fvg_detector.check_ce_entry(price, st_state.direction)
                if entry_fvg is None:
                    return None
        
        # 计算止损
        sl = self._calculate_stop_loss(st_state, entry_fvg, price)
        
        # 构建信号
        entry_price = entry_fvg.ce_midpoint if entry_fvg else price
        fvg_size = entry_fvg.size_bps if entry_fvg else 0.0
        
        confidence_factors = {
            "trend_duration": st_state.trend_duration,
            "fvg_age": entry_fvg.age_bars if entry_fvg else 0,
            "fvg_size_bps": fvg_size,
            "atr": round(st_state.atr, 4),
            "bandwidth": round(st_state.bandwidth, 4),
            "entry_mode": self.config.fvg_entry_mode if self.config.require_fvg else "trend_only",
        }
        
        return PrimarySignalV4(
            bar_idx=bar_idx,
            direction=direction,
            entry_price=entry_price,
            stop_loss=sl,
            trend_duration=st_state.trend_duration,
            fvg_size_bps=fvg_size,
            supertrend_line=st_state.supertrend_line,
            atr=st_state.atr,
            confidence_factors=confidence_factors,
        )
    
    def _calculate_stop_loss(
        self,
        st_state: SuperTrendState,
        fvg: FVGSignal | None,
        price: float,
    ) -> float:
        """计算止损价格"""
        buffer = price * self.config.sl_buffer_bps / 10000
        
        if st_state.direction == TrendDirection.LONG:
            candidates = []
            if fvg:
                candidates.append(fvg.bottom)
            if st_state.pivot_low > 0:
                candidates.append(st_state.pivot_low)
            if st_state.supertrend_line > 0:
                candidates.append(st_state.supertrend_line)
            
            if candidates:
                sl = min(candidates) - buffer
            else:
                sl = price * (1 - self.config.atr_factor * st_state.atr / price)
        
        else:  # SHORT
            candidates = []
            if fvg:
                candidates.append(fvg.top)
            if st_state.pivot_high > 0:
                candidates.append(st_state.pivot_high)
            if st_state.supertrend_line > 0:
                candidates.append(st_state.supertrend_line)
            
            if candidates:
                sl = max(candidates) + buffer
            else:
                sl = price * (1 + self.config.atr_factor * st_state.atr / price)
        
        return sl
    
    @property
    def current_direction(self) -> int:
        """获取当前趋势方向 (1=LONG, -1=SHORT, 0=NONE)"""
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
        """获取当前 SuperTrend 状态"""
        if self._supertrend is None:
            return None
        return self._supertrend.current_state
    
    @property
    def active_fvg_count(self) -> int:
        """获取活跃 FVG 数量"""
        if self._fvg_detector is None:
            return 0
        return self._fvg_detector.active_fvg_count
    
    @property
    def is_ready(self) -> bool:
        """是否准备好生成信号"""
        if self._supertrend is None:
            return False
        return self._supertrend.is_ready
    
    def reset(self) -> None:
        """重置引擎状态"""
        if self._supertrend:
            self._supertrend.reset()
        if self._fvg_detector:
            self._fvg_detector.reset()
        
        self._bar_count = 0
        self._signal_count = 0
        self._cooldown_until = 0
        self._bars_buffer.clear()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "bar_count": self._bar_count,
            "signal_count": self._signal_count,
            "current_direction": self.current_direction,
            "active_fvgs": self.active_fvg_count,
            "is_ready": self.is_ready,
            "cooldown_remaining": max(0, self._cooldown_until - self._bar_count),
        }
    
    def get_trend_info(self) -> dict:
        """获取趋势信息"""
        state = self.current_state
        if state is None:
            return {}
        
        return {
            "direction": state.direction.name,
            "supertrend_line": round(state.supertrend_line, 2),
            "bandwidth": round(state.bandwidth, 4),
            "trend_duration": state.trend_duration,
            "pivot_high": round(state.pivot_high, 2),
            "pivot_low": round(state.pivot_low, 2),
            "atr": round(state.atr, 4),
        }
