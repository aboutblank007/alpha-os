"""
FVG (Fair Value Gap) 特征计算模块

将 FVG 从信号过滤器转换为 ML 友好的特征：
- 不过滤逆趋势 FVG，全部记录为特征
- 用 ATR 归一化大小/距离
- 输出连续值特征供模型学习

特征列表：
- bullish_fvg: 是否存在 Bullish FVG (0/1)
- bearish_fvg: 是否存在 Bearish FVG (0/1)
- fvg_size_atr: 最近 FVG 大小 / ATR
- fvg_age_bars: FVG 形成后的 bar 数
- price_to_fvg_mid: (Close - FVG_mid) / ATR
- fvg_filled_pct: FVG 回补程度 (0/0.5/1)
- active_fvg_count: 当前活跃 FVG 数量
- nearest_fvg_distance: 距离最近 FVG 的距离 / ATR
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from alphaos.core.logging import get_logger
from alphaos.v4.types import Bar
from alphaos.data.event_bars.tick_imbalance import EventBar

logger = get_logger(__name__)

EPSILON = 1e-10


class FVGTypeML(Enum):
    """FVG types for ML features."""
    NONE = 0
    BULLISH = 1
    BEARISH = 2


@dataclass
class FVGState:
    """
    单个 FVG 的状态（用于特征计算）
    
    Attributes:
        fvg_type: FVG 类型 (BULLISH/BEARISH)
        top: 上边界
        bottom: 下边界
        midpoint: 中点
        size: 大小 (价格单位)
        bar_idx: 形成的 bar 索引
        is_active: 是否仍然有效
        filled_pct: 回补程度 (0-1)
    """
    fvg_type: FVGTypeML
    top: float
    bottom: float
    midpoint: float
    size: float
    bar_idx: int
    is_active: bool = True
    filled_pct: float = 0.0
    
    @property
    def age(self) -> int:
        """Placeholder - actual age computed externally."""
        return 0


@dataclass
class FVGFeaturesResult:
    """
    FVG 特征计算结果
    
    Attributes:
        bullish_fvg: 是否存在 Bullish FVG (0/1)
        bearish_fvg: 是否存在 Bearish FVG (0/1)
        fvg_size_atr: FVG 大小 / ATR（最近的 FVG）
        fvg_age_bars: FVG 年龄（bars）
        price_to_fvg_mid: (Price - FVG_mid) / ATR
        fvg_filled_pct: 回补程度 (0-1)
        active_fvg_count: 活跃 FVG 数量
        nearest_fvg_distance: 距离最近 FVG / ATR
    """
    bullish_fvg: int = 0
    bearish_fvg: int = 0
    fvg_size_atr: float = 0.0
    fvg_age_bars: int = 0
    price_to_fvg_mid: float = 0.0
    fvg_filled_pct: float = 0.0
    active_fvg_count: int = 0
    nearest_fvg_distance: float = 0.0
    
    def to_array(self) -> NDArray[np.float32]:
        """转换为 numpy 数组"""
        return np.array([
            self.bullish_fvg,
            self.bearish_fvg,
            self.fvg_size_atr,
            self.fvg_age_bars,
            self.price_to_fvg_mid,
            self.fvg_filled_pct,
            self.active_fvg_count,
            self.nearest_fvg_distance,
        ], dtype=np.float32)
    
    @staticmethod
    def feature_names() -> list[str]:
        """特征名称列表"""
        return [
            "bullish_fvg",
            "bearish_fvg", 
            "fvg_size_atr",
            "fvg_age_bars",
            "price_to_fvg_mid",
            "fvg_filled_pct",
            "active_fvg_count",
            "nearest_fvg_distance",
        ]
    
    @staticmethod
    def n_features() -> int:
        """特征数量"""
        return 8


@dataclass
class FVGFeatureConfig:
    """
    FVG 特征计算配置
    
    Args:
        min_size_bps: 最小 FVG 大小（基点）- 用于检测，不用于过滤
        max_age_bars: FVG 最大有效期（bars）
        max_active_fvgs: 最多跟踪的活跃 FVG 数量
        atr_period: ATR 计算周期（用于归一化）
    """
    min_size_bps: float = 0.1  # 低阈值，记录更多 FVG
    max_age_bars: int = 50     # 较长有效期
    max_active_fvgs: int = 10  # 最多跟踪 10 个
    atr_period: int = 14       # ATR 周期


@dataclass
class FVGFeatureCalculator:
    """
    FVG 特征计算器
    
    将 FVG 检测结果转换为 ML 特征。
    
    关键设计：
    - 不过滤逆趋势 FVG - 让模型学习什么是有价值的
    - 所有距离/大小用 ATR 归一化
    - 输出连续值而非二值
    
    使用方式：
    ```python
    calc = FVGFeatureCalculator(config)
    
    # 批量计算
    features = calc.compute_batch(bars)  # (n_bars, 8)
    
    # 流式计算
    for bar in stream:
        feat = calc.update(bar)
    ```
    """
    config: FVGFeatureConfig = field(default_factory=FVGFeatureConfig)
    
    # 内部状态
    _bars: list[Bar] = field(default_factory=list, init=False)
    _active_fvgs: list[FVGState] = field(default_factory=list, init=False)
    _atr_buffer: list[float] = field(default_factory=list, init=False)
    _current_atr: float = field(default=0.0, init=False)
    
    def __post_init__(self) -> None:
        """初始化"""
        logger.info(
            "FVGFeatureCalculator initialized",
            config=self.config.__dict__,
        )
    
    def reset(self) -> None:
        """重置状态"""
        self._bars.clear()
        self._active_fvgs.clear()
        self._atr_buffer.clear()
        self._current_atr = 0.0
    
    def compute_batch(
        self,
        bars: Sequence[EventBar] | Sequence[Bar],
    ) -> NDArray[np.float32]:
        """
        批量计算 FVG 特征
        
        Args:
            bars: Bar 序列
            
        Returns:
            特征矩阵 (n_bars, 8)
        """
        self.reset()
        
        n_bars = len(bars)
        features = np.zeros((n_bars, FVGFeaturesResult.n_features()), dtype=np.float32)
        
        for i, bar in enumerate(bars):
            # 转换为 Bar 格式
            if isinstance(bar, EventBar):
                bar = self._convert_bar(bar)
            
            result = self._process_bar(bar, i)
            features[i] = result.to_array()
        
        logger.info(
            "FVG features computed (batch)",
            n_bars=n_bars,
            total_fvgs_detected=len(self._active_fvgs),
        )
        
        return features
    
    def update(self, bar: EventBar | Bar) -> FVGFeaturesResult:
        """
        流式更新（单 bar）
        
        Args:
            bar: 当前 bar
            
        Returns:
            FVG 特征结果
        """
        if isinstance(bar, EventBar):
            bar = self._convert_bar(bar)
        
        bar_idx = len(self._bars)
        return self._process_bar(bar, bar_idx)
    
    def _convert_bar(self, event_bar: EventBar) -> Bar:
        """将 EventBar 转换为 Bar"""
        return Bar(
            time=event_bar.time,
            open=event_bar.open,
            high=event_bar.high,
            low=event_bar.low,
            close=event_bar.close,
        )
    
    def _process_bar(self, bar: Bar, bar_idx: int) -> FVGFeaturesResult:
        """处理单个 bar"""
        self._bars.append(bar)
        
        # 更新 ATR
        self._update_atr(bar)
        
        # 更新现有 FVG 状态（检查回补）
        self._update_fvg_states(bar)
        
        # 检测新 FVG（需要至少 3 bars）
        if len(self._bars) >= 3:
            self._detect_new_fvg(bar_idx)
        
        # 清理过期 FVG
        self._cleanup_expired_fvgs(bar_idx)
        
        # 计算特征
        return self._compute_features(bar, bar_idx)
    
    def _update_atr(self, bar: Bar) -> None:
        """更新 ATR"""
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
        
        # 保持缓冲区大小
        if len(self._atr_buffer) > self.config.atr_period:
            self._atr_buffer = self._atr_buffer[-self.config.atr_period:]
        
        # 计算 ATR
        if len(self._atr_buffer) >= self.config.atr_period:
            self._current_atr = sum(self._atr_buffer) / len(self._atr_buffer)
        else:
            self._current_atr = sum(self._atr_buffer) / len(self._atr_buffer) if self._atr_buffer else tr
    
    def _detect_new_fvg(self, current_idx: int) -> None:
        """检测新 FVG"""
        bar_0 = self._bars[-3]  # 2 bars ago
        bar_1 = self._bars[-2]  # 1 bar ago (impulse)
        bar_2 = self._bars[-1]  # Current bar
        
        mid_price = (bar_1.high + bar_1.low) / 2
        
        # Bullish FVG: bar_0.high < bar_2.low (gap up)
        if bar_0.high < bar_2.low:
            gap_bottom = bar_0.high
            gap_top = bar_2.low
            size = gap_top - gap_bottom
            size_bps = size / mid_price * 10000
            
            if size_bps >= self.config.min_size_bps:
                fvg = FVGState(
                    fvg_type=FVGTypeML.BULLISH,
                    top=gap_top,
                    bottom=gap_bottom,
                    midpoint=(gap_top + gap_bottom) / 2,
                    size=size,
                    bar_idx=current_idx,
                    is_active=True,
                    filled_pct=0.0,
                )
                self._active_fvgs.append(fvg)
        
        # Bearish FVG: bar_0.low > bar_2.high (gap down)
        if bar_0.low > bar_2.high:
            gap_top = bar_0.low
            gap_bottom = bar_2.high
            size = gap_top - gap_bottom
            size_bps = size / mid_price * 10000
            
            if size_bps >= self.config.min_size_bps:
                fvg = FVGState(
                    fvg_type=FVGTypeML.BEARISH,
                    top=gap_top,
                    bottom=gap_bottom,
                    midpoint=(gap_top + gap_bottom) / 2,
                    size=size,
                    bar_idx=current_idx,
                    is_active=True,
                    filled_pct=0.0,
                )
                self._active_fvgs.append(fvg)
        
        # 限制活跃 FVG 数量
        if len(self._active_fvgs) > self.config.max_active_fvgs:
            self._active_fvgs = self._active_fvgs[-self.config.max_active_fvgs:]
    
    def _update_fvg_states(self, bar: Bar) -> None:
        """更新 FVG 状态（检查回补程度）"""
        for fvg in self._active_fvgs:
            if not fvg.is_active:
                continue
            
            if fvg.fvg_type == FVGTypeML.BULLISH:
                # Bullish FVG: 价格下跌填补
                if bar.low <= fvg.bottom:
                    fvg.filled_pct = 1.0
                    fvg.is_active = False
                elif bar.low <= fvg.midpoint:
                    # 部分回补
                    fill_depth = fvg.top - bar.low
                    fvg.filled_pct = min(1.0, fill_depth / fvg.size)
            
            elif fvg.fvg_type == FVGTypeML.BEARISH:
                # Bearish FVG: 价格上涨填补
                if bar.high >= fvg.top:
                    fvg.filled_pct = 1.0
                    fvg.is_active = False
                elif bar.high >= fvg.midpoint:
                    # 部分回补
                    fill_depth = bar.high - fvg.bottom
                    fvg.filled_pct = min(1.0, fill_depth / fvg.size)
    
    def _cleanup_expired_fvgs(self, current_idx: int) -> None:
        """清理过期 FVG"""
        still_active = []
        for fvg in self._active_fvgs:
            age = current_idx - fvg.bar_idx
            if age < self.config.max_age_bars and fvg.is_active:
                still_active.append(fvg)
            elif fvg.filled_pct < 1.0:
                # 保留部分回补但未完全回补的
                still_active.append(fvg)
        
        self._active_fvgs = still_active[-self.config.max_active_fvgs:]
    
    def _compute_features(self, bar: Bar, bar_idx: int) -> FVGFeaturesResult:
        """计算当前 bar 的 FVG 特征"""
        result = FVGFeaturesResult()
        
        if not self._active_fvgs:
            return result
        
        atr = max(self._current_atr, EPSILON)
        price = bar.close
        
        # 统计活跃 FVG
        bullish_fvgs = [f for f in self._active_fvgs if f.fvg_type == FVGTypeML.BULLISH]
        bearish_fvgs = [f for f in self._active_fvgs if f.fvg_type == FVGTypeML.BEARISH]
        
        result.bullish_fvg = 1 if bullish_fvgs else 0
        result.bearish_fvg = 1 if bearish_fvgs else 0
        result.active_fvg_count = len(self._active_fvgs)
        
        # 找最近的 FVG
        nearest_fvg = None
        nearest_distance = float('inf')
        
        for fvg in self._active_fvgs:
            dist = abs(price - fvg.midpoint)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_fvg = fvg
        
        if nearest_fvg is not None:
            result.fvg_size_atr = nearest_fvg.size / atr
            result.fvg_age_bars = bar_idx - nearest_fvg.bar_idx
            result.price_to_fvg_mid = (price - nearest_fvg.midpoint) / atr
            result.fvg_filled_pct = nearest_fvg.filled_pct
            result.nearest_fvg_distance = nearest_distance / atr
        
        return result
    
    @property
    def active_fvg_count(self) -> int:
        """活跃 FVG 数量"""
        return len(self._active_fvgs)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "n_bars": len(self._bars),
            "active_fvgs": self.active_fvg_count,
            "current_atr": round(self._current_atr, 4),
            "bullish_fvgs": sum(1 for f in self._active_fvgs if f.fvg_type == FVGTypeML.BULLISH),
            "bearish_fvgs": sum(1 for f in self._active_fvgs if f.fvg_type == FVGTypeML.BEARISH),
        }
