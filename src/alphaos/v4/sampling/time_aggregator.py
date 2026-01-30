"""
时间 Bar 聚合器

将 EventBar 序列聚合为固定时间间隔的 Bar：
- 支持 1m, 5m, 15m, 30m, 1h 等时间框架
- 批量和流式两种模式
- 用于多时间框架特征计算

使用方式：
```python
aggregator = TimeBarAggregator(interval_seconds=900)  # 15m

# 批量聚合
bars_15m = aggregator.aggregate_batch(event_bars)

# 流式聚合
for event_bar in stream:
    bar_15m = aggregator.update(event_bar)
    if bar_15m is not None:
        # 15m bar 完成
        process(bar_15m)
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from alphaos.core.logging import get_logger
from alphaos.v4.types import Bar, Timeframe
from alphaos.data.event_bars.tick_imbalance import EventBar

logger = get_logger(__name__)


@dataclass
class TimeBar:
    """
    固定时间间隔的 Bar
    
    Attributes:
        time: Bar 开始时间
        open: 开盘价
        high: 最高价
        low: 最低价
        close: 收盘价
        tick_volume: Tick 数量
        event_bar_count: 包含的 EventBar 数量
        duration_seconds: 实际持续秒数
    """
    time: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int = 0
    event_bar_count: int = 0
    duration_seconds: int = 0
    
    @property
    def range(self) -> float:
        """Bar 范围"""
        return self.high - self.low
    
    @property
    def body(self) -> float:
        """Bar 实体"""
        return self.close - self.open
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        return self.close < self.open
    
    def to_dict(self) -> dict:
        return {
            "time": self.time.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "tick_volume": self.tick_volume,
            "event_bar_count": self.event_bar_count,
            "duration_seconds": self.duration_seconds,
        }
    
    def to_bar(self) -> Bar:
        """转换为标准 Bar 类型"""
        return Bar(
            time=self.time,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            tick_volume=self.tick_volume,
        )


@dataclass
class TimeBarAggregatorConfig:
    """
    时间聚合器配置
    
    Args:
        interval_seconds: 聚合间隔（秒）
        align_to_interval: 是否对齐到整数时间间隔
        fill_gaps: 是否填充空缺的 bar
    """
    interval_seconds: int = 900  # 15 分钟
    align_to_interval: bool = True
    fill_gaps: bool = False  # 暂不填充空缺
    
    @classmethod
    def from_timeframe(cls, tf: Timeframe) -> "TimeBarAggregatorConfig":
        """从 Timeframe 创建配置"""
        return cls(interval_seconds=tf.value)


@dataclass
class TimeBarAggregator:
    """
    时间 Bar 聚合器
    
    将 EventBar 聚合为固定时间间隔的 Bar。
    
    关键逻辑：
    1. 根据 EventBar 的时间戳确定所属的时间窗口
    2. 在同一窗口内聚合 OHLCV
    3. 窗口结束时输出完成的 TimeBar
    
    使用方式：
    ```python
    # 批量模式
    agg = TimeBarAggregator(interval_seconds=900)
    bars_15m = agg.aggregate_batch(event_bars)
    
    # 流式模式
    for eb in stream:
        tb = agg.update(eb)
        if tb:
            handle_completed_bar(tb)
    ```
    """
    config: TimeBarAggregatorConfig = field(default_factory=TimeBarAggregatorConfig)
    
    # 内部状态
    _current_window_start: datetime | None = field(default=None, init=False)
    _current_open: float = field(default=0.0, init=False)
    _current_high: float = field(default=0.0, init=False)
    _current_low: float = field(default=float('inf'), init=False)
    _current_close: float = field(default=0.0, init=False)
    _current_tick_volume: int = field(default=0, init=False)
    _current_event_bar_count: int = field(default=0, init=False)
    _completed_bars: list[TimeBar] = field(default_factory=list, init=False)
    
    def __post_init__(self) -> None:
        """初始化"""
        logger.info(
            "TimeBarAggregator initialized",
            interval_seconds=self.config.interval_seconds,
            align_to_interval=self.config.align_to_interval,
        )
    
    def reset(self) -> None:
        """重置状态"""
        self._current_window_start = None
        self._current_open = 0.0
        self._current_high = 0.0
        self._current_low = float('inf')
        self._current_close = 0.0
        self._current_tick_volume = 0
        self._current_event_bar_count = 0
        self._completed_bars.clear()
    
    def aggregate_batch(
        self,
        event_bars: Sequence[EventBar],
    ) -> list[TimeBar]:
        """
        批量聚合 EventBar 为 TimeBar
        
        Args:
            event_bars: EventBar 序列
            
        Returns:
            TimeBar 列表
        """
        self.reset()
        
        for eb in event_bars:
            self.update(eb)
        
        # 最后一个未完成的 bar 也输出
        if self._current_window_start is not None:
            final_bar = self._finalize_current_bar()
            if final_bar:
                self._completed_bars.append(final_bar)
        
        logger.info(
            "Time bar aggregation complete",
            input_bars=len(event_bars),
            output_bars=len(self._completed_bars),
            interval_seconds=self.config.interval_seconds,
        )
        
        return self._completed_bars.copy()
    
    def update(self, event_bar: EventBar) -> TimeBar | None:
        """
        流式更新
        
        Args:
            event_bar: 新的 EventBar
            
        Returns:
            完成的 TimeBar（如果有），否则 None
        """
        bar_time = event_bar.time
        window_start = self._get_window_start(bar_time)
        
        completed_bar = None
        
        # 检查是否需要开始新窗口
        if self._current_window_start is None:
            # 第一个 bar
            self._start_new_window(window_start, event_bar)
        elif window_start > self._current_window_start:
            # 新窗口开始，完成当前 bar
            completed_bar = self._finalize_current_bar()
            if completed_bar:
                self._completed_bars.append(completed_bar)
            self._start_new_window(window_start, event_bar)
        else:
            # 同一窗口，更新 OHLCV
            self._update_current_bar(event_bar)
        
        return completed_bar
    
    def _get_window_start(self, dt: datetime) -> datetime:
        """计算时间窗口的开始时间"""
        if not self.config.align_to_interval:
            return dt
        
        interval = self.config.interval_seconds
        
        # 计算从午夜开始的秒数
        seconds_from_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
        
        # 对齐到间隔
        aligned_seconds = (seconds_from_midnight // interval) * interval
        
        # 构建对齐后的时间
        aligned_dt = dt.replace(
            hour=aligned_seconds // 3600,
            minute=(aligned_seconds % 3600) // 60,
            second=aligned_seconds % 60,
            microsecond=0,
        )
        
        return aligned_dt
    
    def _start_new_window(self, window_start: datetime, event_bar: EventBar) -> None:
        """开始新的时间窗口"""
        self._current_window_start = window_start
        self._current_open = event_bar.open
        self._current_high = event_bar.high
        self._current_low = event_bar.low
        self._current_close = event_bar.close
        self._current_tick_volume = event_bar.tick_count
        self._current_event_bar_count = 1
    
    def _update_current_bar(self, event_bar: EventBar) -> None:
        """更新当前窗口的 OHLCV"""
        self._current_high = max(self._current_high, event_bar.high)
        self._current_low = min(self._current_low, event_bar.low)
        self._current_close = event_bar.close
        self._current_tick_volume += event_bar.tick_count
        self._current_event_bar_count += 1
    
    def _finalize_current_bar(self) -> TimeBar | None:
        """完成当前 bar"""
        if self._current_window_start is None:
            return None
        
        if self._current_event_bar_count == 0:
            return None
        
        bar = TimeBar(
            time=self._current_window_start,
            open=self._current_open,
            high=self._current_high,
            low=self._current_low,
            close=self._current_close,
            tick_volume=self._current_tick_volume,
            event_bar_count=self._current_event_bar_count,
            duration_seconds=self.config.interval_seconds,
        )
        
        return bar
    
    @property
    def current_bar_count(self) -> int:
        """当前已完成的 bar 数量"""
        return len(self._completed_bars)
    
    @property
    def interval_seconds(self) -> int:
        """聚合间隔（秒）"""
        return self.config.interval_seconds
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "interval_seconds": self.config.interval_seconds,
            "completed_bars": len(self._completed_bars),
            "current_window_start": self._current_window_start.isoformat() if self._current_window_start else None,
            "current_event_bar_count": self._current_event_bar_count,
        }


def create_multi_timeframe_bars(
    event_bars: Sequence[EventBar],
    timeframes: list[Timeframe] | list[int],
) -> dict[int, list[TimeBar]]:
    """
    创建多时间框架的 Bar
    
    Args:
        event_bars: EventBar 序列
        timeframes: 时间框架列表（Timeframe 枚举或秒数）
        
    Returns:
        时间框架（秒）-> TimeBar 列表的映射
    """
    result = {}
    
    for tf in timeframes:
        interval = tf.value if isinstance(tf, Timeframe) else tf
        aggregator = TimeBarAggregator(
            config=TimeBarAggregatorConfig(interval_seconds=interval)
        )
        result[interval] = aggregator.aggregate_batch(event_bars)
    
    logger.info(
        "Multi-timeframe bars created",
        timeframes=[tf if isinstance(tf, int) else tf.value for tf in timeframes],
        bar_counts={k: len(v) for k, v in result.items()},
    )
    
    return result


def align_features_to_lower_timeframe(
    higher_tf_bars: list[TimeBar],
    lower_tf_bars: list[EventBar] | list[TimeBar],
) -> NDArray[np.int64]:
    """
    将高时间框架特征对齐到低时间框架
    
    返回每个低时间框架 bar 对应的高时间框架 bar 索引。
    
    Args:
        higher_tf_bars: 高时间框架 bar 列表
        lower_tf_bars: 低时间框架 bar 列表
        
    Returns:
        索引数组，形状 (len(lower_tf_bars),)
    """
    if not higher_tf_bars:
        return np.zeros(len(lower_tf_bars), dtype=np.int64)
    
    # 构建高时间框架的时间戳
    htf_times = [b.time for b in higher_tf_bars]
    
    # 为每个低时间框架 bar 找对应的高时间框架 bar
    alignment = np.zeros(len(lower_tf_bars), dtype=np.int64)
    
    htf_idx = 0
    for i, ltf_bar in enumerate(lower_tf_bars):
        ltf_time = ltf_bar.time
        
        # 找到最近的高时间框架 bar（不超过当前时间）
        while htf_idx < len(htf_times) - 1 and htf_times[htf_idx + 1] <= ltf_time:
            htf_idx += 1
        
        alignment[i] = htf_idx
    
    return alignment
