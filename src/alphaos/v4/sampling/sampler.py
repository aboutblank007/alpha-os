"""
v4.0 统一采样器

封装 VolumeBarBuilder 和 TickImbalanceBarBuilder，提供：
- 配置驱动的采样模式切换
- 统一的 EventBar 输出格式
- Tick 流处理接口

参考：
- 交易模型研究.md Section 2.3 - Volume Bars
- 降噪LNN特征提取与信号过滤.md Section 2.1 - Imbalance Bars
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from alphaos.core.logging import get_logger
from alphaos.core.types import Tick
from alphaos.data.event_bars.volume_bars import VolumeBarBuilder
from alphaos.data.event_bars.tick_imbalance import TickImbalanceBarBuilder, EventBar

logger = get_logger(__name__)


class SamplingMode(Enum):
    """采样模式"""
    VOLUME_BARS = "volume_bars"          # 成交量条形图（默认）
    TICK_IMBALANCE = "tick_imbalance"    # Tick 失衡条形图
    DOLLAR_IMBALANCE = "dollar_imbalance"  # 美元失衡条形图（TODO）


class VolumeSource(Enum):
    """成交量来源"""
    REAL = "real"              # 真实成交量（MT5 可能为 0）
    TICK_COUNT = "tick_count"  # Tick 计数（回退方案）
    SYNTHETIC = "synthetic"    # 合成成交量（波动率 * 点差）


@dataclass
class SamplingConfig:
    """
    采样配置
    
    Args:
        mode: 采样模式（默认 volume_bars）
        volume_source: 成交量来源（默认 tick_count，用于 Sim2Real）
        
        # Volume Bars 参数
        target_volume: Volume Bar 目标成交量（tick_count 模式下即 tick 数）
        
        # Tick Imbalance Bars 参数
        initial_expected_ticks: 初始期望 Tick 数
        initial_expected_imbalance: 初始期望失衡率
        ewma_alpha: EWMA 衰减系数
        tick_rule_gamma: 贝叶斯 Tick Rule 衰减
        tick_rule_threshold: 中性分类阈值
        
        # 通用参数
        max_buffer_size: 最大 Bar 缓冲数量
        synthetic_base_volume: 合成成交量基准值
    """
    mode: SamplingMode = SamplingMode.VOLUME_BARS
    volume_source: VolumeSource = VolumeSource.TICK_COUNT
    
    # Volume Bars 参数
    target_volume: float = 100.0  # tick_count 模式下 = 100 ticks/bar
    
    # Tick Imbalance Bars 参数
    initial_expected_ticks: float = 50.0
    initial_expected_imbalance: float = 0.5
    ewma_alpha: float = 0.1
    tick_rule_gamma: float = 0.95
    tick_rule_threshold: float = 0.5
    
    # 通用参数
    max_buffer_size: int = 500
    synthetic_base_volume: float = 100.0
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "mode": self.mode.value,
            "volume_source": self.volume_source.value,
            "target_volume": self.target_volume,
            "initial_expected_ticks": self.initial_expected_ticks,
            "initial_expected_imbalance": self.initial_expected_imbalance,
            "ewma_alpha": self.ewma_alpha,
            "tick_rule_gamma": self.tick_rule_gamma,
            "tick_rule_threshold": self.tick_rule_threshold,
            "max_buffer_size": self.max_buffer_size,
            "synthetic_base_volume": self.synthetic_base_volume,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SamplingConfig":
        """从字典创建"""
        return cls(
            mode=SamplingMode(data.get("mode", "volume_bars")),
            volume_source=VolumeSource(data.get("volume_source", "tick_count")),
            target_volume=data.get("target_volume", 100.0),
            initial_expected_ticks=data.get("initial_expected_ticks", 50.0),
            initial_expected_imbalance=data.get("initial_expected_imbalance", 0.5),
            ewma_alpha=data.get("ewma_alpha", 0.1),
            tick_rule_gamma=data.get("tick_rule_gamma", 0.95),
            tick_rule_threshold=data.get("tick_rule_threshold", 0.5),
            max_buffer_size=data.get("max_buffer_size", 500),
            synthetic_base_volume=data.get("synthetic_base_volume", 100.0),
        )


@dataclass
class EventBarStream:
    """
    Event Bar 流数据结构
    
    用于批量处理已完成的 Bars
    """
    bars: list[EventBar] = field(default_factory=list)
    config: SamplingConfig = field(default_factory=SamplingConfig)
    total_ticks: int = 0
    
    def to_numpy(self) -> NDArray:
        """
        转换为 numpy 结构化数组
        
        Returns:
            结构化数组，包含所有 Bar 数据
        """
        from alphaos.v4.schemas import BarSchema
        
        if not self.bars:
            dtype = BarSchema.get_dtype()
            return np.array([], dtype=dtype)
        
        dtype = BarSchema.get_dtype()
        arr = np.zeros(len(self.bars), dtype=dtype)
        
        for i, bar in enumerate(self.bars):
            arr[i]["open_time_us"] = int(bar.time.timestamp() * 1e6) if bar.time else 0
            arr[i]["close_time_us"] = int(bar.close_time.timestamp() * 1e6) if bar.close_time else 0
            arr[i]["open"] = bar.open
            arr[i]["high"] = bar.high
            arr[i]["low"] = bar.low
            arr[i]["close"] = bar.close
            arr[i]["tick_count"] = bar.tick_count
            arr[i]["volume"] = bar.tick_count  # Volume = tick count in our model
            arr[i]["imbalance"] = bar.imbalance
            arr[i]["buy_count"] = bar.buy_count
            arr[i]["sell_count"] = bar.sell_count
            arr[i]["spread_sum"] = bar.spread_sum
            arr[i]["duration_ms"] = bar.duration_ms
        
        return arr
    
    def get_ohlc(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """获取 OHLC 数组"""
        n = len(self.bars)
        opens = np.zeros(n, dtype=np.float64)
        highs = np.zeros(n, dtype=np.float64)
        lows = np.zeros(n, dtype=np.float64)
        closes = np.zeros(n, dtype=np.float64)
        
        for i, bar in enumerate(self.bars):
            opens[i] = bar.open
            highs[i] = bar.high
            lows[i] = bar.low
            closes[i] = bar.close
        
        return opens, highs, lows, closes


@dataclass
class UnifiedSampler:
    """
    统一采样器
    
    封装 VolumeBarBuilder / TickImbalanceBarBuilder，提供一致的接口。
    
    使用方式:
    ```python
    config = SamplingConfig(mode=SamplingMode.VOLUME_BARS, target_volume=100)
    sampler = UnifiedSampler(config)
    
    # 处理单个 Tick
    completed_bar = sampler.add_tick(tick)
    if completed_bar:
        # Bar 完成，可以进行特征计算
        pass
    
    # 批量处理
    stream = sampler.process_ticks(ticks)
    bars = stream.bars
    ```
    """
    config: SamplingConfig = field(default_factory=SamplingConfig)
    on_bar_complete: Callable[[EventBar], None] | None = None
    
    # 内部构建器
    _builder: VolumeBarBuilder | TickImbalanceBarBuilder | None = field(
        default=None, init=False
    )
    _total_ticks: int = field(default=0, init=False)
    _total_bars: int = field(default=0, init=False)
    
    def __post_init__(self) -> None:
        """初始化内部构建器"""
        self._init_builder()
    
    def _init_builder(self) -> None:
        """根据配置初始化构建器"""
        if self.config.mode == SamplingMode.VOLUME_BARS:
            self._builder = VolumeBarBuilder(
                target_volume=self.config.target_volume,
                volume_source=self.config.volume_source.value,
                synthetic_base_volume=self.config.synthetic_base_volume,
                on_bar_complete=self._on_bar,
                max_buffer_size=self.config.max_buffer_size,
            )
        elif self.config.mode == SamplingMode.TICK_IMBALANCE:
            self._builder = TickImbalanceBarBuilder(
                initial_expected_ticks=self.config.initial_expected_ticks,
                initial_expected_imbalance=self.config.initial_expected_imbalance,
                ewma_alpha=self.config.ewma_alpha,
                tick_rule_gamma=self.config.tick_rule_gamma,
                tick_rule_threshold=self.config.tick_rule_threshold,
                on_bar_complete=self._on_bar,
                max_buffer_size=self.config.max_buffer_size,
            )
        else:
            raise ValueError(f"Unsupported sampling mode: {self.config.mode}")
        
        logger.info(
            "UnifiedSampler initialized",
            mode=self.config.mode.value,
            volume_source=self.config.volume_source.value,
            target_volume=self.config.target_volume,
        )
    
    def _on_bar(self, bar: EventBar) -> None:
        """Bar 完成回调"""
        self._total_bars += 1
        if self.on_bar_complete:
            self.on_bar_complete(bar)
    
    def add_tick(self, tick: Tick) -> EventBar | None:
        """
        添加单个 Tick
        
        Args:
            tick: Tick 数据
            
        Returns:
            如果 Bar 完成则返回 EventBar，否则返回 None
        """
        self._total_ticks += 1
        return self._builder.add_tick(tick)
    
    def process_ticks(self, ticks: Sequence[Tick]) -> EventBarStream:
        """
        批量处理 Ticks（自动选择最优实现）
        
        Args:
            ticks: Tick 序列
            
        Returns:
            EventBarStream 包含所有完成的 Bars
        """
        # 对于大量数据，使用向量化实现
        if len(ticks) > 10000 and self.config.mode == SamplingMode.VOLUME_BARS:
            return self._process_ticks_vectorized(ticks)
        
        # 小量数据或 Tick Imbalance 模式使用原始实现
        return self._process_ticks_sequential(ticks)
    
    def _process_ticks_sequential(self, ticks: Sequence[Tick]) -> EventBarStream:
        """顺序处理 Ticks（原始实现）"""
        completed_bars = []
        
        for tick in ticks:
            bar = self.add_tick(tick)
            if bar is not None:
                completed_bars.append(bar)
        
        return EventBarStream(
            bars=completed_bars,
            config=self.config,
            total_ticks=len(ticks),
        )
    
    def _process_ticks_vectorized(self, ticks: Sequence[Tick]) -> EventBarStream:
        """
        向量化批量处理 Ticks（针对 Volume Bars 优化）
        
        使用 numpy 向量化操作，比 Python 循环快 10x+
        """
        from datetime import datetime, timezone
        
        n_ticks = len(ticks)
        target = int(self.config.target_volume)
        
        # 提取数组（避免重复属性访问）
        timestamps_us = np.array([t.timestamp_us for t in ticks], dtype=np.int64)
        bids = np.array([t.bid for t in ticks], dtype=np.float64)
        asks = np.array([t.ask for t in ticks], dtype=np.float64)
        mids = (bids + asks) / 2
        spreads = asks - bids
        
        # 计算 bar 边界索引（每 target 个 tick 一个 bar）
        bar_indices = np.arange(target, n_ticks + 1, target)
        n_bars = len(bar_indices)
        
        if n_bars == 0:
            return EventBarStream(
                bars=[],
                config=self.config,
                total_ticks=n_ticks,
            )
        
        # 向量化计算每个 bar 的 OHLC
        completed_bars = []
        start_idx = 0
        
        for end_idx in bar_indices:
            # 提取 bar 内的数据
            bar_mids = mids[start_idx:end_idx]
            bar_spreads = spreads[start_idx:end_idx]
            bar_timestamps = timestamps_us[start_idx:end_idx]
            
            # 计算 OHLC
            open_price = bar_mids[0]
            high_price = np.max(bar_mids)
            low_price = np.min(bar_mids)
            close_price = bar_mids[-1]
            
            # 计算方向（简化：比较 open 和 close）
            tick_directions = np.sign(np.diff(bar_mids, prepend=bar_mids[0]))
            buy_count = int(np.sum(tick_directions > 0))
            sell_count = int(np.sum(tick_directions < 0))
            imbalance = (buy_count - sell_count) / max(1, buy_count + sell_count)
            
            # 创建 EventBar
            bar_time = datetime.fromtimestamp(
                bar_timestamps[0] / 1_000_000, tz=timezone.utc
            )
            close_time = datetime.fromtimestamp(
                bar_timestamps[-1] / 1_000_000, tz=timezone.utc
            )
            duration_ms = int((bar_timestamps[-1] - bar_timestamps[0]) / 1000)
            
            bar = EventBar(
                time=bar_time,
                close_time=close_time,
                open=float(open_price),
                high=float(high_price),
                low=float(low_price),
                close=float(close_price),
                tick_count=end_idx - start_idx,
                imbalance=float(imbalance),
                buy_count=buy_count,
                sell_count=sell_count,
                spread_sum=float(np.sum(bar_spreads)),
                duration_ms=duration_ms,
            )
            completed_bars.append(bar)
            start_idx = end_idx
        
        # 更新统计
        self._total_ticks += n_ticks
        self._total_bars += n_bars
        
        logger.info(f"Generated {n_bars} bars from {n_ticks} ticks (vectorized)")
        
        return EventBarStream(
            bars=completed_bars,
            config=self.config,
            total_ticks=n_ticks,
        )
    
    def iter_ticks(self, ticks: Iterator[Tick]) -> Iterator[EventBar]:
        """
        迭代处理 Ticks（生成器模式）
        
        Args:
            ticks: Tick 迭代器
            
        Yields:
            完成的 EventBar
        """
        for tick in ticks:
            bar = self.add_tick(tick)
            if bar is not None:
                yield bar
    
    @property
    def bars(self) -> list[EventBar]:
        """获取缓冲的 Bars"""
        return self._builder.bars
    
    @property
    def current_bar(self) -> EventBar | None:
        """获取当前未完成的 Bar"""
        return self._builder.current_bar
    
    @property
    def progress(self) -> float:
        """
        当前 Bar 的完成进度（0-1）
        
        对于 Volume Bars：current_volume / target_volume
        对于 Tick Imbalance Bars：|imbalance| / threshold
        """
        if isinstance(self._builder, VolumeBarBuilder):
            return self._builder.progress
        elif isinstance(self._builder, TickImbalanceBarBuilder):
            threshold = self._builder.trigger_threshold
            if threshold > 0:
                return min(1.0, abs(self._builder.current_imbalance) / threshold)
            return 0.0
        return 0.0
    
    def get_recent_bars(self, count: int) -> list[EventBar]:
        """
        获取最近 N 个完成的 Bars
        
        Args:
            count: 数量
            
        Returns:
            Bar 列表（最老在前）
        """
        return self._builder.get_recent_bars(count)
    
    def get_stats(self) -> dict:
        """获取采样器统计信息"""
        base_stats = self._builder.get_stats()
        return {
            "mode": self.config.mode.value,
            "volume_source": self.config.volume_source.value,
            "total_ticks": self._total_ticks,
            "total_bars": self._total_bars,
            "progress": round(self.progress, 3),
            **base_stats,
        }
    
    def reset(self) -> None:
        """重置采样器"""
        self._builder.reset()
        self._total_ticks = 0
        self._total_bars = 0
    
    def initialize_from_ticks(self, ticks: Sequence[Tick]) -> None:
        """
        从历史 Ticks 初始化
        
        Args:
            ticks: 历史 Tick 序列（最老在前）
        """
        logger.info(f"Initializing sampler with {len(ticks)} ticks")
        self._builder.initialize_from_ticks(ticks)
        self._total_ticks = len(ticks)
        self._total_bars = len(self._builder.bars)
        logger.info(
            "Sampler initialized",
            total_bars=self._total_bars,
            **self.get_stats(),
        )


def create_sampler(
    mode: str = "volume_bars",
    volume_source: str = "tick_count",
    target_volume: float = 100.0,
    **kwargs,
) -> UnifiedSampler:
    """
    工厂函数：创建采样器
    
    Args:
        mode: 采样模式 ("volume_bars", "tick_imbalance")
        volume_source: 成交量来源 ("real", "tick_count", "synthetic")
        target_volume: 目标成交量（Volume Bars）或初始期望 Ticks
        **kwargs: 其他配置参数
        
    Returns:
        配置好的 UnifiedSampler
    """
    config = SamplingConfig(
        mode=SamplingMode(mode),
        volume_source=VolumeSource(volume_source),
        target_volume=target_volume,
        **kwargs,
    )
    return UnifiedSampler(config)
