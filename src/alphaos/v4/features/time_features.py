"""
时间/Session 特征计算模块

计算与时间相关的 ML 特征：
- 交易 Session（亚洲/伦敦/纽约）
- 时间周期特征（小时、分钟、星期）
- Session 开盘后 bar 数

这些特征对于学习市场微观结构的时间模式非常重要。

特征列表：
- session: 交易 Session (0=亚洲, 1=伦敦, 2=纽约, 3=其他)
- hour_of_day: 小时 (0-23)
- minute_of_hour: 分钟 (0-59)
- day_of_week: 星期几 (0=周一, 6=周日)
- bars_from_session_open: Session 开盘后 bar 数
- is_session_open: 是否在 Session 开盘前 30 分钟内
- is_session_close: 是否在 Session 收盘前 30 分钟内
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta
from typing import Sequence
from enum import IntEnum
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray

from alphaos.core.logging import get_logger
from alphaos.data.event_bars.tick_imbalance import EventBar
from alphaos.v4.types import Bar

logger = get_logger(__name__)


class TradingSession(IntEnum):
    """交易 Session 枚举"""
    ASIA = 0       # 亚洲盘: 00:00-08:00 UTC (Tokyo/Sydney)
    LONDON = 1     # 伦敦盘: 08:00-16:00 UTC
    NEWYORK = 2    # 纽约盘: 13:00-21:00 UTC
    OVERLAP = 3    # 重叠时段
    CLOSED = 4     # 收盘时段


# Session 时间定义 (UTC)
SESSION_TIMES = {
    TradingSession.ASIA: (dt_time(0, 0), dt_time(8, 0)),      # 00:00-08:00 UTC
    TradingSession.LONDON: (dt_time(8, 0), dt_time(16, 0)),   # 08:00-16:00 UTC
    TradingSession.NEWYORK: (dt_time(13, 0), dt_time(21, 0)), # 13:00-21:00 UTC
}


@dataclass
class TimeFeaturesResult:
    """
    时间特征计算结果
    
    Attributes:
        session: 交易 Session (0-4)
        hour_of_day: 小时 (0-23)
        minute_of_hour: 分钟 (0-59)
        day_of_week: 星期几 (0-6)
        bars_from_session_open: Session 开盘后 bar 数
        is_session_open: 是否在 Session 开盘期
        is_session_close: 是否在 Session 收盘期
        hour_sin: 小时正弦编码
        hour_cos: 小时余弦编码
    """
    session: int = 0
    hour_of_day: int = 0
    minute_of_hour: int = 0
    day_of_week: int = 0
    bars_from_session_open: int = 0
    is_session_open: int = 0
    is_session_close: int = 0
    hour_sin: float = 0.0
    hour_cos: float = 0.0
    
    def to_array(self) -> NDArray[np.float32]:
        """转换为 numpy 数组"""
        return np.array([
            self.session,
            self.hour_of_day,
            self.minute_of_hour,
            self.day_of_week,
            self.bars_from_session_open,
            self.is_session_open,
            self.is_session_close,
            self.hour_sin,
            self.hour_cos,
        ], dtype=np.float32)
    
    @staticmethod
    def feature_names() -> list[str]:
        """特征名称列表"""
        return [
            "session",
            "hour_of_day",
            "minute_of_hour",
            "day_of_week",
            "bars_from_session_open",
            "is_session_open",
            "is_session_close",
            "hour_sin",
            "hour_cos",
        ]
    
    @staticmethod
    def n_features() -> int:
        """特征数量"""
        return 9


@dataclass
class TimeFeatureConfig:
    """
    时间特征计算配置
    
    Args:
        timezone: 时区（用于计算本地时间）
        session_open_window_minutes: Session 开盘窗口（分钟）
        session_close_window_minutes: Session 收盘窗口（分钟）
    """
    timezone: str = "UTC"
    session_open_window_minutes: int = 30
    session_close_window_minutes: int = 30


@dataclass
class TimeFeatureCalculator:
    """
    时间特征计算器
    
    计算与时间相关的 ML 特征。
    
    关键设计：
    - Session 分类（亚洲/伦敦/纽约）
    - 周期性时间特征（正弦/余弦编码）
    - Session 边界特征（开盘/收盘）
    
    使用方式：
    ```python
    calc = TimeFeatureCalculator(config)
    
    # 批量计算
    features = calc.compute_batch(bars)  # (n_bars, 9)
    
    # 单次计算
    feat = calc.compute(timestamp)
    ```
    """
    config: TimeFeatureConfig = field(default_factory=TimeFeatureConfig)
    
    # Session 状态跟踪
    _current_session: TradingSession = field(default=TradingSession.CLOSED, init=False)
    _session_start_bar: int = field(default=0, init=False)
    _bar_count: int = field(default=0, init=False)
    _prev_session: TradingSession = field(default=TradingSession.CLOSED, init=False)
    
    def __post_init__(self) -> None:
        """初始化"""
        logger.info(
            "TimeFeatureCalculator initialized",
            timezone=self.config.timezone,
        )
    
    def reset(self) -> None:
        """重置状态"""
        self._current_session = TradingSession.CLOSED
        self._session_start_bar = 0
        self._bar_count = 0
        self._prev_session = TradingSession.CLOSED
    
    def compute_batch(
        self,
        bars: Sequence[EventBar] | Sequence[Bar],
    ) -> NDArray[np.float32]:
        """
        批量计算时间特征
        
        Args:
            bars: Bar 序列
            
        Returns:
            特征矩阵 (n_bars, 9)
        """
        self.reset()
        
        n_bars = len(bars)
        features = np.zeros((n_bars, TimeFeaturesResult.n_features()), dtype=np.float32)
        
        for i, bar in enumerate(bars):
            self._bar_count = i
            result = self._compute_features(bar.time)
            features[i] = result.to_array()
        
        logger.info(
            "Time features computed (batch)",
            n_bars=n_bars,
        )
        
        return features
    
    def compute(self, timestamp: datetime) -> TimeFeaturesResult:
        """
        计算单个时间戳的特征
        
        Args:
            timestamp: 时间戳
            
        Returns:
            时间特征结果
        """
        self._bar_count += 1
        return self._compute_features(timestamp)
    
    def _compute_features(self, timestamp: datetime) -> TimeFeaturesResult:
        """计算时间特征"""
        result = TimeFeaturesResult()
        
        # 确保时间戳是 UTC
        if timestamp.tzinfo is None:
            utc_time = timestamp
        else:
            utc_time = timestamp.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        
        # 基础时间特征
        result.hour_of_day = utc_time.hour
        result.minute_of_hour = utc_time.minute
        result.day_of_week = utc_time.weekday()
        
        # 周期性编码（24小时周期）
        hour_frac = (utc_time.hour + utc_time.minute / 60) / 24
        result.hour_sin = np.sin(2 * np.pi * hour_frac)
        result.hour_cos = np.cos(2 * np.pi * hour_frac)
        
        # Session 分类
        session = self._get_session(utc_time.time())
        result.session = int(session)
        
        # 检测 Session 变化
        if session != self._prev_session and session != TradingSession.CLOSED:
            self._session_start_bar = self._bar_count
            self._current_session = session
        
        self._prev_session = session
        
        # Session 开盘后 bar 数
        result.bars_from_session_open = self._bar_count - self._session_start_bar
        
        # Session 开盘/收盘窗口
        result.is_session_open = int(self._is_near_session_open(utc_time))
        result.is_session_close = int(self._is_near_session_close(utc_time))
        
        return result
    
    def _get_session(self, t: dt_time) -> TradingSession:
        """获取当前时间的 Session"""
        # 检查纽约盘（最高优先级）
        ny_start, ny_end = SESSION_TIMES[TradingSession.NEWYORK]
        if ny_start <= t < ny_end:
            # 检查是否与伦敦重叠 (13:00-16:00)
            london_start, london_end = SESSION_TIMES[TradingSession.LONDON]
            if london_start <= t < london_end:
                return TradingSession.OVERLAP
            return TradingSession.NEWYORK
        
        # 检查伦敦盘
        london_start, london_end = SESSION_TIMES[TradingSession.LONDON]
        if london_start <= t < london_end:
            return TradingSession.LONDON
        
        # 检查亚洲盘
        asia_start, asia_end = SESSION_TIMES[TradingSession.ASIA]
        if asia_start <= t < asia_end:
            return TradingSession.ASIA
        
        # 周末或其他时段
        return TradingSession.CLOSED
    
    def _is_near_session_open(self, dt: datetime) -> bool:
        """是否接近 Session 开盘"""
        t = dt.time()
        window = timedelta(minutes=self.config.session_open_window_minutes)
        
        for session, (start, end) in SESSION_TIMES.items():
            start_dt = datetime.combine(dt.date(), start)
            if start_dt <= dt < start_dt + window:
                return True
        
        return False
    
    def _is_near_session_close(self, dt: datetime) -> bool:
        """是否接近 Session 收盘"""
        t = dt.time()
        window = timedelta(minutes=self.config.session_close_window_minutes)
        
        for session, (start, end) in SESSION_TIMES.items():
            end_dt = datetime.combine(dt.date(), end)
            if end_dt - window <= dt < end_dt:
                return True
        
        return False
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "bar_count": self._bar_count,
            "current_session": self._current_session.name,
            "bars_in_session": self._bar_count - self._session_start_bar,
        }


def get_session_name(session: int) -> str:
    """获取 Session 名称"""
    names = {
        0: "Asia",
        1: "London",
        2: "NewYork",
        3: "Overlap",
        4: "Closed",
    }
    return names.get(session, "Unknown")
