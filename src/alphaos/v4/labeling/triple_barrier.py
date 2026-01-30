"""
Triple Barrier Method 标签生成

三栏栅法（López de Prado）：
- 上栏栅（止盈）：价格上涨 sigma * M_up
- 下栏栅（止损）：价格下跌 sigma * M_down
- 垂直栏栅（时间）：最大持仓 Bar 数

标签生成规则：
- 触及上栏栅 → Label = 1（做多盈利）
- 触及下栏栅 → Label = -1（做空盈利/做多亏损）
- 触及垂直栏栅 → Label = 0（无显著波动）

参考：降噪LNN特征提取与信号过滤.md Section 5.1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from alphaos.core.logging import get_logger
from alphaos.v4.schemas import LabelSchema

logger = get_logger(__name__)

# 尝试导入 Numba（可选优化）
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # 定义 no-op 装饰器
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


class BarrierType(Enum):
    """栏栅类型"""
    UPPER = 1      # 上栏栅（止盈）
    LOWER = -1     # 下栏栅（止损）
    VERTICAL = 0   # 垂直栏栅（时间）


@dataclass
class BarrierEvent:
    """
    单个栏栅事件
    
    Attributes:
        bar_idx: 入场 Bar 索引
        entry_price: 入场价格
        upper_barrier: 上栏栅价格
        lower_barrier: 下栏栅价格
        vertical_barrier: 垂直栏栅 Bar 索引
        exit_bar_idx: 出场 Bar 索引
        exit_price: 出场价格
        barrier_hit: 触碰的栏栅类型
        return_pct: 收益率（百分比）
        holding_bars: 持仓 Bar 数
        primary_direction: Primary 信号方向
    """
    bar_idx: int
    entry_price: float
    upper_barrier: float
    lower_barrier: float
    vertical_barrier: int
    exit_bar_idx: int = -1
    exit_price: float = 0.0
    barrier_hit: BarrierType = BarrierType.VERTICAL
    return_pct: float = 0.0
    holding_bars: int = 0
    primary_direction: int = 0
    
    @property
    def is_success(self) -> bool:
        """交易是否成功（与 primary 方向一致）"""
        if self.primary_direction == 0:
            return False
        if self.primary_direction == 1:  # 做多
            return self.barrier_hit == BarrierType.UPPER
        else:  # 做空
            return self.barrier_hit == BarrierType.LOWER
    
    def to_dict(self) -> dict:
        return {
            "bar_idx": self.bar_idx,
            "entry_price": round(self.entry_price, 2),
            "upper_barrier": round(self.upper_barrier, 2),
            "lower_barrier": round(self.lower_barrier, 2),
            "vertical_barrier": self.vertical_barrier,
            "exit_bar_idx": self.exit_bar_idx,
            "exit_price": round(self.exit_price, 2),
            "barrier_hit": self.barrier_hit.name,
            "return_pct": round(self.return_pct, 4),
            "holding_bars": self.holding_bars,
            "primary_direction": self.primary_direction,
            "is_success": self.is_success,
        }


@dataclass
class TripleBarrierConfig:
    """
    Triple Barrier 配置
    
    Args:
        # 栏栅参数（波动率倍数）
        upper_multiplier: 上栏栅倍数（止盈）
        lower_multiplier: 下栏栅倍数（止损）
        vertical_bars: 垂直栏栅（最大持仓 Bar 数）
        
        # 波动率计算
        volatility_window: 波动率计算窗口
        volatility_type: 波动率类型 ("realized", "ewma")
        ewma_lambda: EWMA 衰减系数（如果 type="ewma"）
        
        # 最小阈值（防止过小）
        min_barrier_pct: 最小栏栅宽度（百分比）
        
        # 其他
        use_log_returns: 是否使用对数收益率
    """
    upper_multiplier: float = 2.0
    lower_multiplier: float = 2.0
    vertical_bars: int = 20
    
    volatility_window: int = 20
    volatility_type: str = "realized"
    ewma_lambda: float = 0.94
    
    min_barrier_pct: float = 0.1  # 0.1%
    
    use_log_returns: bool = True
    
    def to_dict(self) -> dict:
        return {
            "upper_multiplier": self.upper_multiplier,
            "lower_multiplier": self.lower_multiplier,
            "vertical_bars": self.vertical_bars,
            "volatility_window": self.volatility_window,
            "volatility_type": self.volatility_type,
            "ewma_lambda": self.ewma_lambda,
            "min_barrier_pct": self.min_barrier_pct,
            "use_log_returns": self.use_log_returns,
        }


# Numba 优化的核心计算函数
@jit(nopython=True, parallel=True, cache=True)
def _compute_barriers_numba(
    closes: np.ndarray,
    volatilities: np.ndarray,
    event_indices: np.ndarray,
    primary_directions: np.ndarray,
    upper_mult: float,
    lower_mult: float,
    vertical_bars: int,
    min_barrier_pct: float,
) -> tuple:
    """
    Numba 优化的栏栅计算
    
    Returns:
        (exit_indices, exit_prices, barrier_hits, returns_pct, holding_bars)
    """
    n_events = len(event_indices)
    n_bars = len(closes)
    
    exit_indices = np.zeros(n_events, dtype=np.int64)
    exit_prices = np.zeros(n_events, dtype=np.float64)
    barrier_hits = np.zeros(n_events, dtype=np.int32)
    returns_pct = np.zeros(n_events, dtype=np.float64)
    holding_bars_out = np.zeros(n_events, dtype=np.int32)
    
    for i in prange(n_events):
        entry_idx = event_indices[i]
        entry_price = closes[entry_idx]
        sigma = volatilities[entry_idx]
        direction = primary_directions[i]
        
        # 计算栏栅
        barrier_pct = max(sigma * 100, min_barrier_pct)  # 百分比
        upper_barrier = entry_price * (1 + barrier_pct * upper_mult / 100)
        lower_barrier = entry_price * (1 - barrier_pct * lower_mult / 100)
        vertical_idx = min(entry_idx + vertical_bars, n_bars - 1)
        
        # 搜索触碰栏栅
        exit_idx = vertical_idx
        exit_price_ = closes[vertical_idx]
        hit = 0  # VERTICAL
        
        for j in range(entry_idx + 1, vertical_idx + 1):
            price = closes[j]
            
            # 检查上栏栅
            if price >= upper_barrier:
                exit_idx = j
                exit_price_ = price
                hit = 1  # UPPER
                break
            
            # 检查下栏栅
            if price <= lower_barrier:
                exit_idx = j
                exit_price_ = price
                hit = -1  # LOWER
                break
        
        # 计算收益
        if entry_price > 0:
            ret_pct = (exit_price_ - entry_price) / entry_price * 100
        else:
            ret_pct = 0.0
        
        exit_indices[i] = exit_idx
        exit_prices[i] = exit_price_
        barrier_hits[i] = hit
        returns_pct[i] = ret_pct
        holding_bars_out[i] = exit_idx - entry_idx
    
    return exit_indices, exit_prices, barrier_hits, returns_pct, holding_bars_out


@dataclass
class TripleBarrierLabeler:
    """
    Triple Barrier 标签生成器
    
    使用方式：
    ```python
    labeler = TripleBarrierLabeler(config)
    
    # 批量生成标签
    events = labeler.generate_labels(
        closes=closes,
        event_indices=signal_indices,
        primary_directions=signal_directions,
    )
    
    # 转换为训练数据
    labels = labeler.events_to_labels(events)
    ```
    """
    config: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)
    
    def generate_labels(
        self,
        closes: NDArray[np.float64],
        event_indices: NDArray[np.int64],
        primary_directions: NDArray[np.int32],
        volatilities: NDArray[np.float64] | None = None,
    ) -> list[BarrierEvent]:
        """
        生成 Triple Barrier 标签
        
        Args:
            closes: 收盘价序列
            event_indices: Primary 信号触发的 Bar 索引
            primary_directions: Primary 信号方向 (1=LONG, -1=SHORT)
            volatilities: 波动率序列（可选，否则内部计算）
            
        Returns:
            BarrierEvent 列表
        """
        n_bars = len(closes)
        n_events = len(event_indices)
        
        if n_events == 0:
            return []
        
        # 计算波动率（如果未提供）
        if volatilities is None:
            volatilities = self._compute_volatility(closes)
        
        # 使用 Numba 优化（如果可用）
        if HAS_NUMBA and n_events > 10:
            exit_indices, exit_prices, barrier_hits, returns_pct, holding_bars = (
                _compute_barriers_numba(
                    closes.astype(np.float64),
                    volatilities.astype(np.float64),
                    event_indices.astype(np.int64),
                    primary_directions.astype(np.int32),
                    self.config.upper_multiplier,
                    self.config.lower_multiplier,
                    self.config.vertical_bars,
                    self.config.min_barrier_pct,
                )
            )
        else:
            # Python 实现
            exit_indices, exit_prices, barrier_hits, returns_pct, holding_bars = (
                self._compute_barriers_python(
                    closes, volatilities, event_indices, primary_directions
                )
            )
        
        # 构建 BarrierEvent 列表
        events = []
        for i in range(n_events):
            entry_idx = event_indices[i]
            entry_price = closes[entry_idx]
            sigma = volatilities[entry_idx]
            
            # 计算实际栏栅价格
            barrier_pct = max(sigma * 100, self.config.min_barrier_pct)
            upper_barrier = entry_price * (1 + barrier_pct * self.config.upper_multiplier / 100)
            lower_barrier = entry_price * (1 - barrier_pct * self.config.lower_multiplier / 100)
            
            event = BarrierEvent(
                bar_idx=int(entry_idx),
                entry_price=entry_price,
                upper_barrier=upper_barrier,
                lower_barrier=lower_barrier,
                vertical_barrier=min(int(entry_idx) + self.config.vertical_bars, n_bars - 1),
                exit_bar_idx=int(exit_indices[i]),
                exit_price=exit_prices[i],
                barrier_hit=BarrierType(barrier_hits[i]),
                return_pct=returns_pct[i],
                holding_bars=int(holding_bars[i]),
                primary_direction=int(primary_directions[i]),
            )
            events.append(event)
        
        logger.info(
            "Generated triple barrier labels",
            n_events=n_events,
            upper_hits=sum(1 for e in events if e.barrier_hit == BarrierType.UPPER),
            lower_hits=sum(1 for e in events if e.barrier_hit == BarrierType.LOWER),
            vertical_hits=sum(1 for e in events if e.barrier_hit == BarrierType.VERTICAL),
            success_rate=sum(1 for e in events if e.is_success) / max(n_events, 1),
        )
        
        return events
    
    def _compute_barriers_python(
        self,
        closes: NDArray[np.float64],
        volatilities: NDArray[np.float64],
        event_indices: NDArray[np.int64],
        primary_directions: NDArray[np.int32],
    ) -> tuple:
        """Python 实现的栏栅计算"""
        n_events = len(event_indices)
        n_bars = len(closes)
        
        exit_indices = np.zeros(n_events, dtype=np.int64)
        exit_prices = np.zeros(n_events, dtype=np.float64)
        barrier_hits = np.zeros(n_events, dtype=np.int32)
        returns_pct = np.zeros(n_events, dtype=np.float64)
        holding_bars = np.zeros(n_events, dtype=np.int32)
        
        for i in range(n_events):
            entry_idx = event_indices[i]
            entry_price = closes[entry_idx]
            sigma = volatilities[entry_idx]
            
            # 计算栏栅
            barrier_pct = max(sigma * 100, self.config.min_barrier_pct)
            upper_barrier = entry_price * (1 + barrier_pct * self.config.upper_multiplier / 100)
            lower_barrier = entry_price * (1 - barrier_pct * self.config.lower_multiplier / 100)
            vertical_idx = min(entry_idx + self.config.vertical_bars, n_bars - 1)
            
            # 搜索触碰栏栅
            exit_idx = vertical_idx
            exit_price = closes[vertical_idx]
            hit = 0  # VERTICAL
            
            for j in range(entry_idx + 1, vertical_idx + 1):
                price = closes[j]
                
                if price >= upper_barrier:
                    exit_idx = j
                    exit_price = price
                    hit = 1  # UPPER
                    break
                
                if price <= lower_barrier:
                    exit_idx = j
                    exit_price = price
                    hit = -1  # LOWER
                    break
            
            # 计算收益
            if entry_price > 0:
                ret = (exit_price - entry_price) / entry_price * 100
            else:
                ret = 0.0
            
            exit_indices[i] = exit_idx
            exit_prices[i] = exit_price
            barrier_hits[i] = hit
            returns_pct[i] = ret
            holding_bars[i] = exit_idx - entry_idx
        
        return exit_indices, exit_prices, barrier_hits, returns_pct, holding_bars
    
    def _compute_volatility(
        self,
        closes: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """计算波动率序列"""
        n = len(closes)
        
        # 计算收益率
        if self.config.use_log_returns:
            returns = np.zeros(n, dtype=np.float64)
            returns[1:] = np.log(closes[1:] / closes[:-1])
        else:
            returns = np.zeros(n, dtype=np.float64)
            returns[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
        
        volatility = np.zeros(n, dtype=np.float64)
        
        if self.config.volatility_type == "realized":
            # 已实现波动率（滚动标准差）
            window = self.config.volatility_window
            for i in range(window, n):
                volatility[i] = np.std(returns[i-window+1:i+1])
            # 填充初始值
            if window < n:
                volatility[:window] = volatility[window]
        
        elif self.config.volatility_type == "ewma":
            # EWMA 波动率
            lmbda = self.config.ewma_lambda
            variance = np.zeros(n, dtype=np.float64)
            variance[0] = returns[0] ** 2 if n > 0 else 1e-6
            
            for i in range(1, n):
                variance[i] = (1 - lmbda) * (returns[i] ** 2) + lmbda * variance[i-1]
            
            volatility = np.sqrt(variance)
        
        # 防止零值
        volatility = np.maximum(volatility, 1e-6)
        
        return volatility
    
    def events_to_labels(
        self,
        events: list[BarrierEvent],
        n_bars: int | None = None,
    ) -> NDArray:
        """
        将 BarrierEvent 列表转换为结构化标签数组
        
        Args:
            events: BarrierEvent 列表
            n_bars: 总 Bar 数（用于创建完整标签数组）
            
        Returns:
            结构化数组（按 LabelSchema 定义）
        """
        dtype = LabelSchema.get_dtype()
        labels = np.zeros(len(events), dtype=dtype)
        
        for i, event in enumerate(events):
            labels[i]["bar_idx"] = event.bar_idx
            labels[i]["primary_direction"] = event.primary_direction
            labels[i]["barrier_hit"] = event.barrier_hit.value
            labels[i]["return_pct"] = event.return_pct
            labels[i]["meta_label"] = 1 if event.is_success else 0
            labels[i]["holding_bars"] = event.holding_bars
            labels[i]["entry_price"] = event.entry_price
            labels[i]["exit_price"] = event.exit_price
            labels[i]["stop_loss"] = event.lower_barrier
            labels[i]["take_profit"] = event.upper_barrier
        
        return labels
    
    def get_meta_labels(
        self,
        events: list[BarrierEvent],
    ) -> NDArray[np.int32]:
        """
        提取元标签（用于训练 meta-model）
        
        Args:
            events: BarrierEvent 列表
            
        Returns:
            元标签数组 (0=失败, 1=成功)
        """
        return np.array([1 if e.is_success else 0 for e in events], dtype=np.int32)
    
    def get_label_distribution(
        self,
        events: list[BarrierEvent],
    ) -> dict:
        """获取标签分布统计"""
        n = len(events)
        if n == 0:
            return {"total": 0, "upper": 0, "lower": 0, "vertical": 0, "success_rate": 0.0}
        
        upper_hits = sum(1 for e in events if e.barrier_hit == BarrierType.UPPER)
        lower_hits = sum(1 for e in events if e.barrier_hit == BarrierType.LOWER)
        vertical_hits = sum(1 for e in events if e.barrier_hit == BarrierType.VERTICAL)
        successes = sum(1 for e in events if e.is_success)
        
        return {
            "total": n,
            "upper": upper_hits,
            "upper_pct": upper_hits / n * 100,
            "lower": lower_hits,
            "lower_pct": lower_hits / n * 100,
            "vertical": vertical_hits,
            "vertical_pct": vertical_hits / n * 100,
            "success": successes,
            "success_rate": successes / n * 100,
            "avg_return_pct": np.mean([e.return_pct for e in events]),
            "avg_holding_bars": np.mean([e.holding_bars for e in events]),
        }
