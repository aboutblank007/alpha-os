"""
多 Horizon 标签生成器

为 ML 模型生成多个时间 horizon 的标签：
- 5 tick horizon
- 10 tick horizon
- 20 tick horizon

每个 horizon 生成三类标签：
- Long: 收益 > threshold
- Short: 收益 < -threshold
- Neutral: 其他

支持每个 bar 生成标签，而非仅在 Primary 信号处。

标签结构：
- label_5_long: 5 tick 后上涨 > threshold
- label_5_short: 5 tick 后下跌 > threshold
- label_10_long/short
- label_20_long/short
- label_direction: 综合方向 (-1/0/1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from alphaos.core.logging import get_logger
from alphaos.data.event_bars.tick_imbalance import EventBar
from alphaos.v4.types import Bar

logger = get_logger(__name__)


@dataclass
class MultiHorizonLabelsResult:
    """
    多 Horizon 标签结果
    
    Attributes:
        label_5_long: 5 tick 做多标签 (0/1)
        label_5_short: 5 tick 做空标签 (0/1)
        label_10_long: 10 tick 做多标签
        label_10_short: 10 tick 做空标签
        label_20_long: 20 tick 做多标签
        label_20_short: 20 tick 做空标签
        label_direction: 综合方向 (-1/0/1)
        return_5: 5 tick 收益 (bps)
        return_10: 10 tick 收益 (bps)
        return_20: 20 tick 收益 (bps)
    """
    label_5_long: int = 0
    label_5_short: int = 0
    label_10_long: int = 0
    label_10_short: int = 0
    label_20_long: int = 0
    label_20_short: int = 0
    label_direction: int = 0
    return_5: float = 0.0
    return_10: float = 0.0
    return_20: float = 0.0
    
    def to_array(self) -> NDArray[np.float32]:
        """转换为 numpy 数组"""
        return np.array([
            self.label_5_long,
            self.label_5_short,
            self.label_10_long,
            self.label_10_short,
            self.label_20_long,
            self.label_20_short,
            self.label_direction,
            self.return_5,
            self.return_10,
            self.return_20,
        ], dtype=np.float32)
    
    @staticmethod
    def label_names() -> list[str]:
        """标签名称列表"""
        return [
            "label_5_long",
            "label_5_short",
            "label_10_long",
            "label_10_short",
            "label_20_long",
            "label_20_short",
            "label_direction",
            "return_5",
            "return_10",
            "return_20",
        ]
    
    @staticmethod
    def n_labels() -> int:
        """标签数量"""
        return 10


@dataclass
class MultiHorizonConfig:
    """
    多 Horizon 标签配置
    
    Args:
        horizons: 预测 horizon 列表（bar 数）
        threshold_bps: 分类阈值（基点）
        use_log_returns: 是否使用对数收益率
    """
    horizons: list[int] = field(default_factory=lambda: [5, 10, 20])
    threshold_bps: float = 3.0  # 3 bps = 0.03%
    use_log_returns: bool = True


@dataclass
class MultiHorizonLabeler:
    """
    多 Horizon 标签生成器
    
    为每个 bar 生成多个时间 horizon 的标签。
    
    关键设计：
    - 不依赖 Primary 信号（每个 bar 都有标签）
    - 多 horizon 让模型学习最优预测窗口
    - 同时输出原始收益值（用于回归任务）
    
    使用方式：
    ```python
    labeler = MultiHorizonLabeler(config)
    
    # 批量生成
    labels = labeler.compute_batch(bars)  # (n_bars, 10)
    
    # 提取特定 horizon 的标签
    labels_10 = labels[:, 2:4]  # label_10_long, label_10_short
    ```
    """
    config: MultiHorizonConfig = field(default_factory=MultiHorizonConfig)
    
    def __post_init__(self) -> None:
        """初始化"""
        logger.info(
            "MultiHorizonLabeler initialized",
            horizons=self.config.horizons,
            threshold_bps=self.config.threshold_bps,
        )
    
    def compute_batch(
        self,
        bars: Sequence[EventBar] | Sequence[Bar],
    ) -> NDArray[np.float32]:
        """
        批量计算多 Horizon 标签
        
        Args:
            bars: Bar 序列
            
        Returns:
            标签矩阵 (n_bars, 10)
        """
        n_bars = len(bars)
        labels = np.zeros((n_bars, MultiHorizonLabelsResult.n_labels()), dtype=np.float32)
        
        # 提取收盘价
        closes = np.array([b.close for b in bars], dtype=np.float64)
        
        # 计算各 horizon 的收益
        max_horizon = max(self.config.horizons)
        
        for i in range(n_bars):
            result = self._compute_labels_for_bar(closes, i, n_bars)
            labels[i] = result.to_array()
        
        # 统计
        label_counts = {
            "horizon_5": {
                "long": int(np.sum(labels[:, 0])),
                "short": int(np.sum(labels[:, 1])),
            },
            "horizon_10": {
                "long": int(np.sum(labels[:, 2])),
                "short": int(np.sum(labels[:, 3])),
            },
            "horizon_20": {
                "long": int(np.sum(labels[:, 4])),
                "short": int(np.sum(labels[:, 5])),
            },
        }
        
        logger.info(
            "Multi-horizon labels computed",
            n_bars=n_bars,
            label_counts=label_counts,
        )
        
        return labels
    
    def _compute_labels_for_bar(
        self,
        closes: NDArray[np.float64],
        bar_idx: int,
        n_bars: int,
    ) -> MultiHorizonLabelsResult:
        """计算单个 bar 的标签"""
        result = MultiHorizonLabelsResult()
        threshold = self.config.threshold_bps / 10000  # 转换为小数
        
        current_price = closes[bar_idx]
        
        # 计算各 horizon 收益
        for horizon in self.config.horizons:
            future_idx = bar_idx + horizon
            
            if future_idx >= n_bars:
                # 未来数据不足，标签为 0
                continue
            
            future_price = closes[future_idx]
            
            # 计算收益
            if self.config.use_log_returns:
                ret = np.log(future_price / current_price)
            else:
                ret = (future_price - current_price) / current_price
            
            ret_bps = ret * 10000
            
            # 设置标签
            if horizon == 5:
                result.return_5 = ret_bps
                result.label_5_long = 1 if ret > threshold else 0
                result.label_5_short = 1 if ret < -threshold else 0
            elif horizon == 10:
                result.return_10 = ret_bps
                result.label_10_long = 1 if ret > threshold else 0
                result.label_10_short = 1 if ret < -threshold else 0
            elif horizon == 20:
                result.return_20 = ret_bps
                result.label_20_long = 1 if ret > threshold else 0
                result.label_20_short = 1 if ret < -threshold else 0
        
        # 计算综合方向（基于 10 tick horizon）
        if result.label_10_long:
            result.label_direction = 1
        elif result.label_10_short:
            result.label_direction = -1
        else:
            result.label_direction = 0
        
        return result
    
    def get_label_columns(self, horizon: int) -> tuple[int, int]:
        """
        获取特定 horizon 的标签列索引
        
        Args:
            horizon: 5, 10, 或 20
            
        Returns:
            (long_idx, short_idx) 元组
        """
        if horizon == 5:
            return (0, 1)
        elif horizon == 10:
            return (2, 3)
        elif horizon == 20:
            return (4, 5)
        else:
            raise ValueError(f"Unsupported horizon: {horizon}")
    
    def get_stats(self) -> dict:
        """获取配置统计"""
        return {
            "horizons": self.config.horizons,
            "threshold_bps": self.config.threshold_bps,
            "use_log_returns": self.config.use_log_returns,
        }


def compute_simple_labels(
    bars: Sequence[EventBar] | Sequence[Bar],
    horizon: int = 10,
    threshold_bps: float = 3.0,
) -> NDArray[np.int32]:
    """
    简化版标签计算（单 horizon，三分类）
    
    Args:
        bars: Bar 序列
        horizon: 预测 horizon (bar 数)
        threshold_bps: 分类阈值 (bps)
        
    Returns:
        标签数组 (n_bars,)，值为 -1/0/1
    """
    n_bars = len(bars)
    labels = np.zeros(n_bars, dtype=np.int32)
    
    closes = np.array([b.close for b in bars], dtype=np.float64)
    threshold = threshold_bps / 10000
    
    for i in range(n_bars):
        future_idx = i + horizon
        if future_idx >= n_bars:
            continue
        
        ret = np.log(closes[future_idx] / closes[i])
        
        if ret > threshold:
            labels[i] = 1   # Long
        elif ret < -threshold:
            labels[i] = -1  # Short
        else:
            labels[i] = 0   # Neutral
    
    return labels
