"""
Meta-Labeling 生成器

元标签技术（López de Prado）：
1. Primary Model: 生成交易方向信号（LONG/SHORT）
2. Meta Model: 预测 Primary 信号是否正确（P(success)）

Meta Model 输入：
- 微观结构特征（VPIN, OFI, Kyle Lambda 等）
- Primary 信号方向
- 市场状态（温度、熵、相位）

Meta Model 输出：
- 交易成功概率 P(success | features, direction)

参考：降噪LNN特征提取与信号过滤.md Section 5.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from alphaos.core.logging import get_logger
from alphaos.v4.labeling.triple_barrier import (
    TripleBarrierLabeler,
    TripleBarrierConfig,
    BarrierEvent,
)

logger = get_logger(__name__)


@dataclass
class MetaLabelConfig:
    """
    Meta-Labeling 配置
    
    Args:
        triple_barrier_config: Triple Barrier 配置
        min_signals: 最小信号数量（用于统计可靠性）
        sample_weight_method: 样本权重方法 ("uniform", "return_based", "time_decay")
        time_decay_half_life: 时间衰减半衰期（Bar 数）
    """
    triple_barrier_config: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)
    min_signals: int = 10
    sample_weight_method: str = "return_based"
    time_decay_half_life: int = 100
    
    def to_dict(self) -> dict:
        return {
            "triple_barrier_config": self.triple_barrier_config.to_dict(),
            "min_signals": self.min_signals,
            "sample_weight_method": self.sample_weight_method,
            "time_decay_half_life": self.time_decay_half_life,
        }


@dataclass
class MetaLabelResult:
    """
    Meta-Labeling 结果
    
    Attributes:
        event_indices: Primary 信号 Bar 索引
        primary_directions: Primary 信号方向
        meta_labels: 元标签 (0=失败, 1=成功)
        sample_weights: 样本权重
        returns_pct: 收益率（百分比）
        holding_bars: 持仓 Bar 数
        events: 完整的 BarrierEvent 列表
    """
    event_indices: NDArray[np.int64]
    primary_directions: NDArray[np.int32]
    meta_labels: NDArray[np.int32]
    sample_weights: NDArray[np.float32]
    returns_pct: NDArray[np.float32]
    holding_bars: NDArray[np.int32]
    events: list[BarrierEvent] = field(default_factory=list)
    
    @property
    def n_samples(self) -> int:
        return len(self.meta_labels)
    
    @property
    def positive_rate(self) -> float:
        if self.n_samples == 0:
            return 0.0
        return np.mean(self.meta_labels)
    
    def get_distribution(self) -> dict:
        """获取分布统计"""
        if self.n_samples == 0:
            return {"n_samples": 0}
        
        return {
            "n_samples": self.n_samples,
            "positive": int(np.sum(self.meta_labels)),
            "positive_rate": float(self.positive_rate),
            "long_signals": int(np.sum(self.primary_directions == 1)),
            "short_signals": int(np.sum(self.primary_directions == -1)),
            "avg_return_pct": float(np.mean(self.returns_pct)),
            "avg_holding_bars": float(np.mean(self.holding_bars)),
            "total_weight": float(np.sum(self.sample_weights)),
        }


@dataclass
class MetaLabelGenerator:
    """
    Meta-Label 生成器
    
    整合 Primary 信号与 Triple Barrier 生成元标签。
    
    使用方式：
    ```python
    generator = MetaLabelGenerator(config)
    
    # 从 Primary 信号生成元标签
    result = generator.generate(
        closes=closes,
        primary_signals=signals,  # (bar_idx, direction)
    )
    
    # 获取训练数据
    X = features[result.event_indices]
    y = result.meta_labels
    weights = result.sample_weights
    ```
    """
    config: MetaLabelConfig = field(default_factory=MetaLabelConfig)
    
    # 内部组件
    _labeler: TripleBarrierLabeler | None = field(default=None, init=False)
    
    def __post_init__(self) -> None:
        """初始化组件"""
        self._labeler = TripleBarrierLabeler(self.config.triple_barrier_config)
    
    def generate(
        self,
        closes: NDArray[np.float64],
        primary_signals: list[tuple[int, int]],
        volatilities: NDArray[np.float64] | None = None,
    ) -> MetaLabelResult:
        """
        生成元标签
        
        Args:
            closes: 收盘价序列
            primary_signals: Primary 信号列表 [(bar_idx, direction), ...]
            volatilities: 波动率序列（可选）
            
        Returns:
            MetaLabelResult 包含所有元标签数据
        """
        if not primary_signals:
            return MetaLabelResult(
                event_indices=np.array([], dtype=np.int64),
                primary_directions=np.array([], dtype=np.int32),
                meta_labels=np.array([], dtype=np.int32),
                sample_weights=np.array([], dtype=np.float32),
                returns_pct=np.array([], dtype=np.float32),
                holding_bars=np.array([], dtype=np.int32),
                events=[],
            )
        
        # 提取信号索引和方向
        event_indices = np.array([s[0] for s in primary_signals], dtype=np.int64)
        primary_directions = np.array([s[1] for s in primary_signals], dtype=np.int32)
        
        # 生成 Triple Barrier 标签
        events = self._labeler.generate_labels(
            closes=closes,
            event_indices=event_indices,
            primary_directions=primary_directions,
            volatilities=volatilities,
        )
        
        # 提取元标签
        meta_labels = np.array([1 if e.is_success else 0 for e in events], dtype=np.int32)
        returns_pct = np.array([e.return_pct for e in events], dtype=np.float32)
        holding_bars = np.array([e.holding_bars for e in events], dtype=np.int32)
        
        # 计算样本权重
        sample_weights = self._compute_sample_weights(
            events, event_indices, len(closes)
        )
        
        result = MetaLabelResult(
            event_indices=event_indices,
            primary_directions=primary_directions,
            meta_labels=meta_labels,
            sample_weights=sample_weights,
            returns_pct=returns_pct,
            holding_bars=holding_bars,
            events=events,
        )
        
        logger.info(
            "Generated meta labels",
            **result.get_distribution(),
        )
        
        return result
    
    def generate_from_primary_engine(
        self,
        closes: NDArray[np.float64],
        signal_bars: NDArray[np.int64],
        signal_directions: NDArray[np.int32],
        volatilities: NDArray[np.float64] | None = None,
    ) -> MetaLabelResult:
        """
        从 Primary Engine 输出生成元标签
        
        这是更直接的接口，直接接收数组输入。
        
        Args:
            closes: 收盘价序列
            signal_bars: 信号 Bar 索引数组
            signal_directions: 信号方向数组 (1=LONG, -1=SHORT)
            volatilities: 波动率序列（可选）
            
        Returns:
            MetaLabelResult
        """
        # 转换为 tuple 列表
        primary_signals = list(zip(signal_bars.tolist(), signal_directions.tolist()))
        return self.generate(closes, primary_signals, volatilities)
    
    def _compute_sample_weights(
        self,
        events: list[BarrierEvent],
        event_indices: NDArray[np.int64],
        n_bars: int,
    ) -> NDArray[np.float32]:
        """计算样本权重"""
        n = len(events)
        if n == 0:
            return np.array([], dtype=np.float32)
        
        weights = np.ones(n, dtype=np.float32)
        
        method = self.config.sample_weight_method
        
        if method == "uniform":
            # 均匀权重
            pass
        
        elif method == "return_based":
            # 基于收益幅度的权重（更大的收益/亏损 = 更明确的信号）
            returns = np.array([abs(e.return_pct) for e in events], dtype=np.float32)
            if np.max(returns) > 0:
                weights = returns / (np.mean(returns) + 1e-6)
            weights = np.clip(weights, 0.1, 10.0)  # 防止极端权重
        
        elif method == "time_decay":
            # 时间衰减权重（近期样本权重更高）
            half_life = self.config.time_decay_half_life
            decay_rate = np.log(2) / half_life
            
            max_idx = n_bars - 1
            for i, idx in enumerate(event_indices):
                time_diff = max_idx - idx
                weights[i] = np.exp(-decay_rate * time_diff)
        
        # 归一化
        weights = weights / (np.sum(weights) + 1e-6) * n
        
        return weights
    
    def get_class_weights(
        self,
        result: MetaLabelResult,
    ) -> dict[int, float]:
        """
        计算类别权重（用于不平衡校正）
        
        Args:
            result: MetaLabelResult
            
        Returns:
            {0: weight_0, 1: weight_1}
        """
        n = result.n_samples
        if n == 0:
            return {0: 1.0, 1: 1.0}
        
        n_pos = np.sum(result.meta_labels)
        n_neg = n - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return {0: 1.0, 1: 1.0}
        
        # 反比例权重
        w_pos = n / (2 * n_pos)
        w_neg = n / (2 * n_neg)
        
        return {0: float(w_neg), 1: float(w_pos)}


def create_training_data(
    features: NDArray[np.float32],
    closes: NDArray[np.float64],
    primary_signals: list[tuple[int, int]],
    config: MetaLabelConfig | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.float32]]:
    """
    便捷函数：创建训练数据
    
    Args:
        features: 特征矩阵 (n_bars, n_features)
        closes: 收盘价序列
        primary_signals: Primary 信号列表 [(bar_idx, direction), ...]
        config: MetaLabelConfig（可选）
        
    Returns:
        (X, y, sample_weights)
    """
    if config is None:
        config = MetaLabelConfig()
    
    generator = MetaLabelGenerator(config)
    result = generator.generate(closes, primary_signals)
    
    # 提取对应的特征
    X = features[result.event_indices]
    y = result.meta_labels
    weights = result.sample_weights
    
    return X, y, weights
