"""
CPCV (Combinatorial Purged Cross-Validation)

金融时间序列的交叉验证方法，防止数据泄露：
- Purging: 移除训练集中与测试集时间重叠的样本
- Embargo: 在测试集前后添加缓冲期

参考：
- López de Prado, "Advances in Financial Machine Learning", Chapter 7
- 降噪LNN特征提取与信号过滤.md Section 6
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence
import itertools

import numpy as np
from numpy.typing import NDArray

from alphaos.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbargoConfig:
    """
    Embargo 配置
    
    Args:
        embargo_pct: Embargo 期占训练集的比例
        min_embargo_bars: 最小 Embargo Bar 数
        max_embargo_bars: 最大 Embargo Bar 数
    """
    embargo_pct: float = 0.01  # 1%
    min_embargo_bars: int = 1
    max_embargo_bars: int = 100
    
    def get_embargo_size(self, n_samples: int) -> int:
        """计算 Embargo 大小"""
        size = int(n_samples * self.embargo_pct)
        return max(self.min_embargo_bars, min(size, self.max_embargo_bars))


@dataclass
class PurgedKFold:
    """
    Purged K-Fold 交叉验证
    
    在标准 K-Fold 基础上添加：
    - Purging: 移除与测试集时间重叠的训练样本
    - Embargo: 在测试集后添加缓冲期，防止信息泄露
    
    使用方式：
    ```python
    cv = PurgedKFold(n_splits=5, embargo_config=EmbargoConfig())
    
    for train_idx, test_idx in cv.split(X, y, event_times):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # 训练模型...
    ```
    
    Args:
        n_splits: 折数
        embargo_config: Embargo 配置
        shuffle: 是否打乱（金融数据通常不打乱）
    """
    n_splits: int = 5
    embargo_config: EmbargoConfig = field(default_factory=EmbargoConfig)
    shuffle: bool = False
    
    def split(
        self,
        X: NDArray,
        y: NDArray | None = None,
        event_times: NDArray | None = None,
        holding_periods: NDArray | None = None,
    ) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """
        生成训练/测试索引
        
        Args:
            X: 特征矩阵
            y: 标签（可选）
            event_times: 事件时间索引（Bar 索引）
            holding_periods: 持仓周期（Bar 数）
            
        Yields:
            (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        # 计算折大小
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        test_starts = []
        test_ends = []
        
        for fold_size in fold_sizes:
            test_starts.append(current)
            test_ends.append(current + fold_size)
            current += fold_size
        
        # 计算 embargo 大小
        embargo_size = self.embargo_config.get_embargo_size(n_samples)
        
        # 生成每折的索引
        for fold_idx in range(self.n_splits):
            test_start = test_starts[fold_idx]
            test_end = test_ends[fold_idx]
            test_indices = indices[test_start:test_end]
            
            # 计算需要 purge 的范围
            # Purge: 训练样本的持仓期与测试期重叠的样本
            # Embargo: 测试期结束后的缓冲期
            
            train_indices_list = []
            
            for i in range(self.n_splits):
                if i == fold_idx:
                    continue
                
                fold_start = test_starts[i]
                fold_end = test_ends[i]
                
                # 检查是否需要 purge
                if event_times is not None and holding_periods is not None:
                    # 复杂的 purging 逻辑
                    valid_mask = self._get_valid_train_mask(
                        indices[fold_start:fold_end],
                        test_start,
                        test_end,
                        event_times,
                        holding_periods,
                        embargo_size,
                    )
                    fold_indices = indices[fold_start:fold_end][valid_mask]
                else:
                    # 简单的 embargo 逻辑
                    fold_indices = self._apply_simple_embargo(
                        indices[fold_start:fold_end],
                        test_start,
                        test_end,
                        embargo_size,
                    )
                
                train_indices_list.append(fold_indices)
            
            train_indices = np.concatenate(train_indices_list)
            
            yield train_indices, test_indices
    
    def _get_valid_train_mask(
        self,
        train_candidates: NDArray[np.int64],
        test_start: int,
        test_end: int,
        event_times: NDArray,
        holding_periods: NDArray,
        embargo_size: int,
    ) -> NDArray[np.bool_]:
        """获取有效训练样本的掩码（应用 Purging）"""
        mask = np.ones(len(train_candidates), dtype=np.bool_)
        
        for i, idx in enumerate(train_candidates):
            event_time = event_times[idx]
            holding_period = holding_periods[idx]
            event_end = event_time + holding_period
            
            # Purging: 如果事件持仓期与测试期重叠，移除
            if event_time < test_end + embargo_size and event_end > test_start:
                mask[i] = False
        
        return mask
    
    def _apply_simple_embargo(
        self,
        train_indices: NDArray[np.int64],
        test_start: int,
        test_end: int,
        embargo_size: int,
    ) -> NDArray[np.int64]:
        """应用简单的 Embargo（基于索引距离）"""
        # 移除测试期前后 embargo_size 范围内的训练样本
        purge_start = test_start - embargo_size
        purge_end = test_end + embargo_size
        
        mask = (train_indices < purge_start) | (train_indices >= purge_end)
        return train_indices[mask]
    
    def get_n_splits(self) -> int:
        """获取折数"""
        return self.n_splits


@dataclass
class CPCVSplitter:
    """
    Combinatorial Purged Cross-Validation
    
    组合式交叉验证：从 N 折中选择 k 折作为测试集，
    生成 C(N, k) 种组合。
    
    优势：
    - 更多的测试路径组合
    - 更稳定的性能估计
    - 更好地利用数据
    
    使用方式：
    ```python
    cv = CPCVSplitter(n_splits=10, n_test_splits=2)
    
    # 获取所有组合
    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"Combination {i}: test folds = {cv.get_test_folds(i)}")
    ```
    
    Args:
        n_splits: 总折数
        n_test_splits: 测试集折数
        embargo_config: Embargo 配置
    """
    n_splits: int = 10
    n_test_splits: int = 2
    embargo_config: EmbargoConfig = field(default_factory=EmbargoConfig)
    
    # 缓存
    _combinations: list[tuple[int, ...]] | None = field(default=None, init=False)
    
    def __post_init__(self) -> None:
        """生成所有组合"""
        self._combinations = list(
            itertools.combinations(range(self.n_splits), self.n_test_splits)
        )
        logger.info(
            f"CPCVSplitter: {self.n_splits} folds, {self.n_test_splits} test splits, "
            f"{len(self._combinations)} combinations"
        )
    
    def split(
        self,
        X: NDArray,
        y: NDArray | None = None,
        event_times: NDArray | None = None,
        holding_periods: NDArray | None = None,
    ) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """
        生成所有组合的训练/测试索引
        
        Yields:
            (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # 计算折边界
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        fold_starts = np.zeros(self.n_splits, dtype=int)
        fold_ends = np.zeros(self.n_splits, dtype=int)
        
        current = 0
        for i, fold_size in enumerate(fold_sizes):
            fold_starts[i] = current
            fold_ends[i] = current + fold_size
            current += fold_size
        
        embargo_size = self.embargo_config.get_embargo_size(n_samples)
        
        # 遍历所有组合
        for test_folds in self._combinations:
            # 收集测试索引
            test_indices_list = []
            for fold_idx in test_folds:
                test_indices_list.append(indices[fold_starts[fold_idx]:fold_ends[fold_idx]])
            test_indices = np.concatenate(test_indices_list)
            
            # 计算测试期范围（用于 purging）
            test_range_start = min(fold_starts[f] for f in test_folds)
            test_range_end = max(fold_ends[f] for f in test_folds)
            
            # 收集训练索引
            train_indices_list = []
            for fold_idx in range(self.n_splits):
                if fold_idx in test_folds:
                    continue
                
                fold_start = fold_starts[fold_idx]
                fold_end = fold_ends[fold_idx]
                fold_indices = indices[fold_start:fold_end]
                
                # 应用 embargo
                if event_times is not None and holding_periods is not None:
                    valid_mask = self._get_valid_train_mask(
                        fold_indices,
                        test_range_start,
                        test_range_end,
                        event_times,
                        holding_periods,
                        embargo_size,
                    )
                    fold_indices = fold_indices[valid_mask]
                else:
                    # 简单 embargo
                    purge_start = test_range_start - embargo_size
                    purge_end = test_range_end + embargo_size
                    mask = (fold_indices < purge_start) | (fold_indices >= purge_end)
                    fold_indices = fold_indices[mask]
                
                train_indices_list.append(fold_indices)
            
            train_indices = np.concatenate(train_indices_list) if train_indices_list else np.array([], dtype=np.int64)
            
            yield train_indices, test_indices
    
    def _get_valid_train_mask(
        self,
        train_candidates: NDArray[np.int64],
        test_start: int,
        test_end: int,
        event_times: NDArray,
        holding_periods: NDArray,
        embargo_size: int,
    ) -> NDArray[np.bool_]:
        """获取有效训练样本的掩码"""
        mask = np.ones(len(train_candidates), dtype=np.bool_)
        
        for i, idx in enumerate(train_candidates):
            event_time = event_times[idx]
            holding_period = holding_periods[idx]
            event_end = event_time + holding_period
            
            # Purging
            if event_time < test_end + embargo_size and event_end > test_start:
                mask[i] = False
        
        return mask
    
    def get_n_splits(self) -> int:
        """获取组合总数"""
        return len(self._combinations)
    
    def get_test_folds(self, combination_idx: int) -> tuple[int, ...]:
        """获取特定组合的测试折索引"""
        return self._combinations[combination_idx]
