"""
ML 友好特征管道

整合所有新特征模块的统一管道：
- 基础特征 (FeaturePipelineV4)
- FVG 特征 (FVGFeatureCalculator)
- 15m SuperTrend 特征 (SuperTrendFeatureCalculator)
- 时间特征 (TimeFeatureCalculator)
- 交叉特征 (InteractionFeatureCalculator)

输出完整的特征矩阵供 ML 模型使用。

使用方式：
```python
pipeline = MLFeaturePipeline(config)
features = pipeline.compute_batch(event_bars)  # (n_bars, ~47)
labels = pipeline.compute_labels(event_bars)   # (n_bars, 10)
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from alphaos.core.logging import get_logger
from alphaos.data.event_bars.tick_imbalance import EventBar
from alphaos.v4.types import Bar, Timeframe

# 新特征模块
from alphaos.v4.features.fvg_features import (
    FVGFeatureCalculator,
    FVGFeatureConfig,
    FVGFeaturesResult,
)
from alphaos.v4.features.fvg_event_features import (
    FVGEventCalculator,
    FVGEventConfig,
    FVGEventFeaturesResult,
    ATRRatioCalculator,
    STAlignmentCalculator,
)
from alphaos.v4.features.supertrend_features import (
    SuperTrendFeatureCalculator,
    SuperTrendFeatureConfig,
    SuperTrendFeaturesResult,
    compute_aligned_supertrend_features,
)
from alphaos.v4.features.time_features import (
    TimeFeatureCalculator,
    TimeFeatureConfig,
    TimeFeaturesResult,
)
from alphaos.v4.features.interaction_features import (
    InteractionFeatureCalculator,
    InteractionFeaturesResult,
    compute_interaction_features_batch,
)

# 时间聚合
from alphaos.v4.sampling.time_aggregator import (
    TimeBarAggregator,
    TimeBarAggregatorConfig,
    TimeBar,
    align_features_to_lower_timeframe,
)

# 标签
from alphaos.v4.labeling.multi_horizon import (
    MultiHorizonLabeler,
    MultiHorizonConfig,
    MultiHorizonLabelsResult,
)

logger = get_logger(__name__)


@dataclass
class MLFeatureConfig:
    """
    ML 特征管道配置
    
    Args:
        # FVG 特征（活跃状态）
        fvg_enabled: 是否计算 FVG 特征
        fvg_config: FVG 特征配置
        
        # FVG 事件特征（事件型 + 因果跟随）
        fvg_event_enabled: 是否计算 FVG 事件特征
        fvg_event_config: FVG 事件特征配置
        
        # 15m SuperTrend 特征
        supertrend_15m_enabled: 是否计算 15m SuperTrend 特征
        supertrend_config: SuperTrend 特征配置
        aggregation_interval_15m: 15m 聚合间隔（秒）
        
        # 时间特征
        time_enabled: 是否计算时间特征
        time_config: 时间特征配置
        
        # 交叉特征
        interaction_enabled: 是否计算交叉特征
        
        # ATR 特征
        atr_features_enabled: 是否计算 ATR 相关特征
        
        # ST Alignment 特征
        st_alignment_enabled: 是否计算 ST 对齐特征
        
        # 标签
        label_config: 多 Horizon 标签配置
    """
    # FVG 特征（活跃状态）
    fvg_enabled: bool = True
    fvg_config: FVGFeatureConfig = field(default_factory=FVGFeatureConfig)
    
    # FVG 事件特征（事件型 + 因果跟随）
    fvg_event_enabled: bool = True
    fvg_event_config: FVGEventConfig = field(default_factory=FVGEventConfig)
    
    # 15m SuperTrend
    supertrend_15m_enabled: bool = True
    supertrend_config: SuperTrendFeatureConfig = field(default_factory=SuperTrendFeatureConfig)
    aggregation_interval_15m: int = 900  # 15 分钟
    
    # 时间特征
    time_enabled: bool = True
    time_config: TimeFeatureConfig = field(default_factory=TimeFeatureConfig)
    
    # 交叉特征
    interaction_enabled: bool = True
    
    # ATR 特征
    atr_features_enabled: bool = True
    
    # ST Alignment 特征
    st_alignment_enabled: bool = True
    
    # 标签
    label_config: MultiHorizonConfig = field(default_factory=MultiHorizonConfig)
    
    def to_dict(self) -> dict:
        return {
            "fvg_enabled": self.fvg_enabled,
            "fvg_event_enabled": self.fvg_event_enabled,
            "supertrend_15m_enabled": self.supertrend_15m_enabled,
            "aggregation_interval_15m": self.aggregation_interval_15m,
            "time_enabled": self.time_enabled,
            "interaction_enabled": self.interaction_enabled,
            "atr_features_enabled": self.atr_features_enabled,
            "st_alignment_enabled": self.st_alignment_enabled,
            "label_horizons": self.label_config.horizons,
            "label_threshold_bps": self.label_config.threshold_bps,
        }


@dataclass
class MLFeatureResult:
    """
    ML 特征计算结果
    
    Attributes:
        features: 完整特征矩阵 (n_samples, n_features)
        labels: 多 Horizon 标签矩阵 (n_samples, 10)
        feature_names: 特征名称列表
        label_names: 标签名称列表
        n_samples: 样本数量
        n_features: 特征数量
    """
    features: NDArray[np.float32]
    labels: NDArray[np.float32] | None = None
    feature_names: list[str] = field(default_factory=list)
    label_names: list[str] = field(default_factory=list)
    n_samples: int = 0
    n_features: int = 0
    
    def get_feature(self, name: str) -> NDArray[np.float32]:
        """按名称获取特征列"""
        if name not in self.feature_names:
            raise ValueError(f"Feature not found: {name}")
        idx = self.feature_names.index(name)
        return self.features[:, idx]
    
    def get_label(self, name: str) -> NDArray[np.float32]:
        """按名称获取标签列"""
        if self.labels is None:
            raise ValueError("Labels not computed")
        if name not in self.label_names:
            raise ValueError(f"Label not found: {name}")
        idx = self.label_names.index(name)
        return self.labels[:, idx]
    
    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "label_names": self.label_names,
        }


@dataclass
class MLFeaturePipeline:
    """
    ML 友好特征管道
    
    整合所有新特征模块，提供统一的特征计算接口。
    
    关键设计：
    - 不过滤任何数据（每个 bar 都计算特征）
    - 输出连续值特征（模型自己学阈值）
    - 支持多时间框架（1m + 15m）
    - 计算交叉特征（趋势 × FVG）
    
    使用方式：
    ```python
    pipeline = MLFeaturePipeline(config)
    
    # 计算特征和标签
    result = pipeline.compute_batch(event_bars, compute_labels=True)
    
    X = result.features  # (n_bars, n_features)
    y = result.labels    # (n_bars, 10)
    
    # 训练模型
    model.fit(X, y[:, 2])  # 使用 label_10_long 作为目标
    ```
    """
    config: MLFeatureConfig = field(default_factory=MLFeatureConfig)
    
    # 内部计算器
    _fvg_calc: FVGFeatureCalculator | None = field(default=None, init=False)
    _fvg_event_calc: FVGEventCalculator | None = field(default=None, init=False)
    _st_calc: SuperTrendFeatureCalculator | None = field(default=None, init=False)
    _time_calc: TimeFeatureCalculator | None = field(default=None, init=False)
    _interaction_calc: InteractionFeatureCalculator | None = field(default=None, init=False)
    _time_aggregator: TimeBarAggregator | None = field(default=None, init=False)
    _labeler: MultiHorizonLabeler | None = field(default=None, init=False)
    
    # 特征名称
    _feature_names: list[str] = field(default_factory=list, init=False)
    
    def __post_init__(self) -> None:
        """初始化组件"""
        self._init_components()
        self._build_feature_names()
    
    def _init_components(self) -> None:
        """初始化内部组件"""
        # FVG 特征计算器（活跃状态）
        if self.config.fvg_enabled:
            self._fvg_calc = FVGFeatureCalculator(config=self.config.fvg_config)
        
        # FVG 事件特征计算器（事件型 + 因果跟随）
        if self.config.fvg_event_enabled:
            self._fvg_event_calc = FVGEventCalculator(config=self.config.fvg_event_config)
        
        # SuperTrend 特征计算器
        if self.config.supertrend_15m_enabled:
            self._st_calc = SuperTrendFeatureCalculator(config=self.config.supertrend_config)
            self._time_aggregator = TimeBarAggregator(
                config=TimeBarAggregatorConfig(
                    interval_seconds=self.config.aggregation_interval_15m
                )
            )
        
        # 时间特征计算器
        if self.config.time_enabled:
            self._time_calc = TimeFeatureCalculator(config=self.config.time_config)
        
        # 交叉特征计算器
        if self.config.interaction_enabled:
            self._interaction_calc = InteractionFeatureCalculator()
        
        # 标签生成器
        self._labeler = MultiHorizonLabeler(config=self.config.label_config)
        
        logger.info(
            "MLFeaturePipeline initialized",
            config=self.config.to_dict(),
        )
    
    def _build_feature_names(self) -> None:
        """构建特征名称列表"""
        names = []
        
        # FVG 特征（活跃状态）
        if self.config.fvg_enabled:
            names.extend(FVGFeaturesResult.feature_names())
        
        # FVG 事件特征（事件型 + 因果跟随）
        if self.config.fvg_event_enabled:
            names.extend(FVGEventFeaturesResult.feature_names())
        
        # 15m SuperTrend 特征
        if self.config.supertrend_15m_enabled:
            names.extend(SuperTrendFeaturesResult.feature_names(suffix="15m"))
        
        # 时间特征
        if self.config.time_enabled:
            names.extend(TimeFeaturesResult.feature_names())
        
        # 交叉特征
        if self.config.interaction_enabled:
            names.extend(InteractionFeaturesResult.feature_names())
        
        # ST Alignment 特征
        if self.config.st_alignment_enabled:
            names.append("st_alignment")
        
        # ATR 特征
        if self.config.atr_features_enabled:
            names.extend(["atr_ratio_1m_15m", "atr_1m", "atr_15m", "mid_15m_range"])
        
        self._feature_names = names
    
    def compute_batch(
        self,
        bars: Sequence[EventBar],
        compute_labels: bool = True,
        use_parallel: bool = True,
        max_workers: int = 4,
    ) -> MLFeatureResult:
        """
        批量计算特征（支持并行化）
        
        Args:
            bars: EventBar 序列
            compute_labels: 是否计算标签
            use_parallel: 是否使用并行计算（默认 True）
            max_workers: 最大并行工作线程数
            
        Returns:
            MLFeatureResult 包含特征矩阵和可选的标签
        """
        n_bars = len(bars)
        
        if n_bars == 0:
            return MLFeatureResult(
                features=np.zeros((0, len(self._feature_names)), dtype=np.float32),
                labels=None,
                feature_names=self._feature_names.copy(),
                label_names=MultiHorizonLabelsResult.label_names(),
                n_samples=0,
                n_features=len(self._feature_names),
            )
        
        # 对于大量数据，使用并行计算
        if use_parallel and n_bars > 1000:
            return self._compute_batch_parallel(bars, compute_labels, max_workers)
        
        return self._compute_batch_sequential(bars, compute_labels)
    
    def _compute_batch_parallel(
        self,
        bars: Sequence[EventBar],
        compute_labels: bool,
        max_workers: int,
    ) -> MLFeatureResult:
        """并行批量计算特征"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        n_bars = len(bars)
        
        # Phase 1: 计算 15m 相关数据（顺序执行，其他特征依赖它）
        bars_15m = []
        mid_15m_arr = np.zeros(n_bars, dtype=np.float64)
        atr_15m_arr = np.ones(n_bars, dtype=np.float64)
        alignment_indices = np.zeros(n_bars, dtype=np.int64)
        
        if self.config.supertrend_15m_enabled and self._time_aggregator is not None:
            bars_15m = self._time_aggregator.aggregate_batch(bars)
            if len(bars_15m) > 0:
                alignment_indices = align_features_to_lower_timeframe(bars_15m, bars)
                for i, idx in enumerate(alignment_indices):
                    if 0 <= idx < len(bars_15m):
                        bar_15m = bars_15m[idx]
                        mid_15m_arr[i] = (bar_15m.high + bar_15m.low) / 2
                        atr_15m_arr[i] = bar_15m.high - bar_15m.low
        
        # Phase 2: 并行计算独立特征
        feature_results = {}
        
        def compute_fvg():
            if self.config.fvg_enabled and self._fvg_calc is not None:
                return self._fvg_calc.compute_batch(bars)
            return None
        
        def compute_fvg_event():
            if self.config.fvg_event_enabled and self._fvg_event_calc is not None:
                return self._fvg_event_calc.compute_batch(
                    bars,
                    mid_15m_arr=mid_15m_arr,
                    atr_1m_arr=None,
                    atr_15m_arr=atr_15m_arr,
                )
            return None
        
        def compute_supertrend():
            if self.config.supertrend_15m_enabled and self._st_calc is not None:
                if len(bars_15m) > 0:
                    st_features_15m = self._st_calc.compute_batch(bars_15m)
                    st_features = np.zeros((n_bars, SuperTrendFeaturesResult.n_features()), dtype=np.float32)
                    for i, idx in enumerate(alignment_indices):
                        if 0 <= idx < len(st_features_15m):
                            st_features[i] = st_features_15m[idx]
                    return st_features
                else:
                    return np.zeros((n_bars, SuperTrendFeaturesResult.n_features()), dtype=np.float32)
            return None
        
        def compute_time():
            if self.config.time_enabled and self._time_calc is not None:
                return self._time_calc.compute_batch(bars)
            return None
        
        # 使用线程池并行执行
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(compute_fvg): "fvg",
                executor.submit(compute_fvg_event): "fvg_event",
                executor.submit(compute_supertrend): "supertrend",
                executor.submit(compute_time): "time",
            }
            
            for future in as_completed(futures):
                key = futures[future]
                try:
                    feature_results[key] = future.result()
                except Exception as e:
                    logger.warning(f"Feature computation failed for {key}: {e}")
                    feature_results[key] = None
        
        # 收集结果
        feature_arrays = []
        fvg_features = feature_results.get("fvg")
        fvg_event_features = feature_results.get("fvg_event")
        st_features = feature_results.get("supertrend")
        time_features = feature_results.get("time")
        
        if fvg_features is not None:
            feature_arrays.append(fvg_features)
        if fvg_event_features is not None:
            feature_arrays.append(fvg_event_features)
        if st_features is not None:
            feature_arrays.append(st_features)
        if time_features is not None:
            feature_arrays.append(time_features)
        
        # Phase 3: 计算依赖特征（顺序执行）
        # 交叉特征
        if self.config.interaction_enabled and self._interaction_calc is not None:
            if fvg_features is not None and st_features is not None:
                interaction_features = compute_interaction_features_batch(
                    st_features=st_features,
                    fvg_features=fvg_features,
                    time_features=time_features,
                    volatility=None,
                )
                feature_arrays.append(interaction_features)
            else:
                feature_arrays.append(
                    np.zeros((n_bars, InteractionFeaturesResult.n_features()), dtype=np.float32)
                )
        
        # ST Alignment
        if self.config.st_alignment_enabled:
            st_alignment = np.zeros((n_bars, 1), dtype=np.float32)
            if st_features is not None and fvg_event_features is not None:
                st_trend = st_features[:, 0].astype(np.int32)
                fvg_event = fvg_event_features[:, 0].astype(np.int32)
                st_alignment[:, 0] = STAlignmentCalculator.compute_batch(st_trend, fvg_event)
            feature_arrays.append(st_alignment)
        
        # ATR 特征
        if self.config.atr_features_enabled:
            atr_features = np.zeros((n_bars, 4), dtype=np.float32)
            atr_1m = self._fvg_event_calc._current_atr if self._fvg_event_calc else 1.0
            atr_features[:, 0] = atr_1m / np.maximum(atr_15m_arr, 1e-10)
            atr_features[:, 1] = atr_1m
            atr_features[:, 2] = atr_15m_arr
            atr_features[:, 3] = mid_15m_arr
            feature_arrays.append(atr_features)
        
        # 合并所有特征
        features = np.hstack(feature_arrays) if feature_arrays else np.zeros((n_bars, 0), dtype=np.float32)
        
        # 计算标签
        labels = None
        if compute_labels and self._labeler is not None:
            labels = self._labeler.compute_batch(bars)
        
        return MLFeatureResult(
            features=features,
            labels=labels,
            feature_names=self._feature_names.copy(),
            label_names=MultiHorizonLabelsResult.label_names(),
            n_samples=n_bars,
            n_features=features.shape[1] if features.ndim > 1 else 0,
        )
    
    def _compute_batch_sequential(
        self,
        bars: Sequence[EventBar],
        compute_labels: bool,
    ) -> MLFeatureResult:
        """顺序批量计算特征（原始实现）"""
        n_bars = len(bars)
        
        # 收集所有特征
        feature_arrays = []
        
        # 首先聚合 15m bars 和计算 15m 相关数据（供后续使用）
        bars_15m = []
        mid_15m_arr = np.zeros(n_bars, dtype=np.float64)
        atr_15m_arr = np.ones(n_bars, dtype=np.float64)
        alignment_indices = np.zeros(n_bars, dtype=np.int64)
        
        if self.config.supertrend_15m_enabled and self._time_aggregator is not None:
            bars_15m = self._time_aggregator.aggregate_batch(bars)
            if len(bars_15m) > 0:
                alignment_indices = align_features_to_lower_timeframe(bars_15m, bars)
                # 计算 15m mid-range 和 ATR
                for i, idx in enumerate(alignment_indices):
                    if 0 <= idx < len(bars_15m):
                        bar_15m = bars_15m[idx]
                        mid_15m_arr[i] = (bar_15m.high + bar_15m.low) / 2
                        # 简单计算 15m ATR (range)
                        atr_15m_arr[i] = bar_15m.high - bar_15m.low
        
        # 1. FVG 特征（活跃状态）
        fvg_features = None
        if self.config.fvg_enabled and self._fvg_calc is not None:
            fvg_features = self._fvg_calc.compute_batch(bars)
            feature_arrays.append(fvg_features)
        
        # 2. FVG 事件特征（事件型 + 因果跟随）
        fvg_event_features = None
        if self.config.fvg_event_enabled and self._fvg_event_calc is not None:
            fvg_event_features = self._fvg_event_calc.compute_batch(
                bars,
                mid_15m_arr=mid_15m_arr,
                atr_1m_arr=None,  # 让内部计算
                atr_15m_arr=atr_15m_arr,
            )
            feature_arrays.append(fvg_event_features)
        
        # 3. 15m SuperTrend 特征
        st_features = None
        if self.config.supertrend_15m_enabled and self._st_calc is not None:
            if len(bars_15m) > 0:
                # 计算 15m SuperTrend 特征
                st_features_15m = self._st_calc.compute_batch(bars_15m)
                
                # 对齐到 1m bars
                st_features = np.zeros((n_bars, SuperTrendFeaturesResult.n_features()), dtype=np.float32)
                
                for i, idx in enumerate(alignment_indices):
                    if 0 <= idx < len(st_features_15m):
                        st_features[i] = st_features_15m[idx]
            else:
                st_features = np.zeros((n_bars, SuperTrendFeaturesResult.n_features()), dtype=np.float32)
            
            feature_arrays.append(st_features)
        
        # 4. 时间特征
        time_features = None
        if self.config.time_enabled and self._time_calc is not None:
            time_features = self._time_calc.compute_batch(bars)
            feature_arrays.append(time_features)
        
        # 5. 交叉特征
        if self.config.interaction_enabled and self._interaction_calc is not None:
            # 需要 FVG 和 SuperTrend 特征
            if fvg_features is not None and st_features is not None:
                interaction_features = compute_interaction_features_batch(
                    st_features=st_features,
                    fvg_features=fvg_features,
                    time_features=time_features,
                    volatility=None,  # 可以从其他地方获取
                )
                feature_arrays.append(interaction_features)
            else:
                # 创建空的交叉特征
                interaction_features = np.zeros(
                    (n_bars, InteractionFeaturesResult.n_features()), 
                    dtype=np.float32
                )
                feature_arrays.append(interaction_features)
        
        # 6. ST Alignment 特征
        if self.config.st_alignment_enabled:
            st_alignment = np.zeros((n_bars, 1), dtype=np.float32)
            if st_features is not None and fvg_event_features is not None:
                # st_trend_15m (index 0) * fvg_event (index 0)
                st_trend = st_features[:, 0].astype(np.int32)
                fvg_event = fvg_event_features[:, 0].astype(np.int32)
                st_alignment[:, 0] = STAlignmentCalculator.compute_batch(st_trend, fvg_event)
            feature_arrays.append(st_alignment)
        
        # 7. ATR 特征
        if self.config.atr_features_enabled:
            atr_features = np.zeros((n_bars, 4), dtype=np.float32)
            # 从 FVG event calculator 获取 ATR 1m
            if self._fvg_event_calc is not None:
                atr_1m = self._fvg_event_calc._current_atr
            else:
                atr_1m = 1.0
            
            atr_features[:, 0] = atr_1m / np.maximum(atr_15m_arr, 1e-10)  # atr_ratio_1m_15m
            atr_features[:, 1] = atr_1m  # atr_1m (scalar, same for all)
            atr_features[:, 2] = atr_15m_arr  # atr_15m
            atr_features[:, 3] = mid_15m_arr  # mid_15m_range
            feature_arrays.append(atr_features)
        
        # 合并所有特征
        if feature_arrays:
            features = np.hstack(feature_arrays)
        else:
            features = np.zeros((n_bars, 0), dtype=np.float32)
        
        # 计算标签
        labels = None
        if compute_labels and self._labeler is not None:
            labels = self._labeler.compute_batch(bars)
        
        result = MLFeatureResult(
            features=features,
            labels=labels,
            feature_names=self._feature_names.copy(),
            label_names=MultiHorizonLabelsResult.label_names(),
            n_samples=n_bars,
            n_features=features.shape[1] if features.ndim > 1 else 0,
        )
        
        logger.info(
            "ML features computed",
            n_samples=n_bars,
            n_features=result.n_features,
            feature_groups={
                "fvg": FVGFeaturesResult.n_features() if self.config.fvg_enabled else 0,
                "fvg_event": FVGEventFeaturesResult.n_features() if self.config.fvg_event_enabled else 0,
                "supertrend_15m": SuperTrendFeaturesResult.n_features() if self.config.supertrend_15m_enabled else 0,
                "time": TimeFeaturesResult.n_features() if self.config.time_enabled else 0,
                "interaction": InteractionFeaturesResult.n_features() if self.config.interaction_enabled else 0,
                "st_alignment": 1 if self.config.st_alignment_enabled else 0,
                "atr": 4 if self.config.atr_features_enabled else 0,
            },
        )
        
        return result
    
    @property
    def feature_names(self) -> list[str]:
        """特征名称列表"""
        return self._feature_names.copy()
    
    @property
    def n_features(self) -> int:
        """特征数量"""
        return len(self._feature_names)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = {
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "config": self.config.to_dict(),
        }
        
        if self._fvg_calc:
            stats["fvg_stats"] = self._fvg_calc.get_stats()
        if self._st_calc:
            stats["supertrend_stats"] = self._st_calc.get_stats()
        if self._time_calc:
            stats["time_stats"] = self._time_calc.get_stats()
        
        return stats
    
    def reset(self) -> None:
        """重置所有内部状态"""
        if self._fvg_calc:
            self._fvg_calc.reset()
        if self._fvg_event_calc:
            self._fvg_event_calc.reset()
        if self._st_calc:
            self._st_calc.reset()
        if self._time_calc:
            self._time_calc.reset()
        if self._time_aggregator:
            self._time_aggregator.reset()
