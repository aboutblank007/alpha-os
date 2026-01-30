"""
v4.0 特征计算管道

统一的特征计算接口，确保与 FeatureSchema 对齐：
- 批量模式（训练）
- 流式模式（推理）
- 特征验证与归一化

核心特征组：
1. 价格特征：log_return, log_return_zscore, spread_bps
2. 微观结构：delta_t_log, tick_intensity, ofi_count, ofi_weighted, kyle_lambda_pct, pdi
3. 波动率：micro_volatility_pct, realized_volatility_pct
4. 热力学：market_temperature, market_entropy, ts_phase
5. 订单流：vpin, vpin_zscore
6. 趋势：trend_deviation, trend_direction, trend_duration
7. 降噪：kalman_residual_bps, kalman_gain, wavelet_energy_*, wavelet_trend_ratio

参考：降噪LNN特征提取与信号过滤.md Section 4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import math

import numpy as np
from numpy.typing import NDArray

from alphaos.core.logging import get_logger
from alphaos.v4.schemas import FeatureSchema, FeatureCategory
from alphaos.v4.features.vpin import VPINCalculator, VPINConfig
from alphaos.v4.denoise import DenoisePipeline, DenoiseConfig, DenoiseMode
from alphaos.data.event_bars.tick_imbalance import EventBar
from alphaos.v4.features.fvg_features import FVGFeatureCalculator
from alphaos.v4.features.fvg_event_features import FVGEventCalculator
from alphaos.v4.features.supertrend_features import SuperTrendFeatureCalculator
from alphaos.v4.features.time_features import TimeFeatureCalculator
from alphaos.v4.features.interaction_features import compute_interaction_features_batch
from alphaos.v4.sampling.time_aggregator import (
    TimeBarAggregator,
    TimeBarAggregatorConfig,
    align_features_to_lower_timeframe,
)

logger = get_logger(__name__)

EPSILON = 1e-10


@dataclass
class ThermodynamicsConfig:
    """
    热力学配置（v4.1 SSOT）
    
    - 温度单位：Var(log_return) / mean(delta_t_ms) * scale
    - 熵：基于收益方向分布的 Shannon entropy
    """
    temperature_frozen: float = 0.02
    temperature_laminar: float = 0.2
    temperature_turbulent: float = 1.0
    entropy_trend: float = 0.3
    entropy_noise: float = 0.7
    scale: float = 1e6
    
    def to_dict(self) -> dict:
        return {
            "temperature": {
                "frozen": self.temperature_frozen,
                "laminar": self.temperature_laminar,
                "turbulent": self.temperature_turbulent,
            },
            "entropy": {
                "trend": self.entropy_trend,
                "noise": self.entropy_noise,
            },
            "scale": self.scale,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ThermodynamicsConfig":
        defaults = cls()
        temperature = data.get("temperature", {}) if isinstance(data, dict) else {}
        entropy = data.get("entropy", {}) if isinstance(data, dict) else {}
        return cls(
            temperature_frozen=temperature.get("frozen", defaults.temperature_frozen),
            temperature_laminar=temperature.get("laminar", defaults.temperature_laminar),
            temperature_turbulent=temperature.get("turbulent", defaults.temperature_turbulent),
            entropy_trend=entropy.get("trend", defaults.entropy_trend),
            entropy_noise=entropy.get("noise", defaults.entropy_noise),
            scale=data.get("scale", defaults.scale) if isinstance(data, dict) else defaults.scale,
        )


@dataclass
class FeatureConfig:
    """
    特征计算配置
    
    Args:
        # 滚动窗口参数
        zscore_window: Z-Score 计算窗口
        volatility_window: 波动率计算窗口
        ofi_window: OFI 计算窗口
        kyle_lambda_window: Kyle Lambda 计算窗口
        thermo_window: 热力学指标窗口
        
        # EWMA 参数
        tick_intensity_alpha: Tick 强度 EWMA 衰减
        micro_vol_lambda: 微观波动率 EWMA 衰减
        
        # VPIN 配置
        vpin_config: VPIN 计算配置
        
        # 降噪配置
        denoise_config: 降噪配置（可选）
        
        # 热力学配置
        thermodynamics: 热力学阈值与缩放配置
        
        # 其他
        clip_zscore: 是否裁剪 Z-Score
        zscore_clip_value: Z-Score 裁剪值
    """
    # 滚动窗口参数
    zscore_window: int = 100
    volatility_window: int = 20
    ofi_window: int = 20
    kyle_lambda_window: int = 50
    thermo_window: int = 50
    
    # EWMA 参数
    tick_intensity_alpha: float = 0.1
    micro_vol_lambda: float = 0.94
    
    # VPIN 配置
    vpin_config: VPINConfig = field(default_factory=VPINConfig)
    
    # 降噪配置
    denoise_config: DenoiseConfig | None = None
    
    # 热力学配置
    thermodynamics: ThermodynamicsConfig = field(default_factory=ThermodynamicsConfig)
    
    # 其他
    clip_zscore: bool = True
    zscore_clip_value: float = 5.0
    
    def to_dict(self) -> dict:
        return {
            "zscore_window": self.zscore_window,
            "volatility_window": self.volatility_window,
            "ofi_window": self.ofi_window,
            "kyle_lambda_window": self.kyle_lambda_window,
            "thermo_window": self.thermo_window,
            "tick_intensity_alpha": self.tick_intensity_alpha,
            "micro_vol_lambda": self.micro_vol_lambda,
            "vpin_config": self.vpin_config.to_dict(),
            "denoise_config": self.denoise_config.to_dict() if self.denoise_config else None,
            "thermodynamics": self.thermodynamics.to_dict(),
            "clip_zscore": self.clip_zscore,
            "zscore_clip_value": self.zscore_clip_value,
        }


@dataclass
class FeatureResult:
    """
    特征计算结果
    
    Attributes:
        features: 特征矩阵 (n_samples, n_features)
        schema: 使用的 FeatureSchema
        is_valid: 每个样本是否有效
        n_samples: 样本数量
    """
    features: NDArray[np.float32]
    schema: FeatureSchema
    is_valid: NDArray[np.bool_] | None = None
    n_samples: int = 0
    
    def get_feature(self, name: str) -> NDArray[np.float32]:
        """
        按名称获取单个特征序列
        
        Args:
            name: 特征名称
            
        Returns:
            特征值数组
        """
        idx = self.schema.get_index(name)
        return self.features[:, idx]
    
    def to_dict(self) -> dict[str, NDArray]:
        """转换为特征名称 -> 数组的字典"""
        return {
            name: self.features[:, i]
            for i, name in enumerate(self.schema.feature_names)
        }


@dataclass
class FeaturePipelineV4:
    """
    v4.0 特征计算管道
    
    提供批量和流式两种模式，确保与 FeatureSchema 完全对齐。
    
    批量模式（训练）：
    ```python
    pipeline = FeaturePipelineV4(config, schema)
    result = pipeline.compute_batch(bars)
    X = result.features
    ```
    
    流式模式（推理）：
    ```python
    pipeline = FeaturePipelineV4(config, schema)
    for bar in bar_stream:
        features = pipeline.update(bar)
        # features 是长度为 n_features 的向量
    ```
    """
    config: FeatureConfig = field(default_factory=FeatureConfig)
    schema: FeatureSchema = field(default_factory=FeatureSchema.default)
    
    # 内部组件
    _vpin_calc: VPINCalculator | None = field(default=None, init=False)
    _denoise_pipeline: DenoisePipeline | None = field(default=None, init=False)
    
    # ML 友好扩展特征（事件型 FVG + 15m ST + 时间/交叉）
    _fvg_calc: FVGFeatureCalculator | None = field(default=None, init=False)
    _fvg_event_calc: FVGEventCalculator | None = field(default=None, init=False)
    _time_calc: TimeFeatureCalculator | None = field(default=None, init=False)
    
    # 15m 聚合与 SuperTrend（严格因果：仅使用已完成的 15m bar）
    _time_agg_15m: TimeBarAggregator | None = field(default=None, init=False)
    _st_15m_calc: SuperTrendFeatureCalculator | None = field(default=None, init=False)
    _last_mid_15m_range: float = field(default=0.0, init=False)
    _last_atr_15m: float = field(default=0.0, init=False)
    _last_st_15m: NDArray[np.float32] | None = field(default=None, init=False)  # shape=(5,)
    
    # 1m ATR（用于事件强度/归一化；严格因果滚动）
    _atr_1m_period: int = field(default=14, init=False)
    _atr_1m_buffer: list[float] = field(default_factory=list, init=False)
    _atr_1m_current: float = field(default=0.0, init=False)
    
    # Schema 索引缓存（减少 get_index 调用开销）
    _idx: dict[str, int] = field(default_factory=dict, init=False)
    
    # 滚动状态（流式模式）
    _bar_buffer: list[EventBar] = field(default_factory=list, init=False)
    _price_buffer: list[float] = field(default_factory=list, init=False)
    _return_buffer: list[float] = field(default_factory=list, init=False)
    _delta_t_buffer: list[float] = field(default_factory=list, init=False)
    _direction_buffer: list[int] = field(default_factory=list, init=False)
    _tick_count_buffer: list[float] = field(default_factory=list, init=False)
    
    # EWMA 状态
    _tick_intensity: float = field(default=1.0, init=False)
    _micro_vol_sq: float = field(default=0.0, init=False)
    _return_mean: float = field(default=0.0, init=False)
    _return_var: float = field(default=1e-6, init=False)
    
    # 趋势状态（需要外部注入）
    _trend_direction: int = field(default=0, init=False)
    _trend_duration: int = field(default=0, init=False)
    _supertrend_line: float = field(default=0.0, init=False)
    
    # 统计
    _bar_count: int = field(default=0, init=False)
    _last_close: float = field(default=0.0, init=False)
    _last_time_us: int = field(default=0, init=False)
    
    def __post_init__(self) -> None:
        """初始化组件"""
        self._init_components()
    
    def _init_components(self) -> None:
        """初始化内部组件"""
        # VPIN 计算器
        self._vpin_calc = VPINCalculator(self.config.vpin_config)
        
        # 降噪管道（如果配置）
        if self.config.denoise_config is not None:
            self._denoise_pipeline = DenoisePipeline(self.config.denoise_config)
        
        # === ML 友好扩展模块（与 FeatureSchema 对齐）===
        # 说明：这些特征用于“事件中心 + 严格因果”的训练/推理对齐。
        self._fvg_calc = FVGFeatureCalculator()
        self._fvg_event_calc = FVGEventCalculator()
        self._time_calc = TimeFeatureCalculator()
        
        # 15m 聚合器与 15m SuperTrend 计算器（严格因果：仅使用已完成的 15m bar）
        self._time_agg_15m = TimeBarAggregator(
            config=TimeBarAggregatorConfig(interval_seconds=900)
        )
        self._st_15m_calc = SuperTrendFeatureCalculator()
        self._last_st_15m = np.zeros(5, dtype=np.float32)
        
        # Schema 索引缓存（仅缓存本管道会写入的字段；schema 演进时允许缺失）
        self._idx = {}
        for name in [
            # FVG（活跃状态）
            "bullish_fvg", "bearish_fvg", "fvg_size_atr", "fvg_age_bars",
            "price_to_fvg_mid", "fvg_filled_pct", "active_fvg_count", "nearest_fvg_distance",
            # FVG_event（事件型 + 因果跟随）
            "fvg_event", "fvg_impulse_atr", "fvg_location_15m",
            "fvg_follow_up_3", "fvg_follow_dn_3", "fvg_follow_bars", "fvg_follow_net",
            # 15m SuperTrend
            "st_trend_15m", "st_distance_15m", "st_bars_since_flip_15m", "st_slope_15m", "st_bandwidth_15m",
            # 时间/Session
            "session", "hour_of_day", "minute_of_hour", "day_of_week",
            "bars_from_session_open", "is_session_open", "is_session_close", "hour_sin", "hour_cos",
            # 交叉特征
            "trend_fvg_alignment", "trend_strength_x_fvg", "trend_duration_x_vol",
            "trend_fvg_distance", "session_trend_interaction",
            # ST 对齐
            "st_alignment",
            # ATR 相关
            "atr_ratio_1m_15m", "atr_1m", "atr_15m", "mid_15m_range",
            # 波动率（用于交叉）
            # 波动率（用于交叉）
            "micro_volatility_pct",
            # 热力学
            "market_temperature", "market_entropy", "ts_phase",
        ]:
            try:
                self._idx[name] = self.schema.get_index(name)
            except Exception:
                pass
        
        logger.info(
            "FeaturePipelineV4 initialized",
            n_features=self.schema.num_features,
            schema_hash=self.schema.schema_hash,
        )
    
    def compute_batch(
        self,
        bars: Sequence[EventBar],
        supertrend_directions: Sequence[int] | None = None,
        supertrend_lines: Sequence[float] | None = None,
        supertrend_durations: Sequence[int] | None = None,
    ) -> FeatureResult:
        """
        批量计算特征（训练模式）
        
        Args:
            bars: EventBar 序列
            supertrend_directions: 趋势方向序列 (-1, 0, 1)
            supertrend_lines: SuperTrend 线序列
            supertrend_durations: 趋势持续 Bar 数序列
            
        Returns:
            FeatureResult 包含特征矩阵
        """
        n = len(bars)
        n_features = self.schema.num_features
        
        # 初始化特征矩阵
        features = np.zeros((n, n_features), dtype=np.float32)
        is_valid = np.ones(n, dtype=np.bool_)
        
        # 提取 OHLC 和元数据
        closes = np.array([b.close for b in bars], dtype=np.float64)
        highs = np.array([b.high for b in bars], dtype=np.float64)
        lows = np.array([b.low for b in bars], dtype=np.float64)
        tick_counts = np.array([b.tick_count for b in bars], dtype=np.float64)
        durations = np.array([b.duration_ms for b in bars], dtype=np.float64)
        imbalances = np.array([b.imbalance for b in bars], dtype=np.float64)
        buy_counts = np.array([b.buy_count for b in bars], dtype=np.float64)
        sell_counts = np.array([b.sell_count for b in bars], dtype=np.float64)
        spread_sums = np.array([b.spread_sum for b in bars], dtype=np.float64)
        
        # === 1. 价格特征 ===
        log_returns = np.zeros(n, dtype=np.float64)
        log_returns[1:] = np.log(closes[1:] / closes[:-1])
        
        log_return_zscore = self._compute_rolling_zscore(
            log_returns, self.config.zscore_window
        )
        
        avg_spreads = spread_sums / np.maximum(tick_counts, 1)
        spread_bps = avg_spreads / closes * 10000
        
        features[:, self.schema.get_index("log_return")] = log_returns.astype(np.float32)
        features[:, self.schema.get_index("log_return_zscore")] = log_return_zscore.astype(np.float32)
        features[:, self.schema.get_index("spread_bps")] = spread_bps.astype(np.float32)
        
        # === 2. 微观结构特征 ===
        # delta_t_log
        delta_t_ms = np.maximum(durations, 1.0)
        delta_t_log = np.log(delta_t_ms + EPSILON)
        features[:, self.schema.get_index("delta_t_log")] = delta_t_log.astype(np.float32)
        
        # tick_intensity (EWMA of 1/ln(delta_t))
        tick_intensity = self._compute_ewma(
            1.0 / np.maximum(delta_t_log, 0.1),
            self.config.tick_intensity_alpha
        )
        features[:, self.schema.get_index("tick_intensity")] = tick_intensity.astype(np.float32)
        
        # OFI (Order Flow Imbalance)
        ofi_count = self._compute_rolling_sum(
            imbalances, self.config.ofi_window
        )
        features[:, self.schema.get_index("ofi_count")] = ofi_count.astype(np.float32)
        
        # OFI weighted by intensity
        weighted_imb = imbalances / np.maximum(delta_t_ms, 1.0)
        ofi_weighted = self._compute_rolling_sum(
            weighted_imb, self.config.ofi_window
        )
        features[:, self.schema.get_index("ofi_weighted")] = ofi_weighted.astype(np.float32)
        
        # Kyle's Lambda (pct)
        kyle_lambda = self._compute_kyle_lambda(
            closes, tick_counts, self.config.kyle_lambda_window
        )
        features[:, self.schema.get_index("kyle_lambda_pct")] = kyle_lambda.astype(np.float32)
        
        # PDI
        pdi = np.sign(log_returns) * tick_intensity
        features[:, self.schema.get_index("pdi")] = pdi.astype(np.float32)
        
        # === 3. 波动率特征 ===
        # 微观波动率 (EWMA)
        micro_vol = self._compute_ewma_volatility(
            log_returns * 100,  # 转换为百分比
            self.config.micro_vol_lambda
        )
        features[:, self.schema.get_index("micro_volatility_pct")] = micro_vol.astype(np.float32)
        
        # 已实现波动率
        realized_vol = self._compute_realized_volatility(
            log_returns * 100, self.config.volatility_window
        )
        features[:, self.schema.get_index("realized_volatility_pct")] = realized_vol.astype(np.float32)
        
        # === 4. 热力学特征 ===
        temp, entropy, ts_phase = self._compute_thermodynamics(
            log_returns, delta_t_ms, imbalances, tick_counts,
            self.config.thermo_window
        )
        features[:, self.schema.get_index("market_temperature")] = temp.astype(np.float32)
        features[:, self.schema.get_index("market_entropy")] = entropy.astype(np.float32)
        features[:, self.schema.get_index("ts_phase")] = ts_phase.astype(np.float32)
        
        # === 5. 订单流特征 (VPIN) ===
        vpin = self._vpin_calc.compute_batch(closes, tick_counts)
        vpin_zscore = self._compute_rolling_zscore(vpin, self.config.zscore_window)
        features[:, self.schema.get_index("vpin")] = vpin.astype(np.float32)
        features[:, self.schema.get_index("vpin_zscore")] = vpin_zscore.astype(np.float32)
        
        # === 6. 趋势特征 ===
        if supertrend_directions is not None:
            directions = np.array(supertrend_directions, dtype=np.int32)
        else:
            directions = np.zeros(n, dtype=np.int32)
        
        if supertrend_lines is not None:
            st_lines = np.array(supertrend_lines, dtype=np.float64)
            # trend_deviation = (close - supertrend) / close * 100
            trend_deviation = (closes - st_lines) / closes * 100
        else:
            trend_deviation = np.zeros(n, dtype=np.float64)
        
        if supertrend_durations is not None:
            durations_arr = np.array(supertrend_durations, dtype=np.int32)
        else:
            durations_arr = np.zeros(n, dtype=np.int32)
        
        features[:, self.schema.get_index("trend_deviation")] = trend_deviation.astype(np.float32)
        features[:, self.schema.get_index("trend_direction")] = directions.astype(np.float32)
        features[:, self.schema.get_index("trend_duration")] = durations_arr.astype(np.float32)
        
        # === 7. 降噪特征 ===
        if self._denoise_pipeline is not None:
            denoise_result = self._denoise_pipeline.denoise_batch(closes)
            
            if denoise_result.kalman_residual_bps is not None:
                features[:, self.schema.get_index("kalman_residual_bps")] = (
                    denoise_result.kalman_residual_bps.astype(np.float32)
                )
            
            if denoise_result.kalman_gain is not None:
                features[:, self.schema.get_index("kalman_gain")] = (
                    denoise_result.kalman_gain.astype(np.float32)
                )
            
            # 小波能量
            if 1 in denoise_result.wavelet_energy:
                features[:, self.schema.get_index("wavelet_energy_1")] = (
                    denoise_result.wavelet_energy[1].astype(np.float32)
                )
            if 2 in denoise_result.wavelet_energy:
                features[:, self.schema.get_index("wavelet_energy_2")] = (
                    denoise_result.wavelet_energy[2].astype(np.float32)
                )
            
            if denoise_result.wavelet_trend_ratio is not None:
                features[:, self.schema.get_index("wavelet_trend_ratio")] = (
                    denoise_result.wavelet_trend_ratio.astype(np.float32)
                )
        
        # === 裁剪 Z-Score 类特征 ===
        if self.config.clip_zscore:
            clip_val = self.config.zscore_clip_value
            zscore_features = ["log_return_zscore", "vpin_zscore"]
            for name in zscore_features:
                idx = self.schema.get_index(name)
                features[:, idx] = np.clip(features[:, idx], -clip_val, clip_val)
        
        # === 标记无效样本（NaN/Inf）===
        for i in range(n):
            if np.any(np.isnan(features[i])) or np.any(np.isinf(features[i])):
                is_valid[i] = False
        
        # === 8. ML 友好扩展特征：FVG / FVG_event / 15m ST / 时间 / 交叉 / ATR ===
        # 目标：
        # - 将 FVG 建模为离散事件（fvg_event），并提供严格因果的 follow-through 路径状态
        # - 15m SuperTrend 仅使用已完成的 15m bar（避免未来信息泄漏）
        # - 为 XGB 提供独立的 regime 特征（位置/对齐/时间/ATR 比率）
        try:
            # 1) 计算 1m ATR 序列（用于归一化与冲击强度）
            atr_1m_arr = self._compute_atr_1m_series(bars, period=self._atr_1m_period)
            
            # 2) 15m 聚合 + 对齐（mid-range + ATR_15m + SuperTrend_15m）
            agg_15m = TimeBarAggregator(config=TimeBarAggregatorConfig(interval_seconds=900))
            bars_15m = agg_15m.aggregate_batch(bars)
            
            if bars_15m:
                align_idx = align_features_to_lower_timeframe(bars_15m, list(bars))
                
                mid_15m_arr = np.zeros(n, dtype=np.float64)
                atr_15m_arr = np.ones(n, dtype=np.float64)
                for i, j in enumerate(align_idx):
                    if 0 <= j < len(bars_15m):
                        tb = bars_15m[j]
                        mid_15m_arr[i] = (tb.high + tb.low) / 2
                        atr_15m_arr[i] = max(tb.high - tb.low, EPSILON)
                
                st_15m_calc = SuperTrendFeatureCalculator()
                st_15m = st_15m_calc.compute_batch(bars_15m)  # (n_15m, 5)
                st_15m_aligned = np.zeros((n, 5), dtype=np.float32)
                for i, j in enumerate(align_idx):
                    if 0 <= j < len(st_15m):
                        st_15m_aligned[i] = st_15m[j]
            else:
                mid_15m_arr = np.zeros(n, dtype=np.float64)
                atr_15m_arr = np.ones(n, dtype=np.float64)
                st_15m_aligned = np.zeros((n, 5), dtype=np.float32)
            
            # 3) FVG（活跃状态）
            fvg_feat = FVGFeatureCalculator().compute_batch(bars)  # (n, 8)
            
            # 4) FVG_event（事件型 + 因果跟随）
            fvg_event_feat = FVGEventCalculator().compute_batch(
                bars,
                mid_15m_arr=mid_15m_arr,
                atr_1m_arr=atr_1m_arr,
                atr_15m_arr=atr_15m_arr,
            )  # (n, 7)
            
            # 5) 时间特征
            time_feat = TimeFeatureCalculator().compute_batch(bars)  # (n, 9)
            
            # 6) 交叉特征（15m ST × FVG 活跃状态 × 时间）
            micro_vol = features[:, self.schema.get_index("micro_volatility_pct")].astype(np.float32)
            interaction_feat = compute_interaction_features_batch(
                st_features=st_15m_aligned,
                fvg_features=fvg_feat,
                time_features=time_feat,
                volatility=micro_vol,
            )  # (n, 5)
            
            # 7) ST 对齐（顺/逆趋势显式）
            st_alignment = (
                st_15m_aligned[:, 0].astype(np.int32) * fvg_event_feat[:, 0].astype(np.int32)
            ).astype(np.float32)
            
            # 8) ATR 比率
            atr_ratio = (atr_1m_arr / np.maximum(atr_15m_arr, EPSILON)).astype(np.float32)
            
            # 9) 写回 schema 对应字段（缺失字段则跳过）
            def _write(name: str, col: NDArray) -> None:
                try:
                    idx = self.schema.get_index(name)
                    features[:, idx] = col.astype(np.float32)
                except Exception:
                    return
            
            # FVG（8）
            for k, name in enumerate([
                "bullish_fvg", "bearish_fvg", "fvg_size_atr", "fvg_age_bars",
                "price_to_fvg_mid", "fvg_filled_pct", "active_fvg_count", "nearest_fvg_distance",
            ]):
                _write(name, fvg_feat[:, k])
            
            # FVG_event（7）
            for k, name in enumerate([
                "fvg_event", "fvg_impulse_atr", "fvg_location_15m",
                "fvg_follow_up_3", "fvg_follow_dn_3", "fvg_follow_bars", "fvg_follow_net",
            ]):
                _write(name, fvg_event_feat[:, k])
            
            # 15m SuperTrend（5）
            for k, name in enumerate([
                "st_trend_15m", "st_distance_15m", "st_bars_since_flip_15m", "st_slope_15m", "st_bandwidth_15m",
            ]):
                _write(name, st_15m_aligned[:, k])
            
            # 时间（9）
            for k, name in enumerate([
                "session", "hour_of_day", "minute_of_hour", "day_of_week",
                "bars_from_session_open", "is_session_open", "is_session_close", "hour_sin", "hour_cos",
            ]):
                _write(name, time_feat[:, k])
            
            # 交叉（5）
            for k, name in enumerate([
                "trend_fvg_alignment", "trend_strength_x_fvg", "trend_duration_x_vol",
                "trend_fvg_distance", "session_trend_interaction",
            ]):
                _write(name, interaction_feat[:, k])
            
            _write("st_alignment", st_alignment)
            _write("atr_ratio_1m_15m", atr_ratio)
            _write("atr_1m", atr_1m_arr.astype(np.float32))
            _write("atr_15m", atr_15m_arr.astype(np.float32))
            _write("mid_15m_range", mid_15m_arr.astype(np.float32))
        
        except Exception as e:
            # 向后兼容：扩展特征失败时不阻塞主流程，但必须可见（避免 silent 0 特征）
            logger.warning(f"ML extension features computation skipped: {e}")
        
        return FeatureResult(
            features=features,
            schema=self.schema,
            is_valid=is_valid,
            n_samples=n,
        )
    
    def update(
        self,
        bar: EventBar,
        trend_direction: int = 0,
        supertrend_line: float = 0.0,
        trend_duration: int = 0,
    ) -> NDArray[np.float32]:
        """
        流式更新（单 bar，推理模式）
        
        Args:
            bar: 当前完成的 EventBar
            trend_direction: 当前趋势方向
            supertrend_line: 当前 SuperTrend 线
            trend_duration: 趋势持续 Bar 数
            
        Returns:
            特征向量 (n_features,)
        """
        self._bar_count += 1
        
        # 更新趋势状态
        self._trend_direction = trend_direction
        self._supertrend_line = supertrend_line
        self._trend_duration = trend_duration
        
        # 提取当前 bar 数据
        close = bar.close
        high = bar.high
        low = bar.low
        tick_count = bar.tick_count
        duration_ms = max(bar.duration_ms, 1.0)
        imbalance = bar.imbalance
        spread_avg = bar.spread_sum / max(tick_count, 1)
        
        # 计算对数收益率
        if self._last_close > 0:
            log_return = math.log(close / self._last_close)
        else:
            log_return = 0.0
        
        # 时间间隔（微秒）
        if hasattr(bar, 'close_time') and bar.close_time:
            time_us = int(bar.close_time.timestamp() * 1e6)
        else:
            time_us = self._last_time_us + int(duration_ms * 1000)
        
        if self._last_time_us > 0:
            delta_t_us = time_us - self._last_time_us
        else:
            delta_t_us = int(duration_ms * 1000)
        
        # 更新缓冲
        self._price_buffer.append(close)
        self._return_buffer.append(log_return)
        self._delta_t_buffer.append(duration_ms)
        self._direction_buffer.append(imbalance)
        self._tick_count_buffer.append(tick_count)
        
        # 保持缓冲大小
        max_buffer = max(
            self.config.zscore_window,
            self.config.volatility_window,
            self.config.ofi_window,
            self.config.thermo_window,
        ) + 10
        
        if len(self._price_buffer) > max_buffer:
            self._price_buffer.pop(0)
            self._return_buffer.pop(0)
            self._delta_t_buffer.pop(0)
            self._direction_buffer.pop(0)
            self._tick_count_buffer.pop(0)
        
        # 初始化特征向量
        features = self.schema.fill_defaults()
        
        # === 计算特征 ===
        
        # 价格特征
        features[self.schema.get_index("log_return")] = log_return
        
        if len(self._return_buffer) > 1:
            returns = np.array(self._return_buffer)
            mean = np.mean(returns)
            std = np.std(returns)
            if std > EPSILON:
                zscore = (log_return - mean) / std
            else:
                zscore = 0.0
            features[self.schema.get_index("log_return_zscore")] = np.clip(
                zscore, -5.0, 5.0
            )
        
        features[self.schema.get_index("spread_bps")] = spread_avg / close * 10000
        
        # 微观结构特征
        delta_t_log = math.log(duration_ms + EPSILON)
        features[self.schema.get_index("delta_t_log")] = delta_t_log
        
        # Tick intensity EWMA
        alpha = self.config.tick_intensity_alpha
        inv_dt = 1.0 / max(delta_t_log, 0.1)
        self._tick_intensity = alpha * inv_dt + (1 - alpha) * self._tick_intensity
        features[self.schema.get_index("tick_intensity")] = self._tick_intensity
        
        # OFI
        if len(self._direction_buffer) >= self.config.ofi_window:
            ofi_count = sum(self._direction_buffer[-self.config.ofi_window:])
            features[self.schema.get_index("ofi_count")] = ofi_count
        
        # PDI
        pdi = math.copysign(1, log_return) * self._tick_intensity if log_return != 0 else 0
        features[self.schema.get_index("pdi")] = pdi
        
        # 波动率
        lmbda = self.config.micro_vol_lambda
        return_pct = log_return * 100
        self._micro_vol_sq = (1 - lmbda) * (return_pct ** 2) + lmbda * self._micro_vol_sq
        features[self.schema.get_index("micro_volatility_pct")] = math.sqrt(self._micro_vol_sq)
        
        if len(self._return_buffer) >= self.config.volatility_window:
            returns = np.array(self._return_buffer[-self.config.volatility_window:]) * 100
            realized_vol = math.sqrt(np.sum(returns ** 2))
            features[self.schema.get_index("realized_volatility_pct")] = realized_vol
        
        # 热力学特征（流式计算，保持与 batch 口径一致）
        if len(self._return_buffer) >= self.config.thermo_window:
            window = self.config.thermo_window
            window_returns = np.array(self._return_buffer[-window:], dtype=np.float64)
            window_dt = np.array(self._delta_t_buffer[-window:], dtype=np.float64)
            
            temperature, entropy = self._compute_thermo_window(window_returns, window_dt)
            ts_phase = self._classify_phase(temperature)
            
            if "market_temperature" in self._idx:
                features[self._idx["market_temperature"]] = temperature
            if "market_entropy" in self._idx:
                features[self._idx["market_entropy"]] = entropy
            if "ts_phase" in self._idx:
                features[self._idx["ts_phase"]] = float(ts_phase)
        
        # VPIN
        vpin = self._vpin_calc.update(close, tick_count)
        features[self.schema.get_index("vpin")] = vpin
        
        # 趋势特征
        features[self.schema.get_index("trend_direction")] = trend_direction
        features[self.schema.get_index("trend_duration")] = trend_duration
        if supertrend_line > 0:
            trend_dev = (close - supertrend_line) / close * 100
            features[self.schema.get_index("trend_deviation")] = trend_dev
        
        # 降噪特征（如果启用）
        if self._denoise_pipeline is not None:
            kalman_state = self._denoise_pipeline.update(close)
            if kalman_state:
                features[self.schema.get_index("kalman_residual_bps")] = kalman_state.residual_bps
                features[self.schema.get_index("kalman_gain")] = kalman_state.kalman_gain
        
        # === ML 友好扩展特征（严格因果）===
        # 说明：
        # - 15m 相关特征仅使用“已完成”的 15m bar（避免未来信息泄漏）
        # - FVG_event / follow-through 是事件锚定、滚动更新的路径状态（避免固定未来窗口）
        try:
            # 1) 更新 1m ATR（用于冲击强度/归一化）
            tr = (high - low) if self._last_close <= 0 else max(
                high - low,
                abs(high - self._last_close),
                abs(low - self._last_close),
            )
            self._atr_1m_buffer.append(float(tr))
            if len(self._atr_1m_buffer) > self._atr_1m_period:
                self._atr_1m_buffer = self._atr_1m_buffer[-self._atr_1m_period:]
            self._atr_1m_current = float(sum(self._atr_1m_buffer) / max(1, len(self._atr_1m_buffer)))
            atr_1m = max(self._atr_1m_current, EPSILON)
            
            # 2) 更新 15m 聚合器（只有完成 bar 才更新 15m ST）
            if self._time_agg_15m is not None:
                tb = self._time_agg_15m.update(bar)
                if tb is not None:
                    self._last_mid_15m_range = float((tb.high + tb.low) / 2)
                    self._last_atr_15m = float(max(tb.high - tb.low, EPSILON))
                    if self._st_15m_calc is not None:
                        self._last_st_15m = self._st_15m_calc.update(tb).to_array().astype(np.float32)
            
            atr_15m = max(self._last_atr_15m, EPSILON)
            mid_15m = self._last_mid_15m_range
            
            # 3) FVG（活跃状态）
            if self._fvg_calc is not None:
                fvg_arr = self._fvg_calc.update(bar).to_array()
                for k, name in enumerate([
                    "bullish_fvg", "bearish_fvg", "fvg_size_atr", "fvg_age_bars",
                    "price_to_fvg_mid", "fvg_filled_pct", "active_fvg_count", "nearest_fvg_distance",
                ]):
                    if name in self._idx:
                        features[self._idx[name]] = fvg_arr[k]
            
            # 4) FVG_event（事件型 + 因果跟随）
            if self._fvg_event_calc is not None:
                evt_arr = self._fvg_event_calc.update(
                    bar,
                    mid_15m=mid_15m,
                    atr_1m=atr_1m,
                    atr_15m=atr_15m,
                ).to_array()
                for k, name in enumerate([
                    "fvg_event", "fvg_impulse_atr", "fvg_location_15m",
                    "fvg_follow_up_3", "fvg_follow_dn_3", "fvg_follow_bars", "fvg_follow_net",
                ]):
                    if name in self._idx:
                        features[self._idx[name]] = evt_arr[k]
            
            # 5) 15m SuperTrend（使用最后一个已完成 15m bar 的特征）
            if self._last_st_15m is not None:
                for k, name in enumerate([
                    "st_trend_15m", "st_distance_15m", "st_bars_since_flip_15m", "st_slope_15m", "st_bandwidth_15m",
                ]):
                    if name in self._idx:
                        features[self._idx[name]] = self._last_st_15m[k]
            
            # 6) 时间特征
            if self._time_calc is not None:
                t_arr = self._time_calc.compute(bar.time).to_array()
                for k, name in enumerate([
                    "session", "hour_of_day", "minute_of_hour", "day_of_week",
                    "bars_from_session_open", "is_session_open", "is_session_close", "hour_sin", "hour_cos",
                ]):
                    if name in self._idx:
                        features[self._idx[name]] = t_arr[k]
            
            # 7) ATR 特征
            if "atr_1m" in self._idx:
                features[self._idx["atr_1m"]] = float(atr_1m)
            if "atr_15m" in self._idx:
                features[self._idx["atr_15m"]] = float(atr_15m)
            if "mid_15m_range" in self._idx:
                features[self._idx["mid_15m_range"]] = float(mid_15m)
            if "atr_ratio_1m_15m" in self._idx:
                features[self._idx["atr_ratio_1m_15m"]] = float(atr_1m / atr_15m)
            
            # 8) 交叉特征（在线计算）
            st_trend_15m = int(features[self._idx.get("st_trend_15m", -1)]) if "st_trend_15m" in self._idx else 0
            st_distance_15m = float(features[self._idx.get("st_distance_15m", -1)]) if "st_distance_15m" in self._idx else 0.0
            st_bars_since_flip_15m = int(features[self._idx.get("st_bars_since_flip_15m", -1)]) if "st_bars_since_flip_15m" in self._idx else 0
            
            bullish = int(features[self._idx.get("bullish_fvg", -1)]) if "bullish_fvg" in self._idx else 0
            bearish = int(features[self._idx.get("bearish_fvg", -1)]) if "bearish_fvg" in self._idx else 0
            fvg_size_atr = float(features[self._idx.get("fvg_size_atr", -1)]) if "fvg_size_atr" in self._idx else 0.0
            price_to_mid = float(features[self._idx.get("price_to_fvg_mid", -1)]) if "price_to_fvg_mid" in self._idx else 0.0
            micro_vol_now = float(features[self._idx.get("micro_volatility_pct", -1)]) if "micro_volatility_pct" in self._idx else 0.0
            session = float(features[self._idx.get("session", -1)]) if "session" in self._idx else 0.0
            
            if "trend_fvg_alignment" in self._idx:
                features[self._idx["trend_fvg_alignment"]] = float(st_trend_15m * (bullish - bearish))
            if "trend_strength_x_fvg" in self._idx:
                features[self._idx["trend_strength_x_fvg"]] = float(abs(st_distance_15m) * fvg_size_atr)
            if "trend_duration_x_vol" in self._idx:
                features[self._idx["trend_duration_x_vol"]] = float(st_bars_since_flip_15m * micro_vol_now)
            if "trend_fvg_distance" in self._idx:
                features[self._idx["trend_fvg_distance"]] = float(st_distance_15m * price_to_mid)
            if "session_trend_interaction" in self._idx:
                features[self._idx["session_trend_interaction"]] = float(session * st_trend_15m)
            
            # 9) ST 对齐（顺/逆趋势显式）
            if "st_alignment" in self._idx and "fvg_event" in self._idx:
                fvg_event = int(features[self._idx["fvg_event"]])
                features[self._idx["st_alignment"]] = float(st_trend_15m * fvg_event)
        
        except Exception as e:
            logger.warning(f"ML extension features update skipped: {e}")
        
        # 更新状态
        self._last_close = close
        self._last_time_us = time_us
        
        return features
    
    # === 辅助计算函数 ===
    
    def _compute_rolling_zscore(
        self,
        values: NDArray[np.float64],
        window: int,
    ) -> NDArray[np.float64]:
        """计算滚动 Z-Score"""
        n = len(values)
        result = np.zeros(n, dtype=np.float64)
        
        for i in range(n):
            start = max(0, i - window + 1)
            if i - start > 1:
                window_vals = values[start:i+1]
                mean = np.mean(window_vals)
                std = np.std(window_vals)
                if std > EPSILON:
                    result[i] = (values[i] - mean) / std
        
        return result
    
    def _compute_ewma(
        self,
        values: NDArray[np.float64],
        alpha: float,
    ) -> NDArray[np.float64]:
        """计算 EWMA"""
        n = len(values)
        result = np.zeros(n, dtype=np.float64)
        result[0] = values[0]
        
        for i in range(1, n):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        
        return result
    
    def _compute_rolling_sum(
        self,
        values: NDArray[np.float64],
        window: int,
    ) -> NDArray[np.float64]:
        """计算滚动和"""
        n = len(values)
        result = np.zeros(n, dtype=np.float64)
        
        for i in range(n):
            start = max(0, i - window + 1)
            result[i] = np.sum(values[start:i+1])
        
        return result
    
    def _compute_kyle_lambda(
        self,
        closes: NDArray[np.float64],
        tick_counts: NDArray[np.float64],
        window: int,
    ) -> NDArray[np.float64]:
        """计算 Kyle's Lambda（百分比）"""
        n = len(closes)
        result = np.zeros(n, dtype=np.float64)
        
        for i in range(window, n):
            start_price = closes[i - window]
            end_price = closes[i]
            price_change_pct = abs(end_price - start_price) / start_price * 100
            total_ticks = np.sum(tick_counts[i-window+1:i+1])
            
            if total_ticks > 0:
                result[i] = price_change_pct / total_ticks
        
        return result
    
    def _compute_ewma_volatility(
        self,
        returns_pct: NDArray[np.float64],
        lmbda: float,
    ) -> NDArray[np.float64]:
        """计算 EWMA 波动率"""
        n = len(returns_pct)
        variance = np.zeros(n, dtype=np.float64)
        variance[0] = returns_pct[0] ** 2
        
        for i in range(1, n):
            variance[i] = (1 - lmbda) * (returns_pct[i] ** 2) + lmbda * variance[i-1]
        
        return np.sqrt(variance)
    
    def _compute_realized_volatility(
        self,
        returns_pct: NDArray[np.float64],
        window: int,
    ) -> NDArray[np.float64]:
        """计算已实现波动率"""
        n = len(returns_pct)
        result = np.zeros(n, dtype=np.float64)
        
        for i in range(window, n):
            window_returns = returns_pct[i-window+1:i+1]
            result[i] = np.sqrt(np.sum(window_returns ** 2))
        
        return result
    
    def _compute_thermo_window(
        self,
        returns_window: NDArray[np.float64],
        delta_t_window: NDArray[np.float64],
    ) -> tuple[float, float]:
        """计算单窗口热力学值（SSOT: log return）。"""
        var_returns = float(np.var(returns_window))
        mean_dt = float(np.mean(delta_t_window))
        if mean_dt > EPSILON:
            temperature = var_returns / mean_dt * self.config.thermodynamics.scale
        else:
            temperature = 0.0
        
        up = np.sum(returns_window > 0)
        down = np.sum(returns_window < 0)
        total = max(int(up + down), 1)
        p_up = float(up) / total
        p_down = float(down) / total
        probs = np.array([p_up, p_down], dtype=np.float64)
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log(probs + EPSILON))) if probs.size else 0.0
        
        return temperature, entropy
    
    def _classify_phase(self, temperature: float) -> int:
        """基于温度阈值分类相位（0=frozen,1=laminar,2=turbulent,3=transition）。"""
        cfg = self.config.thermodynamics
        if temperature < cfg.temperature_frozen:
            return 0
        if temperature < cfg.temperature_laminar:
            return 1
        if temperature < cfg.temperature_turbulent:
            return 3
        return 2
    
    def _compute_thermodynamics(
        self,
        returns: NDArray[np.float64],
        delta_t_ms: NDArray[np.float64],
        imbalances: NDArray[np.float64],
        tick_counts: NDArray[np.float64],
        window: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]]:
        """计算热力学指标"""
        n = len(returns)
        temperature = np.zeros(n, dtype=np.float64)
        entropy = np.zeros(n, dtype=np.float64)
        ts_phase = np.zeros(n, dtype=np.int32)
        
        for i in range(window, n):
            window_returns = returns[i-window+1:i+1]
            window_dt = delta_t_ms[i-window+1:i+1]
            temp, ent = self._compute_thermo_window(window_returns, window_dt)
            temperature[i] = temp
            entropy[i] = ent
            ts_phase[i] = self._classify_phase(temp)
        
        return temperature, entropy, ts_phase
    
    def _compute_atr_1m_series(
        self,
        bars: Sequence[EventBar],
        period: int = 14,
    ) -> NDArray[np.float64]:
        """
        计算 1m ATR 序列（严格因果，TR 滚动均值）
        
        说明：
        - 该 ATR 仅用于特征归一化/事件强度，不作为交易规则
        - 每个时刻 t 只使用 t 及之前的 bar（不使用未来信息）
        """
        n = len(bars)
        atr = np.zeros(n, dtype=np.float64)
        
        tr_buf: list[float] = []
        prev_close: float | None = None
        
        for i, b in enumerate(bars):
            if prev_close is None:
                tr = b.high - b.low
            else:
                tr = max(
                    b.high - b.low,
                    abs(b.high - prev_close),
                    abs(b.low - prev_close),
                )
            
            tr_buf.append(float(tr))
            if len(tr_buf) > period:
                tr_buf = tr_buf[-period:]
            
            atr[i] = float(sum(tr_buf) / max(1, len(tr_buf)))
            prev_close = b.close
        
        return atr
    
    def reset(self) -> None:
        """重置管道状态"""
        self._bar_buffer.clear()
        self._price_buffer.clear()
        self._return_buffer.clear()
        self._delta_t_buffer.clear()
        self._direction_buffer.clear()
        self._tick_count_buffer.clear()
        
        self._tick_intensity = 1.0
        self._micro_vol_sq = 0.0
        self._return_mean = 0.0
        self._return_var = 1e-6
        
        self._trend_direction = 0
        self._trend_duration = 0
        self._supertrend_line = 0.0
        
        self._bar_count = 0
        self._last_close = 0.0
        self._last_time_us = 0
        
        if self._vpin_calc:
            self._vpin_calc.reset()
        if self._denoise_pipeline:
            self._denoise_pipeline.reset()
        
        # 重置扩展模块状态
        if self._fvg_calc:
            self._fvg_calc.reset()
        if self._fvg_event_calc:
            self._fvg_event_calc.reset()
        if self._time_calc:
            self._time_calc.reset()
        if self._time_agg_15m:
            self._time_agg_15m.reset()
        if self._st_15m_calc:
            self._st_15m_calc.reset()
        
        self._last_mid_15m_range = 0.0
        self._last_atr_15m = 0.0
        self._last_st_15m = np.zeros(5, dtype=np.float32)
        
        self._atr_1m_buffer.clear()
        self._atr_1m_current = 0.0
    
    def get_stats(self) -> dict:
        """获取管道统计"""
        return {
            "bar_count": self._bar_count,
            "buffer_size": len(self._price_buffer),
            "tick_intensity": round(self._tick_intensity, 4),
            "micro_volatility": round(math.sqrt(self._micro_vol_sq), 4),
            "schema_hash": self.schema.schema_hash,
            "n_features": self.schema.num_features,
        }
