"""
v4.0 降噪管道

整合 MODWT 小波变换和卡尔曼滤波：
- MODWT：用于训练数据的批量降噪（保留原始序列对照）
- Kalman：用于实时推理的在线降噪
- 组合模式：先 MODWT 再 Kalman

关键输出特征：
- denoised_price: 降噪后的价格序列
- kalman_residual: 卡尔曼残差（噪声水平指示）
- kalman_gain: 卡尔曼增益（自适应学习率）
- wavelet_energy_*: 各频率层能量（市场状态指示）
- wavelet_trend_ratio: 趋势成分占比

参考：降噪LNN特征提取与信号过滤.md Section 3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from alphaos.core.config import DenoiseConfig as CoreDenoiseConfig
from numpy.typing import NDArray

from alphaos.core.logging import get_logger
from alphaos.features.denoise.wavelet import WaveletDenoiser, compute_wavelet_features
from alphaos.features.denoise.kalman import (
    KalmanFilter,
    AdaptiveKalmanFilter,
    KalmanState,
    compute_kalman_batch,
)

logger = get_logger(__name__)


class DenoiseMode(Enum):
    """降噪模式"""
    NONE = "none"                  # 不降噪
    WAVELET = "wavelet"            # 仅 MODWT 小波（离线）
    KALMAN = "kalman"              # 仅卡尔曼（在线）
    ADAPTIVE_KALMAN = "adaptive"   # 自适应卡尔曼（在线）
    COMBINED = "combined"          # MODWT + Kalman（离线）


@dataclass
class DenoiseConfig:
    """
    降噪配置
    
    Args:
        mode: 降噪模式
        
        # 小波参数
        wavelet: 小波基函数（默认 db4）
        wavelet_level: 分解层数（None=自动）
        threshold_mode: 阈值模式（soft/hard）
        threshold_rule: 阈值规则（universal/minimax）
        
        # 卡尔曼参数
        process_variance: 状态转移噪声方差 Q
        measurement_variance: 测量噪声方差 R
        initial_uncertainty: 初始估计方差 P0
        
        # 自适应卡尔曼参数
        adaptation_rate: 噪声参数适应速率
        innovation_window: 创新序列窗口长度
        
        # 通用参数
        preserve_original: 是否保留原始序列（用于对照诊断）
    """
    mode: DenoiseMode = DenoiseMode.ADAPTIVE_KALMAN
    
    # 小波参数
    wavelet: str = "db4"
    wavelet_level: int | None = None
    threshold_mode: Literal["soft", "hard"] = "soft"
    threshold_rule: Literal["universal", "minimax"] = "universal"
    
    # 卡尔曼参数
    process_variance: float = 0.1
    measurement_variance: float = 1.0
    initial_uncertainty: float = 100.0
    
    # 自适应卡尔曼参数
    adaptation_rate: float = 0.1
    innovation_window: int = 20
    
    # 通用参数
    preserve_original: bool = True
    
    @classmethod
    def from_core_config(cls, core_cfg: "CoreDenoiseConfig") -> "DenoiseConfig":
        """
        Create DenoiseConfig from alphaos.core.config.DenoiseConfig (pydantic).
        
        This adapter bridges the two config systems, allowing callers using the
        pydantic config to work with the v4 denoise pipeline.
        
        Args:
            core_cfg: alphaos.core.config.DenoiseConfig instance
            
        Returns:
            DenoiseConfig instance for use with DenoisePipeline
        """
        # Determine mode based on which filters are enabled
        kalman_enabled = core_cfg.kalman.enabled
        wavelet_enabled = core_cfg.wavelet.enabled
        use_adaptive = core_cfg.kalman.use_adaptive
        
        if kalman_enabled and wavelet_enabled:
            mode = DenoiseMode.COMBINED
        elif kalman_enabled:
            mode = DenoiseMode.ADAPTIVE_KALMAN if use_adaptive else DenoiseMode.KALMAN
        elif wavelet_enabled:
            mode = DenoiseMode.WAVELET
        else:
            mode = DenoiseMode.NONE
        
        return cls(
            mode=mode,
            # Wavelet params from core config
            wavelet=core_cfg.wavelet.wavelet,
            wavelet_level=core_cfg.wavelet.level,
            threshold_mode=core_cfg.wavelet.threshold_mode,  # type: ignore
            threshold_rule=core_cfg.wavelet.threshold_rule,  # type: ignore
            # Kalman params from core config
            process_variance=core_cfg.kalman.process_variance,
            measurement_variance=core_cfg.kalman.measurement_variance,
            initial_uncertainty=core_cfg.kalman.initial_uncertainty,
            # Adaptive params (use defaults as core config doesn't have these)
            adaptation_rate=0.1,
            innovation_window=20,
            preserve_original=True,
        )
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "mode": self.mode.value,
            "wavelet": self.wavelet,
            "wavelet_level": self.wavelet_level,
            "threshold_mode": self.threshold_mode,
            "threshold_rule": self.threshold_rule,
            "process_variance": self.process_variance,
            "measurement_variance": self.measurement_variance,
            "initial_uncertainty": self.initial_uncertainty,
            "adaptation_rate": self.adaptation_rate,
            "innovation_window": self.innovation_window,
            "preserve_original": self.preserve_original,
        }


@dataclass
class DenoiseResult:
    """
    降噪结果
    
    Attributes:
        denoised: 降噪后的价格序列
        original: 原始价格序列（如果 preserve_original=True）
        
        # 卡尔曼输出
        kalman_residual: 卡尔曼残差序列
        kalman_residual_bps: 卡尔曼残差（基点）
        kalman_gain: 卡尔曼增益序列
        kalman_uncertainty: 估计不确定性序列
        
        # 小波输出
        wavelet_trend: 趋势成分（低频近似）
        wavelet_energy: 各层能量 {level: energy_array}
        wavelet_trend_ratio: 趋势成分占比
        
        # 元信息
        config: 使用的配置
        n_samples: 样本数量
    """
    denoised: NDArray[np.float64]
    original: NDArray[np.float64] | None = None
    
    # 卡尔曼输出
    kalman_residual: NDArray[np.float64] | None = None
    kalman_residual_bps: NDArray[np.float64] | None = None
    kalman_gain: NDArray[np.float64] | None = None
    kalman_uncertainty: NDArray[np.float64] | None = None
    
    # 小波输出
    wavelet_trend: NDArray[np.float64] | None = None
    wavelet_energy: dict[int, NDArray[np.float64]] = field(default_factory=dict)
    wavelet_trend_ratio: NDArray[np.float64] | None = None
    
    # 元信息
    config: DenoiseConfig | None = None
    n_samples: int = 0
    
    def get_feature_dict(self) -> dict[str, NDArray[np.float64]]:
        """
        获取可用于特征计算的字典
        
        Returns:
            特征名称 -> 数组 的映射
        """
        features = {
            "denoised_price": self.denoised,
        }
        
        if self.original is not None:
            features["original_price"] = self.original
        
        if self.kalman_residual is not None:
            features["kalman_residual"] = self.kalman_residual
        
        if self.kalman_residual_bps is not None:
            features["kalman_residual_bps"] = self.kalman_residual_bps
        
        if self.kalman_gain is not None:
            features["kalman_gain"] = self.kalman_gain
        
        if self.kalman_uncertainty is not None:
            features["kalman_uncertainty"] = self.kalman_uncertainty
        
        if self.wavelet_trend is not None:
            features["wavelet_trend"] = self.wavelet_trend
        
        for level, energy in self.wavelet_energy.items():
            features[f"wavelet_energy_{level}"] = energy
        
        if self.wavelet_trend_ratio is not None:
            features["wavelet_trend_ratio"] = self.wavelet_trend_ratio
        
        return features


@dataclass
class DenoisePipeline:
    """
    降噪管道
    
    提供批量（离线）和流式（在线）两种处理模式。
    
    离线模式（训练）：
    ```python
    pipeline = DenoisePipeline(DenoiseConfig(mode=DenoiseMode.COMBINED))
    result = pipeline.denoise_batch(prices)
    features = result.get_feature_dict()
    ```
    
    在线模式（推理）：
    ```python
    pipeline = DenoisePipeline(DenoiseConfig(mode=DenoiseMode.ADAPTIVE_KALMAN))
    for price in price_stream:
        state = pipeline.update(price)
        denoised = state.price_estimate
        noise_level = abs(state.residual_bps)
    ```
    """
    config: DenoiseConfig = field(default_factory=DenoiseConfig)
    
    # 内部组件
    _wavelet_denoiser: WaveletDenoiser | None = field(default=None, init=False)
    _kalman_filter: KalmanFilter | AdaptiveKalmanFilter | None = field(
        default=None, init=False
    )
    _is_initialized: bool = field(default=False, init=False)
    _tick_count: int = field(default=0, init=False)
    
    def __post_init__(self) -> None:
        """初始化组件"""
        self._init_components()
    
    def _init_components(self) -> None:
        """根据配置初始化组件"""
        mode = self.config.mode
        
        # 初始化小波降噪器
        if mode in (DenoiseMode.WAVELET, DenoiseMode.COMBINED):
            self._wavelet_denoiser = WaveletDenoiser(
                wavelet=self.config.wavelet,
                level=self.config.wavelet_level,
                threshold_mode=self.config.threshold_mode,
                threshold_rule=self.config.threshold_rule,
            )
        
        # 初始化卡尔曼滤波器
        if mode == DenoiseMode.KALMAN:
            self._kalman_filter = KalmanFilter(
                process_variance=self.config.process_variance,
                measurement_variance=self.config.measurement_variance,
                initial_uncertainty=self.config.initial_uncertainty,
            )
        elif mode in (DenoiseMode.ADAPTIVE_KALMAN, DenoiseMode.COMBINED):
            self._kalman_filter = AdaptiveKalmanFilter(
                initial_Q=self.config.process_variance,
                initial_R=self.config.measurement_variance,
                adaptation_rate=self.config.adaptation_rate,
                innovation_window=self.config.innovation_window,
            )
        
        self._is_initialized = True
        logger.info(
            "DenoisePipeline initialized",
            mode=mode.value,
            has_wavelet=self._wavelet_denoiser is not None,
            has_kalman=self._kalman_filter is not None,
        )
    
    def denoise_batch(
        self, 
        prices: NDArray[np.float64],
    ) -> DenoiseResult:
        """
        批量降噪（离线/训练模式）
        
        Args:
            prices: 价格序列
            
        Returns:
            DenoiseResult 包含所有降噪输出
        """
        n = len(prices)
        original = prices.copy() if self.config.preserve_original else None
        
        mode = self.config.mode
        
        # 无降噪模式
        if mode == DenoiseMode.NONE:
            return DenoiseResult(
                denoised=prices.copy(),
                original=original,
                config=self.config,
                n_samples=n,
            )
        
        # 初始化结果
        denoised = prices.copy()
        wavelet_trend = None
        wavelet_energy = {}
        wavelet_trend_ratio = None
        kalman_residual = None
        kalman_residual_bps = None
        kalman_gain = None
        kalman_uncertainty = None
        
        # 小波降噪
        if mode in (DenoiseMode.WAVELET, DenoiseMode.COMBINED):
            # 使用现有的 compute_wavelet_features
            wavelet_result = compute_wavelet_features(
                prices,
                wavelet=self.config.wavelet,
                level=self.config.wavelet_level or 4,
            )
            
            denoised = wavelet_result.get("denoised", prices.copy())
            wavelet_trend = wavelet_result.get("trend")
            
            # 提取各层能量
            for key, value in wavelet_result.items():
                if key.startswith("energy_"):
                    level = int(key.split("_")[1])
                    wavelet_energy[level] = value
            
            # 计算趋势占比
            if wavelet_trend is not None:
                total_var = np.var(prices) + 1e-10
                trend_var = np.var(wavelet_trend)
                wavelet_trend_ratio = np.full(n, trend_var / total_var)
        
        # 卡尔曼滤波
        if mode in (DenoiseMode.KALMAN, DenoiseMode.ADAPTIVE_KALMAN, DenoiseMode.COMBINED):
            # 对降噪后（或原始）的价格进行卡尔曼滤波
            input_prices = denoised if mode == DenoiseMode.COMBINED else prices
            
            kalman_result = compute_kalman_batch(
                input_prices,
                process_variance=self.config.process_variance,
                measurement_variance=self.config.measurement_variance,
                initial_uncertainty=self.config.initial_uncertainty,
            )
            
            denoised = kalman_result["price_estimate"]
            kalman_residual = kalman_result["residual"]
            kalman_residual_bps = kalman_result["residual_bps"]
            kalman_gain = kalman_result["kalman_gain"]
            kalman_uncertainty = kalman_result["uncertainty"]
        
        return DenoiseResult(
            denoised=denoised,
            original=original,
            kalman_residual=kalman_residual,
            kalman_residual_bps=kalman_residual_bps,
            kalman_gain=kalman_gain,
            kalman_uncertainty=kalman_uncertainty,
            wavelet_trend=wavelet_trend,
            wavelet_energy=wavelet_energy,
            wavelet_trend_ratio=wavelet_trend_ratio,
            config=self.config,
            n_samples=n,
        )
    
    def update(self, price: float) -> KalmanState | None:
        """
        在线更新（实时推理模式）
        
        仅支持卡尔曼滤波模式。
        
        Args:
            price: 当前价格
            
        Returns:
            KalmanState 或 None（如果不支持在线模式）
        """
        if self._kalman_filter is None:
            logger.warning(
                "Online update not supported for mode",
                mode=self.config.mode.value,
            )
            return None
        
        self._tick_count += 1
        return self._kalman_filter.update(price)
    
    @property
    def current_estimate(self) -> float:
        """获取当前卡尔曼估计值"""
        if self._kalman_filter is None:
            return 0.0
        return self._kalman_filter.state_estimate
    
    @property
    def current_uncertainty(self) -> float:
        """获取当前估计不确定性"""
        if self._kalman_filter is None:
            return 0.0
        return self._kalman_filter.uncertainty
    
    def reset(self) -> None:
        """重置管道状态"""
        if self._kalman_filter is not None:
            self._kalman_filter.reset()
        self._tick_count = 0
    
    def initialize_from_prices(self, prices: NDArray[np.float64] | list[float]) -> None:
        """
        从历史价格初始化卡尔曼滤波器
        
        Args:
            prices: 历史价格序列
        """
        if self._kalman_filter is None:
            return
        
        logger.info(f"Initializing denoise pipeline with {len(prices)} prices")
        
        if isinstance(self._kalman_filter, KalmanFilter):
            self._kalman_filter.initialize_from_prices(prices)
        elif isinstance(self._kalman_filter, AdaptiveKalmanFilter):
            # AdaptiveKalmanFilter 通过逐个更新初始化
            for p in prices:
                self._kalman_filter.update(p)
        
        self._tick_count = len(prices)
    
    def get_stats(self) -> dict:
        """获取管道统计信息"""
        stats = {
            "mode": self.config.mode.value,
            "tick_count": self._tick_count,
            "is_initialized": self._is_initialized,
        }
        
        if isinstance(self._kalman_filter, AdaptiveKalmanFilter):
            stats["current_Q"] = round(self._kalman_filter.current_Q, 6)
            stats["current_R"] = round(self._kalman_filter.current_R, 6)
            stats["innovation_variance"] = round(
                self._kalman_filter.innovation_variance, 6
            )
        
        return stats


def create_denoise_pipeline(
    mode: str = "adaptive",
    **kwargs,
) -> DenoisePipeline:
    """
    工厂函数：创建降噪管道
    
    Args:
        mode: 降噪模式 ("none", "wavelet", "kalman", "adaptive", "combined")
        **kwargs: 其他配置参数
        
    Returns:
        配置好的 DenoisePipeline
    """
    mode_enum = DenoiseMode(mode)
    config = DenoiseConfig(mode=mode_enum, **kwargs)
    return DenoisePipeline(config)
