"""
v4.0 推理引擎

实时推理流程：
1. Tick → UnifiedSampler → EventBar
2. EventBar → FeaturePipeline → Feature Vector
3. EventBar → PrimaryEngine → PrimarySignal (方向)
4. Feature → CfC (更新隐状态) → XGBoost → meta_confidence
5. PrimarySignal + meta_confidence → 最终信号

v4.0 更新：
- 加载 CfC 编码器和 XGBoost 模型
- 维护 CfC 隐状态，实现在线推理
- 支持完整的 CfC + XGBoost 预测管道

参考：降噪LNN特征提取与信号过滤.md Section 7
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from alphaos.core.logging import get_logger
from alphaos.core.types import Tick
from alphaos.v4.schemas import FeatureSchema
from alphaos.v4.sampling import SamplingConfig, UnifiedSampler
from alphaos.v4.features import FeatureConfig, FeaturePipelineV4, ThermodynamicsConfig
from alphaos.v4.primary import PrimaryEngineConfig, PrimaryEngineV4, PrimarySignalV4
from alphaos.v4.denoise import DenoiseConfig
from alphaos.v4.models import CfCConfig, CfCEncoder
from alphaos.data.event_bars.tick_imbalance import EventBar

logger = get_logger(__name__)


def _compute_feature_list_hash(feature_names: list[str]) -> str:
    """计算特征名称列表的稳定 hash（与训练端一致）"""
    content = json.dumps(feature_names, sort_keys=False, ensure_ascii=True)
    return hashlib.sha256(content.encode()).hexdigest()[:8]


def _compute_schema_mask_combo_hash(
    schema_hash: str,
    lnn_mask_hash: str,
    xgb_mask_hash: str,
) -> str:
    """计算 schema + masks 的组合哈希（与训练端一致）"""
    content = f"{schema_hash}:{lnn_mask_hash}:{xgb_mask_hash}"
    return hashlib.sha256(content.encode()).hexdigest()[:8]


class ConfidenceGateStage:
    """
    置信度门控阶段常量
    
    Cold Start Semantics (Hybrid 3-Stage):
    - HARD_FROZEN: buffer < min_required → 禁止交易
    - FIXED_FALLBACK: min_required <= buffer < full_buffer → 使用固定阈值
    - ROLLING_QUANTILE: buffer >= full_buffer → 使用滚动分位数
    """
    HARD_FROZEN = "HARD_FROZEN"       # Stage 1: 禁止交易
    FIXED_FALLBACK = "FIXED_FALLBACK" # Stage 2: 固定阈值
    ROLLING_QUANTILE = "ROLLING_QUANTILE"  # Stage 3: 滚动分位数


@dataclass
class ConfidenceGateConfig:
    """
    置信度门控配置（Rolling Quantile Threshold with Hybrid Cold Start）
    
    核心理念：
    - 固定阈值（如 0.65）在不同市场状态下表现不一致
    - 滚动分位数阈值可自适应模型信心分布漂移
    - 按相位分层可实现 staged recall release
    
    Cold Start Semantics (Hybrid 3-Stage, 语义冻结):
    - Stage 1 HARD_FROZEN: buffer_size < min_required
      - should_trade = False
      - filtered_reason = "COLD_START"
      - Rationale: 分布未形成，quantile 无物理意义
    
    - Stage 2 FIXED_FALLBACK: min_required <= buffer_size < full_buffer
      - threshold = fixed_fallback_threshold
      - filtered_reason includes "FIXED_FALLBACK" (if filtered)
      - Rationale: 分布开始稳定，但分位数仍不可靠
    
    - Stage 3 ROLLING_QUANTILE: buffer_size >= full_buffer
      - threshold = rolling_quantile(q_by_phase[phase])
      - Normal operation
    
    Args:
        mode: 阈值模式
            - "fixed": 使用固定 min_confidence（向后兼容）
            - "rolling_quantile": 使用滚动分位数阈值
        
        buffer_size: 滚动 buffer 最大大小（bar 数，默认 2000）
        base_quantile: 基础分位数（默认 95，即 top 5% 才交易）
        
        # Cold Start 参数（Hybrid 3-Stage）
        min_required: Stage 1 → 2 的阈值（默认 500）
        full_buffer: Stage 2 → 3 的阈值（默认 1500）
        fixed_fallback_threshold: Stage 2 使用的固定阈值（默认 0.70）
        
        # 分位数保护
        floor: 最小阈值（防止分位数过低导致过度交易）
        ceiling: 最大阈值（防止分位数过高导致永不交易）
        
        # 按相位分层（staged release）
        # 相位编码：0=FROZEN, 1=LAMINAR, 2=TURBULENT, 3=TRANSITION
        quantile_by_phase: 按相位映射不同分位数
            - 默认：TRANSITION=95, TURBULENT=92, LAMINAR=97, FROZEN=99
    """
    mode: str = "fixed"  # "fixed" | "rolling_quantile"
    buffer_size: int = 2000
    base_quantile: int = 95
    
    # Cold Start 参数（Hybrid 3-Stage）
    min_required: int = 500       # Stage 1 → 2
    full_buffer: int = 1500       # Stage 2 → 3
    fixed_fallback_threshold: float = 0.70  # Stage 2 使用的固定阈值
    
    floor: float = 0.1   # 最小阈值
    ceiling: float = 0.9  # 最大阈值
    
    # 按相位分层分位数（staged recall release）
    quantile_by_phase: dict = field(default_factory=lambda: {
        "FROZEN": 99,       # 最严格，几乎不交易
        "LAMINAR": 97,      # 较严格
        "TURBULENT": 92,    # 适度放松
        "TRANSITION": 95,   # 基准
    })
    
    # 按趋势对齐调制分位数（ST(15m) 阈值调制）
    # - ST_ALIGNED: 顺势分位数（更宽松，如 90）
    # - ST_COUNTER: 逆势分位数（更严格，如 97）
    # 空字典 = 不启用调制，仅使用 quantile_by_phase
    quantile_by_trend: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "buffer_size": self.buffer_size,
            "base_quantile": self.base_quantile,
            "min_required": self.min_required,
            "full_buffer": self.full_buffer,
            "fixed_fallback_threshold": self.fixed_fallback_threshold,
            "floor": self.floor,
            "ceiling": self.ceiling,
            "quantile_by_phase": self.quantile_by_phase,
            "quantile_by_trend": self.quantile_by_trend,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConfidenceGateConfig":
        return cls(
            mode=data.get("mode", "fixed"),
            buffer_size=data.get("buffer_size", 2000),
            base_quantile=data.get("base_quantile", 95),
            min_required=data.get("min_required", 500),
            full_buffer=data.get("full_buffer", 1500),
            fixed_fallback_threshold=data.get("fixed_fallback_threshold", 0.70),
            floor=data.get("floor", 0.1),
            ceiling=data.get("ceiling", 0.9),
            quantile_by_phase=data.get("quantile_by_phase", {
                "FROZEN": 99,
                "LAMINAR": 97,
                "TURBULENT": 92,
                "TRANSITION": 95,
            }),
            quantile_by_trend=data.get("quantile_by_trend", {}),
        )


@dataclass
class InferenceConfig:
    """
    推理配置
    
    Args:
        sampling: 采样配置
        features: 特征配置
        primary: Primary Engine 配置
        denoise: 降噪配置
        
        # 模型路径
        model_dir: 模型目录（包含 cfc_encoder.pt, xgb_model.json, schema.json）
        model_path: XGBoost 模型路径（兼容旧配置）
        cfc_model_path: CfC 编码器路径
        schema_path: Schema 文件路径
        
        # 预热参数
        warmup_bars: 预热 Bar 数（预热期间不输出信号）
        
        # 信号参数（传统 fixed 模式）
        min_confidence: 最小置信度阈值（mode=fixed 时使用）
        require_phase_transition: 是否要求 Phase Transition 状态
        
        # 置信度门控配置（新增 rolling_quantile 支持）
        confidence_gate: 置信度门控配置
        
        # 设备
        device: 推理设备 ("cpu", "cuda", "mps")
    """
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    primary: PrimaryEngineConfig = field(default_factory=PrimaryEngineConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    
    model_dir: str = ""
    model_path: str = ""  # 兼容旧配置（XGBoost 路径）
    cfc_model_path: str = ""
    schema_path: str = ""
    
    warmup_bars: int = 50
    min_confidence: float = 0.65  # 向后兼容，mode=fixed 时使用
    require_phase_transition: bool = True
    
    # 事件门控模式（决定“何时进入 meta-model 流程”）
    # - "fvg_event": 以 FeatureSchema.fvg_event != 0 的 bar 作为事件（推荐，与训练对齐）
    # - "fvg_event_or_decay": rising-edge + FVG 活跃衰减窗口
    # - "primary_engine": 向后兼容，使用 PrimaryEngine 输出作为 gate
    event_gate_mode: str = "fvg_event"

    # 趋势对齐来源（用于顺/逆势判断）
    # - "st_15m": 使用 15m SuperTrend (st_trend_15m)
    # - "primary": 使用 PrimaryEngine 当前方向 (trend_direction)
    trend_alignment_source: str = "st_15m"
    
    # v4.0: FVG 事件窗口（包含当前 bar，共 N 根 bar）
    fvg_event_window_bars: int = 5
    
    # v4.1: FVG 衰减窗口（fvg_event 过后仍可短暂放行）
    fvg_decay_bars: int = 3

    # v4.0: CfC delta_t 裁剪范围（秒）
    delta_t_min: float = 0.001
    delta_t_max: float = 100.0

    # v4.0: FVG 事件备用止损（ATR 倍数）
    fallback_sl_atr_factor: float = 3.0

    # v4.x: 结构确认参数（BOS/CHOCH/TL break）
    structure_lookback_bars: int = 20

    # v4.x: 逆势门控（结构确认 + 阈值提升）
    counter_trend_require_structure: bool = False
    counter_trend_confirm_window_bars: int = 0
    counter_trend_threshold_boost: float = 0.0
    counter_trend_volume_scale: float = 1.0

    # v4.0: 诊断日志参数（近零置信度）
    diagnostic_confidence_threshold: float = 0.01
    diagnostic_log_interval: int = 50
    diagnostic_log_initial_bars: int = 10
    
    # 置信度门控配置
    confidence_gate: ConfidenceGateConfig = field(default_factory=ConfidenceGateConfig)
    
    # v4.0: 仓位管理配置 (Kelly Criterion)
    position_sizing: dict | None = field(default=None)
    
    # v4.0: 风险管理配置
    risk: dict | None = field(default=None)
    
    device: str = "cpu"
    
    def to_dict(self) -> dict:
        return {
            "sampling": self.sampling.to_dict(),
            "features": self.features.to_dict(),
            "primary": self.primary.to_dict(),
            "denoise": self.denoise.to_dict(),
            "model_dir": self.model_dir,
            "model_path": self.model_path,
            "cfc_model_path": self.cfc_model_path,
            "schema_path": self.schema_path,
            "warmup_bars": self.warmup_bars,
            "min_confidence": self.min_confidence,
            "require_phase_transition": self.require_phase_transition,
            "event_gate_mode": self.event_gate_mode,
            "trend_alignment_source": self.trend_alignment_source,
            "fvg_event_window_bars": self.fvg_event_window_bars,
            "fvg_decay_bars": self.fvg_decay_bars,
            "delta_t_min": self.delta_t_min,
            "delta_t_max": self.delta_t_max,
            "fallback_sl_atr_factor": self.fallback_sl_atr_factor,
            "structure_lookback_bars": self.structure_lookback_bars,
            "counter_trend_require_structure": self.counter_trend_require_structure,
            "counter_trend_confirm_window_bars": self.counter_trend_confirm_window_bars,
            "counter_trend_threshold_boost": self.counter_trend_threshold_boost,
            "counter_trend_volume_scale": self.counter_trend_volume_scale,
            "diagnostic_confidence_threshold": self.diagnostic_confidence_threshold,
            "diagnostic_log_interval": self.diagnostic_log_interval,
            "diagnostic_log_initial_bars": self.diagnostic_log_initial_bars,
            "confidence_gate": self.confidence_gate.to_dict(),
            "device": self.device,
        }
    
    @classmethod
    def from_yaml_dict(cls, data: dict, model_dir: str = "", device: str = "cpu") -> "InferenceConfig":
        """
        从 YAML 配置字典创建 InferenceConfig
        
        支持 v4 YAML 配置格式 (configs/v4/xauusd.yaml)
        
        Args:
            data: YAML 配置字典
            model_dir: 模型目录路径
            device: 推理设备
            
        Returns:
            InferenceConfig 实例
        """
        from alphaos.v4.sampling import SamplingMode, VolumeSource
        from alphaos.v4.features.vpin import VPINConfig
        
        def _require_section(config: dict, name: str) -> dict:
            if name not in config or config[name] is None:
                raise ValueError(f"缺少配置段: {name}")
            if not isinstance(config[name], dict):
                raise ValueError(f"配置段类型错误: {name}")
            return config[name]

        def _require_keys(section: dict, keys: list[str], prefix: str) -> None:
            for key in keys:
                if key not in section:
                    raise ValueError(f"缺少配置项: {prefix}.{key}")

        # === 解析 sampling 配置 ===
        sampling_data = _require_section(data, "sampling")
        _require_keys(
            sampling_data,
            [
                "mode",
                "volume_source",
                "target_volume",
                "initial_expected_ticks",
                "initial_expected_imbalance",
                "ewma_alpha",
                "tick_rule_gamma",
                "tick_rule_threshold",
                "max_buffer_size",
                "synthetic_base_volume",
            ],
            "sampling",
        )
        sampling_config = SamplingConfig(
            mode=SamplingMode(sampling_data.get("mode", "volume_bars")),
            volume_source=VolumeSource(sampling_data.get("volume_source", "tick_count")),
            target_volume=sampling_data.get("target_volume", 100.0),
            initial_expected_ticks=sampling_data.get("initial_expected_ticks", 50.0),
            initial_expected_imbalance=sampling_data.get("initial_expected_imbalance", 0.5),
            ewma_alpha=sampling_data.get("ewma_alpha", 0.1),
            tick_rule_gamma=sampling_data.get("tick_rule_gamma", 0.95),
            tick_rule_threshold=sampling_data.get("tick_rule_threshold", 0.5),
            max_buffer_size=sampling_data.get("max_buffer_size", 500),
            synthetic_base_volume=sampling_data.get("synthetic_base_volume", 100.0),
        )
        
        # === 解析 denoise 配置（SSOT: core/config.DenoiseConfig）===
        from alphaos.core.config import DenoiseConfig as CoreDenoiseConfig
        from alphaos.v4.denoise import DenoiseConfig as V4DenoiseConfig
        denoise_data = _require_section(data, "denoise")
        _require_keys(denoise_data, ["kalman", "wavelet"], "denoise")
        _require_keys(
            denoise_data.get("kalman", {}),
            ["enabled", "process_variance", "measurement_variance", "initial_uncertainty", "use_adaptive"],
            "denoise.kalman",
        )
        _require_keys(
            denoise_data.get("wavelet", {}),
            ["enabled", "wavelet", "level", "threshold_mode", "threshold_rule"],
            "denoise.wavelet",
        )
        core_denoise_cfg = CoreDenoiseConfig.model_validate(denoise_data)
        denoise_config = V4DenoiseConfig.from_core_config(core_denoise_cfg)
        
        # === 解析 features 配置 ===
        features_data = _require_section(data, "features")
        _require_keys(
            features_data,
            [
                "zscore_window",
                "volatility_window",
                "ofi_window",
                "kyle_lambda_window",
                "thermo_window",
                "tick_intensity_alpha",
                "micro_vol_lambda",
                "vpin",
                "clip_zscore",
                "zscore_clip_value",
            ],
            "features",
        )
        _require_keys(
            features_data.get("vpin", {}),
            ["bucket_volume", "n_buckets"],
            "features.vpin",
        )
        vpin_data = features_data.get("vpin", {})
        vpin_config = VPINConfig(
            bucket_volume=vpin_data.get("bucket_volume", 1000),
            n_buckets=vpin_data.get("n_buckets", 50),
        )
        thermodynamics_config = ThermodynamicsConfig.from_dict(
            features_data.get("thermodynamics", {})
        )
        features_config = FeatureConfig(
            zscore_window=features_data.get("zscore_window", 100),
            volatility_window=features_data.get("volatility_window", 20),
            ofi_window=features_data.get("ofi_window", 20),
            kyle_lambda_window=features_data.get("kyle_lambda_window", 50),
            thermo_window=features_data.get("thermo_window", 50),
            tick_intensity_alpha=features_data.get("tick_intensity_alpha", 0.1),
            micro_vol_lambda=features_data.get("micro_vol_lambda", 0.94),
            vpin_config=vpin_config,
            denoise_config=denoise_config,
            thermodynamics=thermodynamics_config,
            clip_zscore=features_data.get("clip_zscore", True),
            zscore_clip_value=features_data.get("zscore_clip_value", 5.0),
        )
        
        # === 解析 primary 配置 ===
        primary_data = _require_section(data, "primary")
        _require_keys(
            primary_data,
            [
                "pivot_lookback",
                "atr_period",
                "atr_factor",
                "min_fvg_size_bps",
                "max_fvg_age_bars",
                "ce_tolerance_bps",
                "min_trend_duration",
                "cooldown_bars",
                "sl_buffer_bps",
                "require_fvg",
                "fvg_entry_mode",
            ],
            "primary",
        )
        primary_config = PrimaryEngineConfig(
            pivot_lookback=primary_data.get("pivot_lookback", 2),
            atr_period=primary_data.get("atr_period", 10),
            atr_factor=primary_data.get("atr_factor", 3.0),
            min_fvg_size_bps=primary_data.get("min_fvg_size_bps", 0.5),
            max_fvg_age_bars=primary_data.get("max_fvg_age_bars", 30),
            ce_tolerance_bps=primary_data.get("ce_tolerance_bps", 1.0),
            min_trend_duration=primary_data.get("min_trend_duration", 2),
            cooldown_bars=primary_data.get("cooldown_bars", 3),
            sl_buffer_bps=primary_data.get("sl_buffer_bps", 5.0),
            require_fvg=primary_data.get("require_fvg", True),
            fvg_entry_mode=primary_data.get("fvg_entry_mode", "immediate"),
        )
        
        # === 解析 confidence_gate 配置 ===
        gate_data = _require_section(data, "confidence_gate")
        _require_keys(
            gate_data,
            [
                "mode",
                "buffer_size",
                "base_quantile",
                "min_required",
                "full_buffer",
                "fixed_fallback_threshold",
                "floor",
                "ceiling",
                "quantile_by_phase",
                "quantile_by_trend",
            ],
            "confidence_gate",
        )
        quantile_by_phase = gate_data.get("quantile_by_phase", {})
        quantile_by_trend = gate_data.get("quantile_by_trend", {})
        confidence_gate = ConfidenceGateConfig(
            mode=gate_data.get("mode", "rolling_quantile"),
            buffer_size=gate_data.get("buffer_size", 2000),
            base_quantile=gate_data.get("base_quantile", 95),
            min_required=gate_data.get("min_required", 500),
            full_buffer=gate_data.get("full_buffer", 1500),
            fixed_fallback_threshold=gate_data.get("fixed_fallback_threshold", 0.70),
            floor=gate_data.get("floor", 0.1),
            ceiling=gate_data.get("ceiling", 0.9),
            quantile_by_phase=quantile_by_phase,
            quantile_by_trend=quantile_by_trend,
        )
        
        # === 解析 inference 配置 ===
        inference_data = _require_section(data, "inference")
        _require_keys(
            inference_data,
            [
                "warmup_bars",
                "require_phase_transition",
                "event_gate_mode",
                "delta_t_min",
                "delta_t_max",
                "fallback_sl_atr_factor",
                "diagnostic_confidence_threshold",
                "diagnostic_log_interval",
                "diagnostic_log_initial_bars",
            ],
            "inference",
        )
        
        # === 解析 position_sizing 配置（v4.0 Kelly） ===
        execution_data = _require_section(data, "execution")
        exec_position_sizing = _require_section(execution_data, "position_sizing")
        _require_keys(
            exec_position_sizing,
            [
                "mode",
                "kelly_fraction",
                "kelly_max_fraction",
                "expected_edge_pct",
                "win_rate",
                "risk_per_trade_pct",
                "account_balance",
                "risk_reward_ratio",
                "min_lots",
                "max_lots",
                "lot_step",
                "linear_conf_max",
            ],
            "execution.position_sizing",
        )
        legacy_position_sizing = data.get("position_sizing", {})
        for key, value in exec_position_sizing.items():
            if key in legacy_position_sizing and legacy_position_sizing[key] != value:
                raise ValueError(f"position_sizing.{key} 与 execution.position_sizing.{key} 不一致")
        from types import SimpleNamespace
        position_sizing = SimpleNamespace(**exec_position_sizing)
        
        # === 解析 risk 配置 ===
        risk_data = _require_section(data, "risk")
        _require_keys(
            risk_data,
            [
                "max_position_usd",
                "max_daily_loss_pct",
                "max_consecutive_losses",
                "min_temperature",
                "max_entropy",
            ],
            "risk",
        )
        risk = SimpleNamespace(**risk_data)
        
        return cls(
            sampling=sampling_config,
            features=features_config,
            primary=primary_config,
            denoise=denoise_config,
            model_dir=model_dir,
            schema_path=inference_data.get("schema_path", ""),
            warmup_bars=inference_data.get("warmup_bars", 50),
            min_confidence=inference_data.get(
                "min_confidence",
                gate_data.get("fixed_fallback_threshold", 0.65),
            ),
            require_phase_transition=inference_data.get("require_phase_transition", False),
            event_gate_mode=inference_data.get("event_gate_mode", "fvg_event"),
            trend_alignment_source=inference_data.get("trend_alignment_source", "st_15m"),
            fvg_event_window_bars=inference_data.get("fvg_event_window_bars", 5),
            fvg_decay_bars=inference_data.get("fvg_decay_bars", 3),
            delta_t_min=inference_data.get("delta_t_min", 0.001),
            delta_t_max=inference_data.get("delta_t_max", 100.0),
            fallback_sl_atr_factor=inference_data.get("fallback_sl_atr_factor", 3.0),
            structure_lookback_bars=inference_data.get("structure_lookback_bars", 20),
            counter_trend_require_structure=inference_data.get("counter_trend_require_structure", False),
            counter_trend_confirm_window_bars=inference_data.get("counter_trend_confirm_window_bars", 0),
            counter_trend_threshold_boost=inference_data.get("counter_trend_threshold_boost", 0.0),
            counter_trend_volume_scale=inference_data.get("counter_trend_volume_scale", 1.0),
            diagnostic_confidence_threshold=inference_data.get("diagnostic_confidence_threshold", 0.01),
            diagnostic_log_interval=inference_data.get("diagnostic_log_interval", 50),
            diagnostic_log_initial_bars=inference_data.get("diagnostic_log_initial_bars", 10),
            confidence_gate=confidence_gate,
            position_sizing=position_sizing,
            risk=risk,
            device=device,
        )


@dataclass
class InferenceResult:
    """
    单次推理结果
    
    Attributes:
        # Bar 信息
        bar_idx: Bar 索引
        close_price: 收盘价
        
        # Primary 信号
        has_signal: 是否有 Primary 信号
        direction: 信号方向 (1=LONG, -1=SHORT, 0=NONE)
        entry_price: 建议入场价格
        stop_loss: 止损价格
        
        # Meta Model
        meta_confidence: 元模型置信度
        
        # 最终决策
        should_trade: 是否应该交易
        filtered_reason: 过滤原因（如果 should_trade=False）
        
        # 市场状态
        market_phase: 市场相位
        market_temperature: 市场温度
        market_entropy: 市场熵
        trend_duration: 趋势持续 Bar 数
        
        # 诊断字段（v4.0 方向语义修复）
        st_trend_15m: 15分钟 SuperTrend 趋势方向 (+1/-1/0)
        fvg_event: FVG 事件信号 (+1/-1/0)
        trend_direction: PrimaryEngine event-bar ST 方向
        
        # 特征快照（调试用）
        features: 当前特征向量
    """
    bar_idx: int = -1
    close_price: float = 0.0
    
    has_signal: bool = False
    direction: int = 0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    
    meta_confidence: float = 0.0
    
    should_trade: bool = False
    filtered_reason: str = ""
    
    market_phase: str = "UNKNOWN"
    market_temperature: float = 0.0
    market_entropy: float = 0.0
    trend_duration: int = 0
    
    # 诊断字段（v4.0 方向语义修复）
    st_trend_15m: int = 0
    fvg_event: int = 0
    trend_direction: int = 0
    
    features: NDArray[np.float32] | None = None
    
    def to_dict(self) -> dict:
        return {
            "bar_idx": self.bar_idx,
            "close_price": round(self.close_price, 2),
            "has_signal": self.has_signal,
            "direction": self.direction,
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "meta_confidence": round(self.meta_confidence, 4),
            "should_trade": self.should_trade,
            "filtered_reason": self.filtered_reason,
            "market_phase": self.market_phase,
            "market_temperature": round(self.market_temperature, 6),
            "market_entropy": round(self.market_entropy, 6),
            "trend_duration": self.trend_duration,
            # 诊断字段
            "st_trend_15m": self.st_trend_15m,
            "fvg_event": self.fvg_event,
            "trend_direction": self.trend_direction,
        }


@dataclass
class InferenceEngineV4:
    """
    v4.0 实时推理引擎
    
    使用方式：
    ```python
    config = InferenceConfig(
        model_dir="models/v4/run_001",  # 包含 cfc_encoder.pt + xgb_model.json
    )
    engine = InferenceEngineV4(config)
    
    # 流式处理
    for tick in tick_stream:
        result = engine.process_tick(tick)
        if result.should_trade:
            # 执行交易
            pass
    ```
    
    CfC + XGBoost 推理流程：
    1. 每个新 Bar 更新 CfC 隐状态（encode_single_step）
    2. 组合当前特征 + CfC 隐状态作为 XGBoost 输入
    3. XGBoost 输出 meta_confidence
    """
    config: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Schema（加载自文件或使用默认）
    _schema: FeatureSchema = field(default_factory=FeatureSchema.default, init=False)
    
    # 组件
    _sampler: UnifiedSampler | None = field(default=None, init=False)
    _feature_pipeline: FeaturePipelineV4 | None = field(default=None, init=False)
    _primary_engine: PrimaryEngineV4 | None = field(default=None, init=False)
    _xgb_model: Any = field(default=None, init=False)  # XGBoost model
    
    # CfC 编码器
    _cfc_encoder: CfCEncoder | None = field(default=None, init=False)
    _cfc_config: CfCConfig | None = field(default=None, init=False)
    _cfc_hidden_states: list[Tensor] | None = field(default=None, init=False)
    _device: torch.device = field(default=None, init=False)
    
    # 状态
    _bar_count: int = field(default=0, init=False)
    _tick_count: int = field(default=0, init=False)
    _is_warmed_up: bool = field(default=False, init=False)
    _last_bar: EventBar | None = field(default=None, init=False)
    _last_features: NDArray | None = field(default=None, init=False)
    _last_delta_t: float = field(default=1.0, init=False)  # 上一个 bar 的时间间隔
    _prev_trend_direction: int = field(default=0, init=False)
    _last_structure_event_bar: int = field(default=-1, init=False)
    _last_structure_event_dir: int = field(default=0, init=False)
    _structure_lookback: int = field(default=20, init=False)
    _recent_highs: deque[float] = field(default_factory=deque, init=False)
    _recent_lows: deque[float] = field(default_factory=deque, init=False)
    _structure_direction: int = field(default=0, init=False)
    
    # 特征序列缓冲（用于 CfC 序列输入）
    _feature_buffer: list[NDArray] = field(default_factory=list, init=False)
    _delta_t_buffer: list[float] = field(default_factory=list, init=False)
    _sequence_length: int = field(default=100, init=False)
    
    # 特征分离 (LNN vs XGB)
    _use_feature_split: bool = field(default=False, init=False)
    _lnn_feature_names: list[str] = field(default_factory=list, init=False)
    _xgb_feature_names: list[str] = field(default_factory=list, init=False)
    _lnn_indices: list[int] = field(default_factory=list, init=False)
    _xgb_indices: list[int] = field(default_factory=list, init=False)
    
    # 事件门控模式（训练/推理对齐）
    _event_gate_mode: str = field(default="fvg_event", init=False)
    
    # Rising-edge guard 状态（防止 fvg_event 退化为 regime flag）
    # 即使 generator 有 cooldown，推理端也保留 rising-edge guard 作为安全护栏
    _prev_fvg_event: int = field(default=0, init=False)
    
    # FVG 事件窗口状态（rising-edge 后允许持续 N 根 bar）
    _fvg_window_remaining: int = field(default=0, init=False)
    _fvg_window_direction: int = field(default=0, init=False)
    
    # 滚动置信度 buffer（用于 rolling_quantile 阈值模式）
    _confidence_buffer: list[float] = field(default_factory=list, init=False)
    _confidence_buffer_max_size: int = field(default=2000, init=False)
    
    def __post_init__(self) -> None:
        """初始化引擎"""
        self._init_device()
        self._init_components()
    
    def _init_device(self) -> None:
        """初始化计算设备"""
        device_str = self.config.device.lower()
        
        if device_str == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif device_str == "mps" and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        
        logger.info(f"Inference device: {self._device}")
    
    def _init_components(self) -> None:
        """初始化组件"""
        # 确定模型目录
        model_dir = Path(self.config.model_dir) if self.config.model_dir else None
        
        # 加载 Schema
        schema_path = self.config.schema_path
        if not schema_path and model_dir and (model_dir / "schema.json").exists():
            schema_path = str(model_dir / "schema.json")
        
        if schema_path and Path(schema_path).exists():
            self._schema = FeatureSchema.load(schema_path)
            logger.info(f"Loaded schema from {schema_path}")
        
        # 初始化采样器
        self._sampler = UnifiedSampler(self.config.sampling)
        
        # 初始化特征管道
        # 过滤掉 denoise_config 以避免重复传递
        feature_config = FeatureConfig(
            **{k: v for k, v in self.config.features.__dict__.items() 
               if not k.startswith("_") and k != "denoise_config"},
            denoise_config=self.config.denoise,
        )
        self._feature_pipeline = FeaturePipelineV4(feature_config, self._schema)
        
        # 初始化 Primary 引擎
        self._primary_engine = PrimaryEngineV4(self.config.primary)

        # 结构检测窗口初始化
        self._structure_lookback = max(5, int(getattr(self.config, "structure_lookback_bars", 20)))
        self._recent_highs = deque(maxlen=self._structure_lookback)
        self._recent_lows = deque(maxlen=self._structure_lookback)
        
        # 加载 CfC 编码器
        self._load_cfc_encoder(model_dir)
        
        # 加载 XGBoost 模型
        xgb_path = self.config.model_path
        if not xgb_path and model_dir:
            for candidate in ["xgb_model.json", "xgb_model.ubj", "model.json"]:
                if (model_dir / candidate).exists():
                    xgb_path = str(model_dir / candidate)
                    break
        
        if xgb_path and Path(xgb_path).exists():
            self._load_xgb_model(xgb_path)
        
        # 加载 bundle 元数据（获取 sequence_length 和 feature_mask）
        if model_dir and (model_dir / "bundle_meta.json").exists():
            with open(model_dir / "bundle_meta.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._sequence_length = meta.get("sequence_length", 100)
            
            # 事件门控模式：推理配置优先，避免被训练元数据覆盖
            meta_event_mode = meta.get("event_anchor_mode")
            self._event_gate_mode = self.config.event_gate_mode
            if meta_event_mode and meta_event_mode != self._event_gate_mode:
                logger.info(
                    "Ignoring bundle_meta event_anchor_mode; using config event_gate_mode",
                    meta_event_mode=meta_event_mode,
                    config_event_mode=self._event_gate_mode,
                )
            
            # 加载推荐的 confidence_gate 配置（如果有，且当前配置为默认）
            # 这允许训练端"推荐"cold start 参数，同时推理端仍可覆盖
            recommended_gate = meta.get("recommended_confidence_gate")
            if recommended_gate:
                # 仅在用户未显式覆盖时应用推荐值
                if self.config.confidence_gate.mode == "fixed":
                    # 用户使用 fixed 模式，不应用推荐值
                    pass
                else:
                    # 使用推荐的 cold start 参数（如果存在）
                    if "min_required" in recommended_gate:
                        self.config.confidence_gate.min_required = recommended_gate["min_required"]
                    if "full_buffer" in recommended_gate:
                        self.config.confidence_gate.full_buffer = recommended_gate["full_buffer"]
                    if "fixed_fallback_threshold" in recommended_gate:
                        self.config.confidence_gate.fixed_fallback_threshold = recommended_gate["fixed_fallback_threshold"]
                    logger.info(
                        "Loaded recommended cold start params from bundle_meta",
                        min_required=self.config.confidence_gate.min_required,
                        full_buffer=self.config.confidence_gate.full_buffer,
                        fixed_fallback_threshold=self.config.confidence_gate.fixed_fallback_threshold,
                    )
            
            # 加载特征分离配置
            self._use_feature_split = meta.get("use_feature_split", False)
            self._lnn_feature_names = meta.get("lnn_feature_names", [])
            self._xgb_feature_names = meta.get("xgb_feature_names", [])
            
            # 计算特征索引
            if self._use_feature_split and self._lnn_feature_names and self._xgb_feature_names:
                schema_names = self._schema.feature_names
                self._lnn_indices = [schema_names.index(n) for n in self._lnn_feature_names if n in schema_names]
                self._xgb_indices = [schema_names.index(n) for n in self._xgb_feature_names if n in schema_names]
                logger.info(
                    f"Feature split enabled: LNN={len(self._lnn_indices)} features, XGB={len(self._xgb_indices)} features"
                )
            else:
                # 不分离时使用全量特征
                self._lnn_indices = list(range(self._schema.num_features))
                self._xgb_indices = list(range(self._schema.num_features))
            
            # 校验 mask hashes（防止 research/live silently diverge）
            saved_lnn_hash = meta.get("lnn_mask_hash")
            saved_xgb_hash = meta.get("xgb_mask_hash")
            saved_combo_hash = meta.get("schema_mask_combo_hash")
            
            if saved_lnn_hash and saved_xgb_hash:
                # 重新计算 hash 进行校验
                current_lnn_hash = _compute_feature_list_hash(self._lnn_feature_names)
                current_xgb_hash = _compute_feature_list_hash(self._xgb_feature_names)
                current_combo_hash = _compute_schema_mask_combo_hash(
                    self._schema.schema_hash, current_lnn_hash, current_xgb_hash
                )
                
                if current_lnn_hash != saved_lnn_hash:
                    raise ValueError(
                        f"LNN mask hash mismatch! "
                        f"Saved={saved_lnn_hash}, Current={current_lnn_hash}. "
                        f"Research/Live code may have diverged."
                    )
                
                if current_xgb_hash != saved_xgb_hash:
                    raise ValueError(
                        f"XGB mask hash mismatch! "
                        f"Saved={saved_xgb_hash}, Current={current_xgb_hash}. "
                        f"Research/Live code may have diverged."
                    )
                
                logger.info(
                    f"Mask hash validation PASSED: "
                    f"lnn={current_lnn_hash}, xgb={current_xgb_hash}, combo={current_combo_hash}"
                )
            else:
                # 旧包无 hash，仅 warn（向后兼容）
                logger.warning(
                    "Bundle does not contain mask hashes (old format). "
                    "Consider re-training to enable hash validation."
                )
        else:
            # 未加载 bundle_meta 时，使用配置指定的事件门控模式
            self._event_gate_mode = self.config.event_gate_mode
        
        logger.info(
            "InferenceEngineV4 initialized",
            schema_hash=self._schema.schema_hash,
            warmup_bars=self.config.warmup_bars,
            has_cfc=self._cfc_encoder is not None,
            has_xgb=self._xgb_model is not None,
        )
    
    def _load_cfc_encoder(self, model_dir: Path | None) -> None:
        """加载 CfC 编码器"""
        if model_dir is None:
            return
        
        cfc_config_path = model_dir / "cfc_config.json"
        cfc_model_path = self.config.cfc_model_path or (model_dir / "cfc_encoder.pt")
        
        if not cfc_config_path.exists() or not Path(cfc_model_path).exists():
            logger.info("CfC encoder not found, using XGBoost-only mode")
            return
        
        try:
            # 加载 CfC 配置
            with open(cfc_config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            self._cfc_config = CfCConfig.from_dict(config_dict)
            
            # 创建并加载 CfC 编码器
            self._cfc_encoder = CfCEncoder(self._cfc_config)
            self._cfc_encoder.load_state_dict(
                torch.load(cfc_model_path, map_location=self._device)
            )
            self._cfc_encoder.to(self._device)
            self._cfc_encoder.eval()
            
            # 初始化隐状态
            self._cfc_hidden_states = self._cfc_encoder.init_hidden_states(
                batch_size=1,
                device=self._device,
            )
            
            logger.info(
                f"Loaded CfC encoder from {cfc_model_path}",
                hidden_dim=self._cfc_config.hidden_dim,
                num_layers=self._cfc_config.num_layers,
            )
            
        except Exception as e:
            logger.error(f"Failed to load CfC encoder: {e}")
            self._cfc_encoder = None
            self._cfc_hidden_states = None
    
    def _load_xgb_model(self, path: str) -> None:
        """加载 XGBoost 模型"""
        try:
            import xgboost as xgb
            
            self._xgb_model = xgb.XGBClassifier()
            self._xgb_model.load_model(path)
            logger.info(f"Loaded XGBoost model from {path}")
            
        except ImportError:
            logger.error("XGBoost not installed. Model predictions disabled.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def process_tick(self, tick: Tick) -> InferenceResult | None:
        """
        处理单个 Tick
        
        Args:
            tick: 输入 Tick
            
        Returns:
            InferenceResult 如果产生新 Bar，否则 None
        """
        self._tick_count += 1
        
        # 添加 Tick 到采样器
        bar = self._sampler.add_tick(tick)
        
        if bar is None:
            # 还没形成完整的 Bar
            return None
        
        # 处理新 Bar
        return self._process_bar(bar)
    
    def _process_bar(self, bar: EventBar) -> InferenceResult:
        """处理完成的 Bar"""
        self._bar_count += 1
        self._last_bar = bar
        
        # 计算 delta_t（用于 CfC）
        delta_t = bar.duration_ms / 1000.0 if bar.duration_ms > 0 else 1.0
        delta_t = max(self.config.delta_t_min, min(self.config.delta_t_max, delta_t))
        self._last_delta_t = delta_t
        
        # 检查预热状态
        if self._bar_count <= self.config.warmup_bars:
            self._is_warmed_up = self._bar_count >= self.config.warmup_bars
        
        # 更新 Primary 引擎
        primary_signal = self._primary_engine.update(bar)
        
        # 获取当前趋势状态
        trend_state = self._primary_engine.current_state
        trend_direction = self._primary_engine.current_direction
        trend_duration = trend_state.trend_duration if trend_state else 0
        supertrend_line = trend_state.supertrend_line if trend_state else 0.0
        
        # 计算特征
        features = self._feature_pipeline.update(
            bar,
            trend_direction=trend_direction,
            supertrend_line=supertrend_line,
            trend_duration=trend_duration,
        )
        self._last_features = features

        # 结构确认信号（BOS/CHOCH/TL break）
        structure_bos = 0
        structure_choch = 0
        structure_tl_break = 0

        prev_swing_high = max(self._recent_highs) if self._recent_highs else 0.0
        prev_swing_low = min(self._recent_lows) if self._recent_lows else 0.0
        bullish_break = prev_swing_high > 0 and bar.close > prev_swing_high
        bearish_break = prev_swing_low > 0 and bar.close < prev_swing_low

        if bullish_break:
            if self._structure_direction == -1:
                structure_choch = 1
            else:
                structure_bos = 1
            self._structure_direction = 1
        elif bearish_break:
            if self._structure_direction == 1:
                structure_choch = -1
            else:
                structure_bos = -1
            self._structure_direction = -1

        # TL break: 价格反向穿越 1m supertrend 线（轻量结构破坏信号）
        if supertrend_line > 0:
            if trend_direction == -1 and bar.close > supertrend_line:
                structure_tl_break = 1
            elif trend_direction == 1 and bar.close < supertrend_line:
                structure_tl_break = -1

        self._recent_highs.append(bar.high)
        self._recent_lows.append(bar.low)

        structure_signal = 0
        if structure_choch != 0:
            structure_signal = 1 if structure_choch > 0 else -1
        elif structure_bos != 0:
            structure_signal = 1 if structure_bos > 0 else -1
        elif structure_tl_break != 0:
            structure_signal = 1 if structure_tl_break > 0 else -1

        if structure_signal != 0:
            self._last_structure_event_bar = self._bar_count
            self._last_structure_event_dir = structure_signal
        self._prev_trend_direction = trend_direction
        
        # 更新特征缓冲
        self._feature_buffer.append(features)
        self._delta_t_buffer.append(delta_t)
        if len(self._feature_buffer) > self._sequence_length:
            self._feature_buffer.pop(0)
            self._delta_t_buffer.pop(0)
        
        # 更新 CfC 隐状态（在线递推）
        cfc_hidden_output = self._update_cfc_hidden_state(features, delta_t)
        
        # 获取市场相位
        ts_phase_idx = self._schema.get_index("ts_phase")
        ts_phase = int(features[ts_phase_idx])
        market_phase = ["FROZEN", "LAMINAR", "TURBULENT", "TRANSITION"][ts_phase]

        # 热力学字段（用于诊断日志；取不到时为 None）
        ts_phase_value = ts_phase
        try:
            temp_idx = self._schema.get_index("market_temperature")
            ent_idx = self._schema.get_index("market_entropy")
            # Keep full precision for risk logic; format only in logs
            market_temperature = float(features[temp_idx])
            market_entropy = float(features[ent_idx])
        except Exception:
            market_temperature = None
            market_entropy = None
        
        # === 事件门控：决定“是否进入 meta-model 流程” ===
        # 设计目标：
        # - event_gate_mode="fvg_event": 以离散冲击 fvg_event 作为 t0（事件中心）
        # - event_gate_mode="fvg_event_or_decay": rising-edge + FVG 衰减窗口
        # - event_gate_mode="primary_engine": 向后兼容，沿用 PrimaryEngine gate
        event_present = False
        event_direction = 0
        entry_price = 0.0
        stop_loss = 0.0
        
        # 用于 FVG_DIR_MISMATCH 检查和诊断的变量（仅在 fvg_event 模式下有效）
        st_trend_15m = 0
        fvg_direction = 0
        fvg_event_val = 0  # 原始 fvg_event 值（用于诊断日志）
        prev_fvg_event = self._prev_fvg_event
        fvg_event_raw = 0
        is_rising_edge = False
        event_direction_source = "none"
        in_event_window = False
        event_window_remaining = 0
        
        if self._event_gate_mode == "primary_engine":
            event_present = primary_signal is not None
            if primary_signal is not None:
                event_direction = int(primary_signal.direction)
                entry_price = float(primary_signal.entry_price)
                stop_loss = float(primary_signal.stop_loss)
        
        else:  # "fvg_event" | "fvg_event_or_decay"
            # ================================================================
            # v4.0 方向语义修复：ST定方向，FVG定出手
            # - Direction: 从 st_trend_15m 获取（+1/-1），而非 fvg_event 的符号
            # - Event gate: fvg_event != 0 (保留 rising-edge guard 作为安全护栏)
            # - FVG_event 的 sign: 仅作为结构信息，用于 FVG_DIR_MISMATCH 过滤
            # ================================================================
            
            # 读取 fvg_event（用于事件门控）
            try:
                fvg_event_idx = self._schema.get_index("fvg_event")
                fvg_event = int(features[fvg_event_idx])
            except Exception:
                fvg_event = 0
            fvg_event_raw = fvg_event
            
            # 读取 st_trend_15m（用于方向决定/对齐）
            try:
                st_trend_15m_idx = self._schema.get_index("st_trend_15m")
                st_trend_15m = int(features[st_trend_15m_idx])
            except Exception:
                st_trend_15m = 0
            
            # Rising-edge guard（安全护栏）：
            # 即使 generator 有 cooldown，推理端也用 rising-edge 防止退化
            # event_present = (fvg_event != 0) AND (prev_fvg_event == 0)
            is_rising_edge = (fvg_event != 0) and (self._prev_fvg_event == 0)
            event_present = False
            
            # 方向来源可配置：默认用 st_trend_15m；如指定 primary 则用 trend_direction
            alignment_base_direction = st_trend_15m
            if self.config.trend_alignment_source == "primary":
                alignment_base_direction = trend_direction

            # FVG 方向（仅用于事件窗口/结构诊断）
            fvg_direction = 1 if fvg_event > 0 else (-1 if fvg_event < 0 else 0)
            fvg_event_val = fvg_event
            
            # 事件窗口：包含当前 bar，共 N 根 bar 进入推理
            window_bars = max(1, int(self.config.fvg_event_window_bars))
            if is_rising_edge:
                self._fvg_window_remaining = window_bars
                self._fvg_window_direction = alignment_base_direction if alignment_base_direction != 0 else fvg_direction
                event_present = True
                event_direction = alignment_base_direction if alignment_base_direction != 0 else fvg_direction
                event_direction_source = "rising_edge"
            elif self._fvg_window_remaining > 0:
                event_present = True
                event_direction = self._fvg_window_direction
                event_direction_source = "window"
            elif self._event_gate_mode == "fvg_event_or_decay":
                # 衰减窗口：FVG 活跃且年龄足够新时也可放行
                try:
                    fvg_age_idx = self._schema.get_index("fvg_age_bars")
                    fvg_age_bars = float(features[fvg_age_idx])
                except Exception:
                    fvg_age_bars = 0.0
                
                try:
                    bullish_idx = self._schema.get_index("bullish_fvg")
                    bearish_idx = self._schema.get_index("bearish_fvg")
                    bullish_active = int(features[bullish_idx]) != 0
                    bearish_active = int(features[bearish_idx]) != 0
                except Exception:
                    bullish_active = False
                    bearish_active = False
                
                decay_bars = max(1, int(self.config.fvg_decay_bars))
                if fvg_age_bars > 0 and fvg_age_bars <= decay_bars and (bullish_active or bearish_active):
                    if alignment_base_direction != 0:
                        event_direction = alignment_base_direction
                    elif bullish_active and not bearish_active:
                        event_direction = 1
                    elif bearish_active and not bullish_active:
                        event_direction = -1
                    else:
                        event_direction = 0
                    
                    if event_direction != 0:
                        event_present = True
                        event_direction_source = "decay"
            
            if event_present:
                in_event_window = True
            event_window_remaining = self._fvg_window_remaining if event_present else 0
            if event_present and self._fvg_window_remaining > 0:
                # 当前 bar 计入窗口消耗
                self._fvg_window_remaining = max(0, self._fvg_window_remaining - 1)
            
            # 更新状态（无论是否交易都要更新，保证状态连续）
            self._prev_fvg_event = fvg_event
            
            # 如果 PrimaryEngine 同时给出结构化入场/止损，则优先使用（更贴近实盘）
            if primary_signal is not None:
                entry_price = float(primary_signal.entry_price)
                stop_loss = float(primary_signal.stop_loss)
            else:
                entry_price = float(bar.close)
                # v4.0 修复：使用 ATR 计算备用 SL，避免硬编码 0.0
                # 当 FVGEventCalculator 检测到事件但 PrimaryEngine 未产生信号时触发
                try:
                    atr_idx = self._schema.get_index("atr_1m")
                    atr = float(features[atr_idx])
                    sl_buffer_factor = self.config.fallback_sl_atr_factor
                    if event_direction == 1:  # LONG
                        stop_loss = bar.close - sl_buffer_factor * atr
                    else:  # SHORT
                        stop_loss = bar.close + sl_buffer_factor * atr
                except Exception:
                    # 如果无法获取 ATR，仍使用 0.0（会被 NO_STOP_LOSS 过滤）
                    stop_loss = 0.0
        
        # 初始化结果
        result = InferenceResult(
            bar_idx=self._bar_count,
            close_price=bar.close,
            has_signal=event_present,
            direction=event_direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            meta_confidence=0.0,
            should_trade=False,
            filtered_reason="",
            market_phase=market_phase,
            market_temperature=market_temperature or 0.0,
            market_entropy=market_entropy or 0.0,
            trend_duration=trend_duration,
            # 诊断字段（v4.0 方向语义修复）
            st_trend_15m=st_trend_15m,
            fvg_event=fvg_event_val,
            trend_direction=trend_direction,
            features=features,
        )

        dynamic_threshold = None
        gate_stage = None
        trend_alignment = "UNKNOWN"
        alignment_direction = 0

        def _log_bar_diagnostic() -> None:
            logger.info(
                "bar_diagnostic",
                bar_idx=self._bar_count,
                close_price=round(result.close_price, 4),
                event_gate_mode=self._event_gate_mode,
                event_present=event_present,
                event_direction=int(event_direction),
                fvg_window_bars=int(self.config.fvg_event_window_bars),
                in_event_window=in_event_window,
                event_window_remaining=event_window_remaining,
                event_direction_source=event_direction_source,
                fvg_event=fvg_event_val,
                fvg_event_raw=fvg_event_raw,
                prev_fvg_event=prev_fvg_event,
                is_rising_edge=is_rising_edge,
                st_trend_15m=st_trend_15m,
                trend_alignment_source=self.config.trend_alignment_source,
                alignment_trend=alignment_trend if "alignment_trend" in locals() else st_trend_15m,
                trend_alignment=trend_alignment,
                structure_bos=structure_bos,
                structure_choch=structure_choch,
                structure_tl_break=structure_tl_break,
                structure_dir=self._structure_direction,
                prev_swing_high=round(prev_swing_high, 4) if prev_swing_high else 0.0,
                prev_swing_low=round(prev_swing_low, 4) if prev_swing_low else 0.0,
                last_structure_event_bar=self._last_structure_event_bar,
                last_structure_event_dir=self._last_structure_event_dir,
                market_phase=market_phase,
                ts_phase=ts_phase_value,
                market_temperature=market_temperature,
                market_entropy=market_entropy,
                dynamic_threshold=dynamic_threshold,
                gate_stage=str(gate_stage) if gate_stage is not None else None,
                meta_confidence=round(result.meta_confidence, 4),
                should_trade=result.should_trade,
                filtered_reason=result.filtered_reason,
            )
        
        # 检查是否应该交易
        if not self._is_warmed_up:
            result.filtered_reason = f"WARMUP ({self._bar_count}/{self.config.warmup_bars})"
            _log_bar_diagnostic()
            return result
        
        if not event_present:
            result.filtered_reason = f"NO_EVENT (mode={self._event_gate_mode})"
            _log_bar_diagnostic()
            return result
        
        # ================================================================
        # v4.0 阈值调制：ST(15m) 不裁决方向，仅调制 quantile
        # - ALIGNED: fvg_direction == st_trend_15m → 更宽松的阈值
        # - COUNTER: fvg_direction != st_trend_15m → 更严格的阈值
        # - UNKNOWN: st_trend_15m == 0 → 使用 phase 默认阈值
        # ================================================================
        
        # 计算 trend alignment（用于阈值调制）
        # 重要：必须用“实际用于交易/门控的方向”（event_direction，包含 window/decay 续航）
        # 而不是当前 bar 的 sign(fvg_event)（window bar 通常 fvg_event=0，会导致对齐误判为 COUNTER）
        alignment_direction = int(event_direction)
        alignment_trend = st_trend_15m
        if self.config.trend_alignment_source == "primary":
            alignment_trend = trend_direction
        if alignment_trend == 0 or alignment_direction == 0:
            trend_alignment = "UNKNOWN"
        elif alignment_direction == alignment_trend:
            trend_alignment = "ALIGNED"
        else:
            trend_alignment = "COUNTER"

        # 逆势结构确认（可选）：仅在 COUNTER 时启用
        if trend_alignment == "COUNTER" and self.config.counter_trend_require_structure:
            window = int(self.config.counter_trend_confirm_window_bars)
            bars_since = self._bar_count - self._last_structure_event_bar
            structure_ok = (
                self._last_structure_event_dir != 0 and
                alignment_direction != 0 and
                self._last_structure_event_dir == alignment_direction and
                (window > 0 and bars_since <= window)
            )
            if not structure_ok:
                result.filtered_reason = (
                    f"COUNTER_TREND_NO_STRUCTURE (window={window}, "
                    f"last_dir={self._last_structure_event_dir}, bars_since={bars_since})"
                )
                _log_bar_diagnostic()
                return result
        
        # 计算 meta_confidence
        result.meta_confidence = self._compute_meta_confidence(features, cfc_hidden_output)
        
        # 更新置信度 buffer（无论是否交易都更新，用于 rolling quantile）
        self._update_confidence_buffer(result.meta_confidence)
        
        # 计算动态阈值（基于 confidence_gate 配置 + trend alignment 调制）
        dynamic_threshold, gate_stage = self._compute_confidence_threshold(market_phase, trend_alignment)
        
        # === Cold Start Stage 检查 ===
        # Stage 1: HARD_FROZEN - 禁止交易（buffer 分布未形成）
        if gate_stage == ConfidenceGateStage.HARD_FROZEN:
            buffer_len = len(self._confidence_buffer)
            min_req = self.config.confidence_gate.min_required
            result.filtered_reason = f"COLD_START ({buffer_len}/{min_req} samples)"
            _log_bar_diagnostic()
            return result
        
        # Stage 2/3: 检查置信度阈值
        if result.meta_confidence < dynamic_threshold:
            gate_mode = self.config.confidence_gate.mode
            # 在 FIXED_FALLBACK 阶段，额外标注
            stage_hint = f", stage={gate_stage}" if gate_stage == ConfidenceGateStage.FIXED_FALLBACK else ""
            result.filtered_reason = (
                f"LOW_CONFIDENCE ({result.meta_confidence:.3f} < {dynamic_threshold:.3f}, "
                f"mode={gate_mode}, phase={market_phase}{stage_hint})"
            )
            _log_bar_diagnostic()
            return result
        
        # 检查相位（如果需要且 confidence_gate 模式不是 rolling_quantile）
        # 注意：rolling_quantile 模式下，相位已经参与了阈值计算，不再单独过滤
        if self.config.require_phase_transition:
            if self.config.confidence_gate.mode == "fixed":
                # fixed 模式下保持原有相位过滤逻辑
                if market_phase != "TRANSITION":
                    result.filtered_reason = f"WRONG_PHASE ({market_phase})"
                    _log_bar_diagnostic()
                    return result
            # rolling_quantile 模式下，相位已融入阈值，不再单独过滤
        
        # ================================================================
        # NO_STOP_LOSS: 强制 SL 非零（禁止裸仓）
        # 如果 should_trade=True 但 stop_loss <= 0，则拒绝交易
        # ================================================================
        if result.stop_loss <= 0:
            result.filtered_reason = (
                f"NO_STOP_LOSS (SL={result.stop_loss:.2f}, "
                f"primary_signal={primary_signal is not None})"
            )
            logger.warning(
                f"NO_STOP_LOSS filter triggered at bar {self._bar_count}: "
                f"entry={result.entry_price:.2f}, SL={result.stop_loss:.2f}, "
                f"direction={result.direction}"
            )
            _log_bar_diagnostic()
            return result
        
        # 所有检查通过
        result.should_trade = True
        
        # 信号通过时输出详细诊断信息，便于人工排查
        # 追加关键特征：FVG/ST、相位、风险、结构（结构字段暂无来源则为 None）
        try:
            temp_idx = self._schema.get_index("market_temperature")
            ent_idx = self._schema.get_index("market_entropy")
            ts_idx = self._schema.get_index("ts_phase")
            market_temperature = round(float(features[temp_idx]), 4)
            market_entropy = round(float(features[ent_idx]), 4)
            ts_phase = int(features[ts_idx])
        except Exception:
            market_temperature = None
            market_entropy = None
            ts_phase = None

        risk_distance = round(abs(result.entry_price - result.stop_loss), 4)

        logger.info(
            "推理信号通过",
            bar_idx=self._bar_count,
            direction=result.direction,
            entry_price=round(result.entry_price, 4),
            stop_loss=round(result.stop_loss, 4),
            risk_distance=risk_distance,
            meta_confidence=round(result.meta_confidence, 4),
            market_phase=market_phase,
            ts_phase=ts_phase,
            market_temperature=market_temperature,
            market_entropy=market_entropy,
            trend_alignment=trend_alignment,
            st_trend_15m=st_trend_15m,
            fvg_window_bars=int(self.config.fvg_event_window_bars),
            in_event_window=in_event_window,
            event_window_remaining=event_window_remaining,
            event_direction_source=event_direction_source,
            fvg_event=fvg_event_val,
            trend_direction=trend_direction,
            structure_bos=structure_bos,
            structure_choch=structure_choch,
            structure_tl_break=structure_tl_break,
            dynamic_threshold=round(dynamic_threshold, 4),
            gate_stage=str(gate_stage),
            buffer_len=len(self._confidence_buffer),
            primary_signal=primary_signal is not None,
        )
        _log_bar_diagnostic()
        return result
    
    def _update_confidence_buffer(self, confidence: float) -> None:
        """
        更新置信度滚动 buffer
        
        用于 rolling_quantile 阈值模式下计算动态阈值
        """
        self._confidence_buffer.append(confidence)
        
        # 保持 buffer 大小不超过配置值
        max_size = self.config.confidence_gate.buffer_size
        if len(self._confidence_buffer) > max_size:
            self._confidence_buffer = self._confidence_buffer[-max_size:]
    
    def _compute_confidence_threshold(
        self, 
        market_phase: str, 
        trend_alignment: str = "UNKNOWN"
    ) -> tuple[float, str]:
        """
        计算动态置信度阈值和门控阶段
        
        模式：
        - fixed: 使用固定 min_confidence（向后兼容）
        - rolling_quantile: 使用滚动分位数，并按相位分层 + trend alignment 调制
        
        Cold Start Semantics (Hybrid 3-Stage, 语义冻结):
        - Stage 1 HARD_FROZEN: buffer_size < min_required
          → should_trade=False, 禁止交易
        - Stage 2 FIXED_FALLBACK: min_required <= buffer_size < full_buffer
          → threshold = fixed_fallback_threshold
        - Stage 3 ROLLING_QUANTILE: buffer_size >= full_buffer
          → threshold = rolling_quantile(q_combined)
        
        Trend Alignment 调制（仅 Stage 3）：
        - 以 phase quantile 为基线
        - ALIGNED: quantile = min(phase_q, q_aligned) → 更宽松
        - COUNTER: quantile = max(phase_q, q_counter) → 更严格
        - UNKNOWN: quantile = phase_q（无调制）
        
        Args:
            market_phase: 当前市场相位 (FROZEN/LAMINAR/TURBULENT/TRANSITION)
            trend_alignment: 趋势对齐状态 (ALIGNED/COUNTER/UNKNOWN)
            
        Returns:
            (threshold, gate_stage): 阈值和当前门控阶段
        """
        gate_config = self.config.confidence_gate
        buffer_len = len(self._confidence_buffer)
        
        if gate_config.mode == "fixed":
            # 固定模式：使用 fixed_fallback_threshold
            threshold = gate_config.fixed_fallback_threshold
            gate_stage = ConfidenceGateStage.ROLLING_QUANTILE
        
        elif gate_config.mode == "rolling_quantile":
            # Rolling quantile 模式：Hybrid 3-Stage Cold Start
            
            # Stage 1: HARD_FROZEN（禁止交易）
            if buffer_len < gate_config.min_required:
                # 返回一个不可能达到的阈值，调用方会根据 stage 直接禁止交易
                return float("inf"), ConfidenceGateStage.HARD_FROZEN
            
            # Stage 2: FIXED_FALLBACK（使用固定阈值）
            if buffer_len < gate_config.full_buffer:
                threshold = gate_config.fixed_fallback_threshold
                gate_stage = ConfidenceGateStage.FIXED_FALLBACK
            else:
                # Stage 3: ROLLING_QUANTILE（正常滚动分位数 + trend alignment 调制）
                # Step 1: 获取 phase 基线分位数
                phase_q = gate_config.quantile_by_phase.get(
                    market_phase,
                    gate_config.base_quantile  # 默认使用基础分位数
                )
                
                # Step 2: Trend alignment 调制（如果配置了 quantile_by_trend）
                quantile = phase_q  # 默认使用 phase 分位数
                qbt = gate_config.quantile_by_trend
                if qbt:
                    # 兼容处理：若用户填 0.90/0.97，自动转换为 90/97
                    q_aligned = qbt.get("ST_ALIGNED", phase_q)
                    q_counter = qbt.get("ST_COUNTER", phase_q)
                    if q_aligned < 1.0:
                        q_aligned = q_aligned * 100
                    if q_counter < 1.0:
                        q_counter = q_counter * 100
                    
                    if trend_alignment == "ALIGNED":
                        # 顺势：更宽松（取较小分位数）
                        quantile = min(phase_q, q_aligned)
                    elif trend_alignment == "COUNTER":
                        # 逆势：更严格（取较大分位数）
                        quantile = max(phase_q, q_counter)
                    # UNKNOWN: 保持 phase_q 不变
                
                # 计算分位阈值
                threshold = float(np.percentile(self._confidence_buffer, quantile))
                
                # 应用 floor/ceiling 保护
                threshold = max(gate_config.floor, min(gate_config.ceiling, threshold))
                gate_stage = ConfidenceGateStage.ROLLING_QUANTILE
        
        else:
            # 未知模式，回退到 fixed
            logger.warning(f"Unknown confidence gate mode: {gate_config.mode}, falling back to fixed")
            threshold = gate_config.fixed_fallback_threshold
            gate_stage = ConfidenceGateStage.ROLLING_QUANTILE

        # 逆势阈值提升（额外严格）
        if trend_alignment == "COUNTER":
            boost = float(getattr(self.config, "counter_trend_threshold_boost", 0.0))
            if boost > 0:
                threshold = threshold + boost
                if gate_config.mode == "rolling_quantile":
                    threshold = max(gate_config.floor, min(gate_config.ceiling, threshold))
                threshold = min(1.0, threshold)

        return threshold, gate_stage
    
    def _update_cfc_hidden_state(
        self,
        features: NDArray,
        delta_t: float,
    ) -> Tensor | None:
        """
        更新 CfC 隐状态（单步在线推理）
        
        ⚠️ SEMANTIC CONSTRAINT (架构约束)
        =================================
        CfC hidden states represent LATENT SYSTEM STATES, not interpretable features.
        
        ❌ 禁止对返回的 hidden state：
          - 做标准化（Normalization/StandardScaler）
          - 做 SHAP / 特征重要性分析当"因子"解读
          - 将某个维度当作有物理意义的指标
          - 做跨 bar 的滚动统计
        
        CfC 回答的是 "what happens next?"（时序动态建模）
        
        Args:
            features: 当前特征向量（全量）
            delta_t: 时间间隔（秒）
            
        Returns:
            CfC 编码输出（如果有 CfC），否则 None
        """
        if self._cfc_encoder is None or self._cfc_hidden_states is None:
            return None
        
        try:
            # 使用 LNN 特征子集（如果启用特征分离）
            if self._use_feature_split and self._lnn_indices:
                features_lnn = features[self._lnn_indices]
            else:
                features_lnn = features
            
            # 转换为 Tensor
            x = torch.tensor(
                features_lnn.reshape(1, -1),
                dtype=torch.float32,
                device=self._device,
            )
            dt = torch.tensor([delta_t], dtype=torch.float32, device=self._device)
            
            # 单步编码
            with torch.no_grad():
                output, self._cfc_hidden_states = self._cfc_encoder.encode_single_step(
                    x, dt, self._cfc_hidden_states
                )
            
            return output
            
        except Exception as e:
            logger.warning(f"CfC update failed: {e}")
            return None
    
    def _compute_meta_confidence(
        self,
        features: NDArray,
        cfc_hidden: Tensor | None,
    ) -> float:
        """
        计算 meta_confidence
        
        ⚠️ SEMANTIC CONSTRAINT (架构约束)
        =================================
        XGB 的角色是 FILTER / CONDITIONER，不是 DECISION MAKER。
        
        - CfC 回答 "what happens next?"（时序动态）
        - XGB 回答 "is this a tradable instance?"（可交易性评估）
        
        meta_confidence 应与其他 filters（warmup, signal, phase, risk）
        共同决策，而非单独决定是否交易。
        
        ❌ 禁止：
          - 让 XGB 单独决定交易方向（方向来自 PrimaryEngine）
          - 让 XGB 学会 "regime 好就 always trade"
          - 忽略其他 filters 只看 meta_confidence
        
        如果有 CfC + XGBoost：使用 concat([xgb_features, cfc_hidden]) 作为输入
        如果只有 XGBoost：使用 xgb_features 作为输入
        如果都没有：返回默认值 0.5
        
        Args:
            features: 当前特征向量（全量）
            cfc_hidden: CfC 编码输出（可选）
            
        Returns:
            meta_confidence [0, 1]
        """
        if self._xgb_model is None:
            return 0.5
        
        try:
            # 使用 XGB 特征子集（如果启用特征分离）
            if self._use_feature_split and self._xgb_indices:
                features_xgb = features[self._xgb_indices]
            else:
                features_xgb = features
            
            if cfc_hidden is not None:
                # CfC + XGBoost 模式：组合特征
                hidden_np = cfc_hidden.cpu().numpy().flatten()
                X = np.concatenate([features_xgb, hidden_np]).reshape(1, -1)
            else:
                # XGBoost-only 模式
                X = features_xgb.reshape(1, -1)
            
            proba = self._xgb_model.predict_proba(X)[0, 1]
            
            # Diagnostic logging for near-zero confidence (helps debug confidence collapse)
            if proba < self.config.diagnostic_confidence_threshold:
                # 采样诊断日志（避免刷屏）
                if (
                    self._bar_count % self.config.diagnostic_log_interval == 0
                    or self._bar_count < self.config.diagnostic_log_initial_bars
                ):
                    # Feature summary for debugging
                    feat_min = float(np.min(X))
                    feat_max = float(np.max(X))
                    feat_mean = float(np.mean(X))
                    nan_count = int(np.sum(np.isnan(X)))
                    inf_count = int(np.sum(np.isinf(X)))
                    
                    logger.warning(
                        f"XGB near-zero confidence: proba={proba:.6f}, "
                        f"bar={self._bar_count}, features_shape={X.shape}, "
                        f"has_cfc={cfc_hidden is not None}, "
                        f"feat_range=[{feat_min:.4f}, {feat_max:.4f}], "
                        f"feat_mean={feat_mean:.4f}, "
                        f"nan={nan_count}, inf={inf_count}"
                    )
            
            return float(proba)
            
        except Exception as e:
            logger.warning(f"Meta confidence prediction failed: {e}")
            return 0.5
    
    def get_current_state(self) -> dict:
        """获取当前状态"""
        # 计算当前动态阈值和门控阶段
        current_threshold, gate_stage = self._compute_confidence_threshold("TRANSITION")  # 使用 TRANSITION 作为基准
        gate_config = self.config.confidence_gate
        buffer_len = len(self._confidence_buffer)
        
        return {
            "bar_count": self._bar_count,
            "tick_count": self._tick_count,
            "is_warmed_up": self._is_warmed_up,
            "primary_direction": self._primary_engine.current_direction if self._primary_engine else 0,
            "active_fvgs": self._primary_engine.active_fvg_count if self._primary_engine else 0,
            "feature_buffer_size": len(self._feature_buffer),
            "schema_hash": self._schema.schema_hash,
            "use_feature_split": self._use_feature_split,
            "lnn_features": len(self._lnn_indices) if self._use_feature_split else self._schema.num_features,
            "xgb_features": len(self._xgb_indices) if self._use_feature_split else self._schema.num_features,
            # 置信度门控状态（Hybrid 3-Stage Cold Start）
            "confidence_gate_mode": gate_config.mode,
            "confidence_gate_stage": gate_stage,
            "confidence_buffer_size": buffer_len,
            "confidence_buffer_min_required": gate_config.min_required,
            "confidence_buffer_full_buffer": gate_config.full_buffer,
            "fixed_fallback_threshold": gate_config.fixed_fallback_threshold,
            "current_threshold": current_threshold if gate_stage != ConfidenceGateStage.HARD_FROZEN else None,
        }
    
    def get_feature_snapshot(self) -> dict:
        """获取当前特征快照"""
        if self._last_features is None:
            return {}
        
        return {
            name: float(self._last_features[i])
            for i, name in enumerate(self._schema.feature_names)
        }
    
    def reset(self) -> None:
        """重置引擎状态"""
        self._bar_count = 0
        self._tick_count = 0
        self._is_warmed_up = False
        self._last_bar = None
        self._last_features = None
        self._last_delta_t = 1.0
        self._feature_buffer.clear()
        self._delta_t_buffer.clear()
        
        # 重置置信度 buffer
        self._confidence_buffer.clear()
        
        # 重置 rising-edge guard 状态
        self._prev_fvg_event = 0
        self._fvg_window_remaining = 0
        self._fvg_window_direction = 0
        
        if self._sampler:
            self._sampler.reset()
        if self._feature_pipeline:
            self._feature_pipeline.reset()
        if self._primary_engine:
            self._primary_engine.reset()
        
        # 重置 CfC 隐状态
        if self._cfc_encoder is not None:
            self._cfc_hidden_states = self._cfc_encoder.init_hidden_states(
                batch_size=1,
                device=self._device,
            )
        
        logger.info("InferenceEngineV4 reset")
    
    @property
    def schema(self) -> FeatureSchema:
        """获取 Feature Schema"""
        return self._schema
    
    @property
    def is_ready(self) -> bool:
        """是否准备好生成信号"""
        return self._is_warmed_up and self._primary_engine.is_ready
    
    @property
    def bar_count(self) -> int:
        """已处理的 Bar 数"""
        return self._bar_count
    
    @property
    def tick_count(self) -> int:
        """已处理的 Tick 数"""
        return self._tick_count
