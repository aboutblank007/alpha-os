"""
AlphaOS Configuration Management

Uses Hydra/OmegaConf for hierarchical configuration with:
- Environment variable interpolation
- Type validation via Pydantic
- Symbol-specific overrides
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field


# ============================================================================
# Pydantic Models for Type-Safe Configuration
# ============================================================================

class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    type: str = "postgresql"
    host: str = "localhost"
    port: int = 5432
    name: str = "alphaos"
    user: str = "alphaos"
    password: str = ""
    pool_size: int = 5
    
    @property
    def connection_string(self) -> str:
        """Generate SQLAlchemy connection string."""
        if self.type in ("postgresql", "timescaledb"):
            # TimescaleDB is PostgreSQL with extension, uses same connection string
            return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        elif self.type == "clickhouse":
            return f"clickhouse://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        raise ValueError(f"Unsupported database type: {self.type}")


class ZeroMQConfig(BaseModel):
    """ZeroMQ communication configuration."""
    tick_endpoint: str = "tcp://127.0.0.1:5555"
    order_endpoint: str = "tcp://127.0.0.1:5556"
    history_endpoint: str = "tcp://127.0.0.1:5559"  # For GET_HISTORY requests
    heartbeat_interval_ms: int = 1000
    reconnect_delay_ms: int = 5000
    recv_timeout_ms: int = 100
    history_timeout_ms: int = 30000  # For GET_HISTORY requests
    history_snd_timeout_ms: int = 5000
    order_recv_timeout_ms: int = 10000
    order_snd_timeout_ms: int = 5000
    tick_staleness_threshold_sec: int = 60


class KalmanDenoiseConfig(BaseModel):
    """Kalman filter denoising configuration (v4.0)."""
    enabled: bool = False
    process_variance: float = 0.1    # Q: state transition noise
    measurement_variance: float = 1.0  # R: observation noise
    initial_uncertainty: float = 100.0  # P_0: initial estimation variance
    use_adaptive: bool = False        # Use adaptive noise estimation


class WaveletDenoiseConfig(BaseModel):
    """Wavelet denoising configuration (v4.0) - for training only."""
    enabled: bool = False
    wavelet: str = "db4"              # Wavelet type
    level: int | None = None          # Decomposition level (None = auto)
    threshold_mode: str = "soft"       # 'soft' or 'hard'
    threshold_rule: str = "universal"  # 'universal', 'minimax', 'sure'


class DenoiseConfig(BaseModel):
    """Denoising configuration (v4.0)."""
    kalman: KalmanDenoiseConfig = Field(default_factory=KalmanDenoiseConfig)
    wavelet: WaveletDenoiseConfig = Field(default_factory=WaveletDenoiseConfig)


class FeatureConfig(BaseModel):
    """Feature engineering parameters."""
    tick_intensity_alpha: float = 0.1
    tick_rule_gamma: float = 0.95
    tick_rule_threshold: float = 0.5
    ofi_window_ticks: int = 50
    kyle_lambda_window: int = 20
    temperature_window: int = 100
    entropy_window: int = 50
    volatility_lambda: float = 0.94
    # Thermodynamic thresholds for T-S phase classification
    min_temperature: float = 0.5  # Below this = low temperature (frozen/laminar)
    max_entropy: float = 0.7  # Above this = high entropy (disordered)
    # v4.0: Denoising configuration
    denoise: DenoiseConfig = Field(default_factory=DenoiseConfig)


class CfCModelConfig(BaseModel):
    """CfC neural network configuration."""
    input_dim: int = 14  # Default feature count
    hidden_dim: int = 64
    num_layers: int = 2
    backbone: str = "cfc"
    sparsity: float = 0.5
    
    # Adaptive Time Scale
    use_adaptive_time: bool = True
    baseline_intensity: float = 0.66  # Baseline tick frequency (Hz)


class XGBoostConfig(BaseModel):
    """
    XGBoost decision head configuration.
    
    v4 Notes:
    - eval_metric: Prefer PR/ROC metrics for imbalanced data
    - log_eval_period: Structured logging for training progress
    - GPU acceleration: tree_method='hist' + device='cuda'
    - NOTE: gpu_hist is DEPRECATED in XGBoost 2.0+, use 'hist' instead
    - max_bin parameter for GPU histogram optimization (256 default)
    - sampling_method='gradient_based' for GPU-optimized sampling
    """
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    objective: str = "multi:softprob"
    num_class: int = 3
    tree_method: str = "hist"  # ALWAYS use "hist" (gpu_hist deprecated)
    device: str = "cuda"  # Use "cuda" for GPU, "cpu" for CPU
    max_bin: int = 256  # GPU histogram bin count
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    log_eval_period: int = 100  # Log training progress every N iterations
    eval_metric: str = "mlogloss"  # Use "auc" for binary, "mlogloss" for multi-class
    early_stopping_rounds: int = 10  # Patience for early stopping


class ModelConfig(BaseModel):
    """Combined model configuration."""
    cfc: CfCModelConfig = Field(default_factory=CfCModelConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)


class TrendBalanceConfig(BaseModel):
    """Trend-segment balancing configuration for training data."""
    enabled: bool = False
    segment_size: int = 5000  # ticks per segment
    up_threshold_pct: float = 0.3  # segment return >= threshold
    down_threshold_pct: float = 0.3  # segment return <= -threshold
    seed: int = 42


class ValidationCVConfig(BaseModel):
    """Cross-validation configuration (v4.0)."""
    mode: str = "purged_kfold"  # "purged_kfold" | "cpcv"
    
    # Purged KFold parameters
    n_splits: int = 5
    
    # CPCV parameters
    cpcv_n_groups: int = 6
    cpcv_n_test_groups: int = 2
    cpcv_max_combinations: int | None = None
    
    # Shared parameters
    purge_gap: int = 100
    embargo_pct: float = 0.01


class TrainingConfig(BaseModel):
    """Training configuration."""
    barrier_m_up: float = 2.0
    barrier_m_down: float = 1.5
    barrier_tau_max: int = 300
    barrier_tau_mode: str = "ticks"  # "ticks" or "intensity"
    barrier_intensity_threshold: float = 100.0
    volatility_lambda: float = 0.94  # EWMA decay for recursive volatility
    volume_mask_prob: float = 0.5
    n_splits: int = 5
    purge_gap: int = 100
    embargo_pct: float = 0.01
    batch_size: int = 256
    sequence_length: int = 100
    epochs: int = 50
    pretrain_epochs: int | None = None  # 预训练轮数，默认 epochs // 2
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 10
    
    # Minimum volatility threshold (per 黄金价格波动分析与模型调整.md)
    # Units: percent log-return (e.g., 0.02 = 0.02%)
    min_sigma_pct: float = 0.02

    # Trend-balanced sampling for label skew mitigation
    trend_balance: "TrendBalanceConfig" = Field(default_factory=lambda: TrendBalanceConfig())
    
    # M2 Pro Memory Optimization
    micro_batch_size: int = 32  # Actual batch size for forward pass
    accumulation_steps: int = 8  # Effective batch = micro_batch * accumulation_steps
    use_mixed_precision: bool = True  # Enable FP16 mixed precision
    mps_memory_management: bool = True  # Enable explicit MPS cache clearing
    
    # GPU Optimization - Larger batch for inference-only encoding
    encoding_batch_size: int = 65536  # Batch size for hidden state pre-computation (GPU inference)
    
    # Contrastive pretraining options
    pretrain_mode: str = "regression"  # "regression" or "contrastive"
    contrastive_temperature: float = 0.1
    contrastive_noise_std: float = 0.01
    contrastive_volume_mask_prob: float = 0.3
    contrastive_time_warp: float = 0.1
    
    # HDF5 Memory Optimization (for large datasets)
    use_hdf5: bool = True  # Enable HDF5 disk-based storage for large datasets
    hdf5_chunk_size: int = 50000  # Samples per chunk during data preparation
    hdf5_cache_dir: str = "data/cache"  # Directory for HDF5 cache files
    memory_threshold_gb: float = 12.0  # Auto-enable HDF5 when estimated memory exceeds this


class RiskConfig(BaseModel):
    """Risk management configuration."""
    min_position_lots: float = 0.2   # 动态手数最小值
    max_position_lots: float = 1.0   # 动态手数最大值
    max_position_usd: float = 10000.0
    max_daily_loss_pct: float = 2.0
    max_consecutive_losses: int = 5
    min_temperature: float = 0.5
    max_entropy: float = 0.7
    gate: "RiskGateConfig" = Field(default_factory=lambda: RiskGateConfig())


class TSPhaseConfig(BaseModel):
    """T-S Phase Diagram thresholds for execution."""
    temp_low: float = 0.3
    temp_high: float = 0.5
    entropy_low: float = 0.4
    entropy_high: float = 0.6


class PhaseStrategyConfig(BaseModel):
    """
    Per-phase execution strategy configuration (v4.0).
    
    Allows different thresholds and behaviors for each T-S phase.
    """
    allow_entry: bool = True
    min_meta_confidence: float = 0.52
    min_signal_confidence: float = 0.52
    order_type: str | None = None  # None = use TSPhaseClassifier default


class PhaseRoutingConfig(BaseModel):
    """
    Phase-based routing configuration (v4.0).
    
    Replaces the single `only_phase_transition` gate with per-phase 
    thresholds and allow_entry flags. This enables "all-weather" trading
    with phase-appropriate strategies.
    
    When enabled:
    - Gateway skips its hard meta_confidence gate
    - Gateway skips its only_phase_transition gate  
    - OrderManager uses per-phase thresholds from this config
    """
    enabled: bool = False
    
    # Per-phase strategies (keys match TSPhase enum names)
    laminar: PhaseStrategyConfig = Field(
        default_factory=lambda: PhaseStrategyConfig(
            allow_entry=True,
            min_meta_confidence=0.55,  # Higher threshold for trend-following
            min_signal_confidence=0.55,
            order_type="limit",
        )
    )
    turbulent: PhaseStrategyConfig = Field(
        default_factory=lambda: PhaseStrategyConfig(
            allow_entry=True,
            min_meta_confidence=0.50,  # Lower threshold, mean-reversion
            min_signal_confidence=0.50,
            order_type="limit",
        )
    )
    phase_transition: PhaseStrategyConfig = Field(
        default_factory=lambda: PhaseStrategyConfig(
            allow_entry=True,
            min_meta_confidence=0.52,  # Aggressive entry on breakout
            min_signal_confidence=0.52,
            order_type="market",
        )
    )
    frozen: PhaseStrategyConfig = Field(
        default_factory=lambda: PhaseStrategyConfig(
            allow_entry=False,  # No trading in frozen market
            min_meta_confidence=0.99,
            min_signal_confidence=0.99,
            order_type="none",
        )
    )


class ExecutionOrderFilterConfig(BaseModel):
    """自动单隔离配置（仅管理自身 magic）。"""
    only_magic: int | None = None
    ignore_magic_zero: bool = True
    ignore_manual_positions: bool = True
    manage_existing_positions: bool = True


class ExecutionPriceGuardConfig(BaseModel):
    """入场价格限制（仅对自动单生效）。"""
    enabled: bool = False
    max_spread_points: float = 0.0
    max_slippage_points: float = 0.0
    entry_band_points: float = 0.0
    reject_if_outside_band: bool = True


class ExecutionGateSideConfig(BaseModel):
    """Execution gate settings for entry/exit."""
    price_guard: ExecutionPriceGuardConfig = Field(default_factory=ExecutionPriceGuardConfig)
    allow_force_market_close: bool = True


class ExecutionGateConfig(BaseModel):
    """Execution gate configuration with entry/exit split."""
    entry: ExecutionGateSideConfig = Field(default_factory=ExecutionGateSideConfig)
    exit: ExecutionGateSideConfig = Field(
        default_factory=lambda: ExecutionGateSideConfig(
            price_guard=ExecutionPriceGuardConfig(enabled=False),
            allow_force_market_close=True,
        )
    )


class ExecutionCloseBackoffConfig(BaseModel):
    """平仓重试退避，防止 REJECTED 风暴。"""
    enabled: bool = True
    max_attempts: int = 5
    cooldown_sec: float = 2.0


class PositionSizingConfig(BaseModel):
    """
    Position sizing configuration (v4.0).
    
    Supports:
    - fixed: Fixed lot size
    - linear: Linear interpolation based on confidence
    - kelly: Kelly criterion based on calibrated probability
    """
    mode: str = "linear"  # "fixed" | "linear" | "kelly"
    
    # Kelly-specific parameters
    kelly_fraction: float = 0.25   # Fractional Kelly (25% = quarter Kelly)
    kelly_max_fraction: float = 0.5  # Maximum fraction of bankroll
    expected_edge_pct: float = 2.0  # Expected edge per trade (%)
    win_rate: float = 0.55         # Historical win rate (for Kelly calculation)
    
    # Risk parameters
    risk_per_trade_pct: float = 1.0  # Maximum risk per trade as % of account
    account_balance: float = 10000.0  # Account balance for sizing

    # 统一手数约束（execution.position_sizing SSOT）
    risk_reward_ratio: float = 3.0
    min_lots: float = 0.01
    max_lots: float = 0.10
    lot_step: float = 0.01
    linear_conf_min: float = 0.60
    linear_conf_max: float = 0.80


class AlignmentMultiplierConfig(BaseModel):
    """
    Alignment multipliers for Exit v2.1.
    
    Adjusts BE/Partial/Trail thresholds based on trend alignment:
    - ALIGNED: Trade direction matches trend → use wider thresholds (let winners run)
    - COUNTER: Trade direction opposes trend → use tighter thresholds (protect profits)
    - UNKNOWN: Trend unclear → use default thresholds
    """
    be_trigger_mult: float = 1.0        # Multiplier for BE trigger threshold
    trail_distance_mult: float = 1.0    # Multiplier for trailing distance


class ExitV21AlignmentConfig(BaseModel):
    """
    Exit v2.1 alignment/phase modulation configuration.
    
    Allows different thresholds for ALIGNED, COUNTER, and UNKNOWN trend states.
    """
    aligned: AlignmentMultiplierConfig = Field(
        default_factory=lambda: AlignmentMultiplierConfig(
            be_trigger_mult=1.3,        # Let winners run longer before BE
            trail_distance_mult=1.2,    # Wider trailing distance
        )
    )
    counter: AlignmentMultiplierConfig = Field(
        default_factory=lambda: AlignmentMultiplierConfig(
            be_trigger_mult=0.7,        # Tighter BE (protect profits sooner)
            trail_distance_mult=0.8,    # Tighter trailing
        )
    )
    unknown: AlignmentMultiplierConfig = Field(
        default_factory=lambda: AlignmentMultiplierConfig(
            be_trigger_mult=1.0,        # Default
            trail_distance_mult=1.0,
        )
    )


class ExitV21Config(BaseModel):
    """
    Exit v2.1 Configuration (bid/ask + cost guard + alignment/phase modulation).
    
    Key improvements over v2.0:
    - Bid/Ask price selection: LONG uses BID for SL hit/PnL, SHORT uses ASK
    - Cost guard: net_pnl = unrealized_pnl - commission - slippage - buffer
    - SL modification isolation: pending flag + cooldown to prevent rapid fire
    - Trend alignment modulation: adjust thresholds based on ALIGNED/COUNTER/UNKNOWN
    
    Reference: Exit v2.1 白皮书
    """
    
    # ======== Timing & Isolation ========
    min_hold_seconds: float = 30.0          # Minimum hold time before any exit action
    modify_cooldown_sec: float = 2.0        # Minimum seconds between SL modifications
    
    # ======== SL Validity ========
    min_sl_gap_price: float = 0.50          # Minimum gap from price to SL (in price units)
    price_precision: int = 2                # Decimal places for price rounding (XAUUSD = 2)
    
    # ======== Cost Estimation ========
    est_commission_usd_per_lot: float = 7.0   # Estimated round-trip commission per lot
    est_slippage_usd_per_lot: float = 2.0     # Estimated slippage per lot
    cost_buffer_usd: float = 1.0              # Additional buffer for safety
    
    # ======== Break Even (BE) Stage ========
    be_trigger_net_usd: float = 15.0        # Net P&L threshold to trigger BE
    be_offset_price: float = 0.50           # Price offset from entry for BE SL
    
    # ======== Partial Close Stage ========
    partial1_trigger_net_usd: float = 30.0  # Net P&L threshold for partial close
    partial1_ratio: float = 0.5             # Fraction of position to close
    min_lots_to_partial: float = 0.02       # Minimum remaining lots after partial
    post_partial_cooldown_sec: float = 5.0  # Cooldown after partial close
    
    # ======== Trailing Stop Stage ========
    trail_start_net_usd: float = 25.0       # Net P&L threshold to start trailing
    trail_distance_price: float = 3.0       # Distance from best price for trailing SL
    trail_step_price: float = 0.50          # Minimum price move to update trailing SL
    
    # ======== Alignment/Phase Modulation ========
    alignment_multipliers: ExitV21AlignmentConfig = Field(
        default_factory=ExitV21AlignmentConfig
    )
    
    # ======== Symbol-specific (XAUUSD defaults) ========
    tick_value_usd_per_lot: float = 1.0     # USD per point per lot (XAUUSD: $1/point/lot)


class ExecutionConfig(BaseModel):
    """Execution configuration."""
    warmup_ticks: int = 200
    min_signal_confidence: float = 0.6
    use_market_order_entropy_threshold: float = 0.3
    limit_order_offset_pips: float = 0.5
    
    # v4.0: Position sizing configuration
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)
    
    # 狙击模式 - 只在 PHASE_TRANSITION（高温+低熵）交易
    # 预期：交易频率从每小时 10-50 笔降到每天 5-10 笔
    only_phase_transition: bool = False
    
    # T-S Phase Diagram Thresholds
    ts_phase: TSPhaseConfig = Field(default_factory=TSPhaseConfig)
    
    # v4.0: Phase-based routing (replaces single gate with per-phase strategies)
    phase_routing: PhaseRoutingConfig = Field(default_factory=PhaseRoutingConfig)
    
    # v4.0: Exit v2.1 staged exit configuration (bid/ask + cost guard + alignment)
    exit_v21: ExitV21Config = Field(default_factory=ExitV21Config)

    # v4.1: 自动单隔离与价格限制
    order_filters: ExecutionOrderFilterConfig = Field(default_factory=ExecutionOrderFilterConfig)
    price_guard: ExecutionPriceGuardConfig = Field(default_factory=ExecutionPriceGuardConfig)
    execution_gate: ExecutionGateConfig = Field(default_factory=ExecutionGateConfig)
    close_backoff: ExecutionCloseBackoffConfig = Field(default_factory=ExecutionCloseBackoffConfig)


class RiskGateSideConfig(BaseModel):
    """Risk gate settings for entry/exit."""
    enforce_regime_filter: bool = True
    enforce_loss_limits: bool = True
    enforce_position_limits: bool = True


class RiskGateConfig(BaseModel):
    """Risk gate configuration with entry/exit split."""
    entry: RiskGateSideConfig = Field(default_factory=RiskGateSideConfig)
    exit: RiskGateSideConfig = Field(
        default_factory=lambda: RiskGateSideConfig(
            enforce_regime_filter=False,
            enforce_loss_limits=False,
            enforce_position_limits=False,
        )
    )


class ModelGuardianConfig(BaseModel):
    """
    Model Guardian configuration.
    
    Production safety kill switch that monitors model outputs
    and internal states for anomalies.
    """
    enabled: bool = True
    nan_inf_check: bool = True
    state_saturation_threshold: float = 100.0
    confidence_collapse_window: int = 10
    confidence_collapse_threshold: float = 0.1
    latency_threshold_ms: float = 500.0
    lock_file_path: str = "logs/model_guardian_lock.json"


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    prometheus_port: int = 9090
    metrics_prefix: str = "alphaos"
    enable_profiling: bool = False
    model_guardian: ModelGuardianConfig = Field(default_factory=ModelGuardianConfig)
    data_store: "DataStoreConfig" = Field(default_factory=lambda: DataStoreConfig())


class DataStoreConfig(BaseModel):
    """Database persistence configuration for realtime data."""
    enable_ticks: bool = True
    enable_bars: bool = True
    enable_decisions: bool = True
    enable_orders: bool = True
    enable_fills: bool = True
    enable_positions: bool = True
    batch_size: int = 1000
    flush_interval_sec: float = 1.0
    max_queue_size: int = 100000


class SymbolInfo(BaseModel):
    """Symbol-specific information."""
    digits: int = 2
    point: float = 0.01
    contract_size: float = 100.0
    tick_value_usd: float = 1.0
    spread_typical: float = 0.20


# ============================================================================
# Meta-Labeling and Stagewise Configuration
# (per Model Optimization with Hardware Support.md)
# ============================================================================

class MetaLabelingConfig(BaseModel):
    """
    Meta-Labeling configuration.
    
    Breaking Changes:
    - Meta-labeling is now MANDATORY (enabled always True)
    - Primary signal comes from Gateway (Percoco rules), not model
    - Removed primary_model_type, alpha191, momentum, high_recall configs
    - Model only outputs meta_confidence (binary classification)
    
    Gateway uses:
    - confidence_threshold: Minimum meta_confidence to execute a trade
    """
    enabled: bool = True  # Always enabled
    confidence_threshold: float = 0.6  # Minimum meta_confidence for trade execution


class PriceScalingConfig(BaseModel):
    """Price scaling attack configuration for Sim2Real."""
    enabled: bool = False
    scale_min: float = 0.5
    scale_max: float = 3.0
    attack_prob: float = 0.3


class LatencyInjectionConfig(BaseModel):
    """Latency injection attack configuration for Sim2Real."""
    enabled: bool = False
    min_delay_ticks: int = 1
    max_delay_ticks: int = 5
    attack_prob: float = 0.3


class Sim2RealConfig(BaseModel):
    """
    Sim2Real adversarial training configuration.
    
    Per 5090优化方案:
    - Volume masking to reduce dependence on tick intensity
    - Price scaling attack for price regime robustness
    - Latency injection for timing robustness
    """
    volume_mask_prob: float = 0.2
    price_scaling: PriceScalingConfig = Field(default_factory=PriceScalingConfig)
    latency_injection: LatencyInjectionConfig = Field(default_factory=LatencyInjectionConfig)


class InferenceXGBoostConfig(BaseModel):
    """XGBoost inference settings for M2 Pro deployment."""
    device: str = "cpu"
    tree_method: str = "hist"


class StagewiseConfig(BaseModel):
    """
    Stagewise training pipeline configuration.
    
    Per Model Optimization with Hardware Support.md Section 4.3:
    Stage 1: LNN Encoder Training
    Stage 2: Latent State Extraction (NumPy mmap)
    Stage 3: XGBoost Training on CPU
    """
    enabled: bool = False
    cache_dir: str = "data/latent_cache"
    inference_xgboost: InferenceXGBoostConfig = Field(default_factory=InferenceXGBoostConfig)


class SessionConfig(BaseModel):
    """Trading session definition."""
    name: str
    start: str = "00:00"
    end: str = "23:59"
    days: list[int] = Field(default_factory=list)


class SessionsConfig(BaseModel):
    """Session filter configuration."""
    active_sessions: list[SessionConfig] = Field(default_factory=list)
    excluded_periods: list[SessionConfig] = Field(default_factory=list)


class SamplingConfig(BaseModel):
    """
    Data Sampling Configuration (v4.0).
    
    Controls how tick data is sampled/aggregated:
    - time_bars: Traditional fixed-time bars (M1, M5, etc.)
    - tick_imbalance_bars: Event-driven bars based on directional imbalance
    - volume_bars: Event-driven bars based on cumulative volume threshold
    
    Reference:
    - "Advances in Financial Machine Learning" Chapter 2
    - 交易模型研究.md Section 2.3
    """
    mode: str = "time_bars"  # "time_bars" | "tick_imbalance_bars" | "volume_bars"
    
    # Tick Imbalance Bar parameters
    initial_expected_ticks: float = 50.0    # Initial E[T] estimate
    initial_expected_imbalance: float = 0.5  # Initial E[|θ|] estimate
    ewma_alpha: float = 0.1                  # EWMA decay for adaptive thresholds
    tick_rule_gamma: float = 0.95            # Bayesian tick rule decay
    tick_rule_threshold: float = 0.5         # Neutral classification threshold
    
    # Volume Bar parameters (v4.0)
    volume_bar_target: float = 10000.0       # Volume threshold per bar
    volume_bar_source: str = "tick_count"    # "real" | "tick_count" | "synthetic"


class PrimaryConfig(BaseModel):
    """
    Primary Signal Engine Configuration (v4.0).
    
    Configures the primary signal generation mode:
    - percoco: Legacy Percoco ChoCh→FVG→CE (M5/M15 EMA filter)
    - supertrend_fvg: PivotSuperTrend + FVG (structural trend + gap entries)
    
    The primary engine provides high-recall directional signals.
    The Meta-model (LNN+XGBoost) filters for precision.
    """
    mode: str = "percoco"  # "percoco" | "supertrend_fvg"
    
    # PivotSuperTrend parameters (for supertrend_fvg mode)
    pivot_lookback: int = 5       # Bars on each side to confirm pivot
    atr_period: int = 14          # ATR calculation period
    atr_factor: float = 2.0       # Multiplier for SuperTrend bands
    min_trend_duration: int = 3   # Minimum bars in trend before signals
    
    # FVG parameters (for supertrend_fvg mode)
    fvg_min_size_bps: float = 5.0   # Minimum FVG size in basis points
    fvg_max_age_bars: int = 20      # Maximum bars before FVG expires
    ce_tolerance_bps: float = 3.0   # Tolerance for CE midpoint entry
    
    # Signal generation
    cooldown_bars: int = 5         # M1 bars between signals
    sl_buffer_bps: float = 5.0     # Buffer added to structure stop loss
    
    # Risk/Reward
    risk_reward_ratio: float = 3.0  # TP = Entry + RR * Risk


class PercocoScalperConfig(BaseModel):
    """
    Percoco Scalper Mode Configuration.
    
    Multi-timeframe scalping strategy:
    - M5/M15 EMA trend filter (HTF intersection)
    - M1 ChoCh→FVG→CE trigger with Fib confluence
    - Event-based Meta-Labeling
    """
    enabled: bool = False
    
    # M5 Trend Filter
    m5_ema_periods: list[int] = Field(default_factory=lambda: [20, 50, 200])
    
    # M15 Trend Filter for HTF confirmation
    m15_ema_periods: list[int] = Field(default_factory=lambda: [20, 50, 200])
    
    # M1 Trigger (ChoCh→FVG→CE)
    m1_pivot_lookback: int = 3
    fvg_min_size_bps: float = 5.0
    fvg_max_age_bars: int = 20
    ce_tolerance_bps: float = 3.0
    cooldown_bars: int = 5
    
    # Fibonacci confluence (Fib 0.618 confirmation)
    require_fib_confluence: bool = False  # If True, require entry near 0.618 Fib level
    fib_target_level: float = 0.618       # Golden ratio
    fib_tolerance: float = 0.15           # ±15% around target (0.468 to 0.768)
    
    # Session Filter (UTC session names)
    sessions: list[str] = Field(default_factory=lambda: ["overlap"])
    
    # Spread Filter
    max_spread_bps: float = 25.0
    
    # Meta-Confirmation (deprecated; use meta_labeling.confidence_threshold instead)
    # This field is kept for backward compatibility but is ignored by Gateway
    meta_confidence_threshold: float = 0.55
    
    # Risk Management
    daily_trade_cap: int = 10
    daily_profit_target_usd: float = 100.0
    daily_loss_limit_usd: float = 50.0
    cooldown_after_loss_minutes: int = 15
    
    # Buffer Sizes
    m1_buffer_size: int = 500
    m5_buffer_size: int = 200
    m15_buffer_size: int = 100  # M15 buffer for HTF trend confirmation
    
    # Risk/Reward (Structure-based TP = Entry + RR * Risk)
    # For LONG: TP = entry + risk_reward_ratio * (entry - sl)
    # For SHORT: TP = entry - risk_reward_ratio * (sl - entry)
    risk_reward_ratio: float = 3.0  # Percoco standard 1:3 R:R
    
    # Training Parameters (legacy sigma-based, used if risk_reward_ratio not applied)
    tp_sigma_mult: float = 2.0
    sl_sigma_mult: float = 1.5
    event_horizon_ticks: int = 500


class SystemConfig(BaseModel):
    """System-wide settings."""
    log_level: str = "INFO"
    log_format: str = "json"
    timezone: str = "UTC"


class AlphaOSConfig(BaseModel):
    """
    Complete AlphaOS configuration.
    
    This is the top-level configuration object that contains
    all settings for the trading system.
    
    Additions:
    - meta_labeling: Meta-Labeling dual-layer architecture
    - sim2real: Sim2Real adversarial training
    - stagewise: Stagewise training pipeline
    - sessions: Trading session filters
    """
    symbol: str = "XAUUSD"
    symbol_info: SymbolInfo = Field(default_factory=SymbolInfo)
    system: SystemConfig = Field(default_factory=SystemConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    signal_database: DatabaseConfig | None = None  # Separate database for signals
    zeromq: ZeroMQConfig = Field(default_factory=ZeroMQConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # Advanced configurations
    sim2real: Sim2RealConfig = Field(default_factory=Sim2RealConfig)
    meta_labeling: MetaLabelingConfig = Field(default_factory=MetaLabelingConfig)
    stagewise: StagewiseConfig = Field(default_factory=StagewiseConfig)
    sessions: SessionsConfig = Field(default_factory=SessionsConfig)
    
    # Percoco Scalper Mode
    percoco_scalper: PercocoScalperConfig = Field(default_factory=PercocoScalperConfig)
    
    # v4.0: Primary Signal Engine Configuration
    primary: PrimaryConfig = Field(default_factory=PrimaryConfig)
    
    # v4.0: Data Sampling Configuration
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    
    # v4.0: Cross-Validation Configuration
    validation: ValidationCVConfig = Field(default_factory=ValidationCVConfig)


# ============================================================================
# Configuration Loading Functions
# ============================================================================

def load_config(
    config_name: str = "xauusd",
    config_dir: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AlphaOSConfig:
    """
    Load configuration from YAML files.
    
    Args:
        config_name: Name of the config file (without .yaml extension)
        config_dir: Directory containing config files. Defaults to 'configs/'
        overrides: Dictionary of values to override
        
    Returns:
        Validated AlphaOSConfig instance
    """
    if config_dir is None:
        # Find configs directory relative to project root
        config_dir = Path(__file__).parent.parent.parent.parent / "configs"
    else:
        config_dir = Path(config_dir)
    
    # Load base config
    base_path = config_dir / "base.yaml"
    if base_path.exists():
        base_cfg = OmegaConf.load(base_path)
    else:
        base_cfg = OmegaConf.create({})
    
    # Load symbol-specific config
    symbol_path = config_dir / f"{config_name}.yaml"
    if symbol_path.exists():
        symbol_cfg = OmegaConf.load(symbol_path)
        # Merge (symbol overrides base)
        cfg = OmegaConf.merge(base_cfg, symbol_cfg)
    else:
        cfg = base_cfg
    
    # Apply runtime overrides
    if overrides:
        override_cfg = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    
    # Resolve interpolations (environment variables, etc.)
    OmegaConf.resolve(cfg)
    
    # Convert to dict and validate with Pydantic
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Remove 'defaults' key if present (Hydra artifact)
    if isinstance(cfg_dict, dict) and "defaults" in cfg_dict:
        del cfg_dict["defaults"]
    
    return AlphaOSConfig.model_validate(cfg_dict)


def get_config_path() -> Path:
    """Get the path to the configs directory."""
    return Path(__file__).parent.parent.parent.parent / "configs"
