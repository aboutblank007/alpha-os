"""
v4.0 统一特征 Schema

确保训练与推理使用完全一致的特征定义：
- 字段名称、顺序、数据类型
- 版本号追踪
- 可序列化存储与加载

参考：交易模型研究.md Section 3.2 - 索引错位/对齐陷阱
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray


class FeatureCategory(Enum):
    """特征类别"""
    PRICE = "price"                    # 价格相关（对数收益率等）
    MICROSTRUCTURE = "microstructure"  # 微观结构（OFI、Kyle Lambda 等）
    THERMODYNAMICS = "thermo"          # 热力学（温度、熵）
    VOLATILITY = "volatility"          # 波动率相关
    ORDERFLOW = "orderflow"            # 订单流（VPIN 等）
    TREND = "trend"                    # 趋势相关（trend_deviation 等）
    KALMAN = "kalman"                  # 卡尔曼滤波输出
    WAVELET = "wavelet"                # 小波分解输出
    FVG = "fvg"                        # FVG 特征 (活跃 FVG 状态)
    FVG_EVENT = "fvg_event"            # FVG 事件特征 (事件发生时的冲击)
    FVG_FOLLOW = "fvg_follow"          # FVG 因果跟随特征 (事件后的反应)
    SUPERTREND_15M = "supertrend_15m"  # 15m SuperTrend 特征
    TIME = "time"                      # 时间/Session 特征
    INTERACTION = "interaction"        # 交叉特征
    ATR = "atr"                        # ATR 相关特征


@dataclass
class FeatureSpec:
    """
    单个特征的规格定义
    
    Attributes:
        name: 特征名称（唯一标识）
        category: 特征类别
        dtype: 数据类型
        description: 特征描述
        scale_invariant: 是否尺度不变（百分比/对数）
        requires_history: 需要的历史 Bar 数量
        default_value: 缺失时的默认值
        min_value: 最小有效值（用于异常检测）
        max_value: 最大有效值（用于异常检测）
    """
    name: str
    category: FeatureCategory
    dtype: Literal["float32", "float64", "int32", "int64"] = "float32"
    description: str = ""
    scale_invariant: bool = True
    requires_history: int = 1
    default_value: float = 0.0
    min_value: float = -1e6
    max_value: float = 1e6
    
    def validate(self, value: float) -> bool:
        """验证值是否在有效范围内"""
        if np.isnan(value) or np.isinf(value):
            return False
        return self.min_value <= value <= self.max_value
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "name": self.name,
            "category": self.category.value,
            "dtype": self.dtype,
            "description": self.description,
            "scale_invariant": self.scale_invariant,
            "requires_history": self.requires_history,
            "default_value": self.default_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSpec":
        """从字典创建"""
        return cls(
            name=data["name"],
            category=FeatureCategory(data["category"]),
            dtype=data["dtype"],
            description=data.get("description", ""),
            scale_invariant=data.get("scale_invariant", True),
            requires_history=data.get("requires_history", 1),
            default_value=data.get("default_value", 0.0),
            min_value=data.get("min_value", -1e6),
            max_value=data.get("max_value", 1e6),
        )


# ============================================================================
# v4.0 标准特征定义
# 基于研究文档定义的完整微观结构特征集
# ============================================================================

V4_FEATURE_SPECS: list[FeatureSpec] = [
    # === 价格相关特征 ===
    FeatureSpec(
        name="log_return",
        category=FeatureCategory.PRICE,
        description="对数收益率 ln(P_t / P_{t-1})",
        scale_invariant=True,
        min_value=-0.1,
        max_value=0.1,
    ),
    FeatureSpec(
        name="log_return_zscore",
        category=FeatureCategory.PRICE,
        description="对数收益率的滚动 Z-Score",
        scale_invariant=True,
        min_value=-5.0,
        max_value=5.0,
    ),
    FeatureSpec(
        name="spread_bps",
        category=FeatureCategory.PRICE,
        description="点差（基点）= (ask - bid) / mid * 10000",
        scale_invariant=True,
        min_value=0.0,
        max_value=100.0,
    ),
    
    # === 微观结构特征 ===
    FeatureSpec(
        name="delta_t_log",
        category=FeatureCategory.MICROSTRUCTURE,
        description="时间间隔对数 ln(delta_t + epsilon)",
        scale_invariant=True,
        requires_history=1,
        min_value=-10.0,
        max_value=10.0,
    ),
    FeatureSpec(
        name="tick_intensity",
        category=FeatureCategory.MICROSTRUCTURE,
        description="Tick 强度 = EWMA(1/ln(delta_t + epsilon))",
        scale_invariant=True,
        min_value=0.0,
        max_value=100.0,
    ),
    FeatureSpec(
        name="ofi_count",
        category=FeatureCategory.MICROSTRUCTURE,
        description="基于计数的订单流失衡 = sum(epsilon_i)",
        scale_invariant=True,
        requires_history=20,
        min_value=-1000.0,
        max_value=1000.0,
    ),
    FeatureSpec(
        name="ofi_weighted",
        category=FeatureCategory.MICROSTRUCTURE,
        description="强度加权 OFI = sum(epsilon_i / delta_t_i)",
        scale_invariant=True,
        requires_history=20,
        min_value=-100.0,
        max_value=100.0,
    ),
    FeatureSpec(
        name="kyle_lambda_pct",
        category=FeatureCategory.MICROSTRUCTURE,
        description="Kyle Lambda（百分比）= |dp_pct| / delta_N",
        scale_invariant=True,
        requires_history=50,
        min_value=0.0,
        max_value=10.0,
    ),
    FeatureSpec(
        name="pdi",
        category=FeatureCategory.MICROSTRUCTURE,
        description="价格驱动失衡 = sign(dp_pct) * tick_intensity",
        scale_invariant=True,
        min_value=-100.0,
        max_value=100.0,
    ),
    
    # === 波动率特征 ===
    FeatureSpec(
        name="micro_volatility_pct",
        category=FeatureCategory.VOLATILITY,
        description="微观波动率（百分比）EWMA",
        scale_invariant=True,
        min_value=0.0,
        max_value=10.0,
    ),
    FeatureSpec(
        name="realized_volatility_pct",
        category=FeatureCategory.VOLATILITY,
        description="已实现波动率（百分比）= sqrt(sum(r^2))",
        scale_invariant=True,
        requires_history=20,
        min_value=0.0,
        max_value=10.0,
    ),
    
    # === 热力学特征 ===
    FeatureSpec(
        name="market_temperature",
        category=FeatureCategory.THERMODYNAMICS,
        description="市场温度 T = var(dp_pct) / mean(delta_t)",
        scale_invariant=True,
        requires_history=20,
        min_value=0.0,
        max_value=100.0,
    ),
    FeatureSpec(
        name="market_entropy",
        category=FeatureCategory.THERMODYNAMICS,
        description="市场熵 S = -sum(p_i * log(p_i))",
        scale_invariant=True,
        requires_history=50,
        min_value=0.0,
        max_value=2.0,
    ),
    FeatureSpec(
        name="ts_phase",
        category=FeatureCategory.THERMODYNAMICS,
        description="T-S 相位编码（0=frozen, 1=laminar, 2=turbulent, 3=transition）",
        dtype="int32",
        scale_invariant=True,
        min_value=0.0,
        max_value=3.0,
    ),
    
    # === 订单流特征 ===
    FeatureSpec(
        name="vpin",
        category=FeatureCategory.ORDERFLOW,
        description="VPIN = mean(|OI|) / V over n buckets",
        scale_invariant=True,
        requires_history=50,
        min_value=0.0,
        max_value=1.0,
    ),
    FeatureSpec(
        name="vpin_zscore",
        category=FeatureCategory.ORDERFLOW,
        description="VPIN 的滚动 Z-Score",
        scale_invariant=True,
        requires_history=100,
        min_value=-5.0,
        max_value=5.0,
    ),
    
    # === 趋势特征 ===
    FeatureSpec(
        name="trend_deviation",
        category=FeatureCategory.TREND,
        description="当前价格偏离 SuperTrend 线的百分比",
        scale_invariant=True,
        requires_history=20,
        min_value=-10.0,
        max_value=10.0,
    ),
    FeatureSpec(
        name="trend_direction",
        category=FeatureCategory.TREND,
        description="趋势方向编码（-1=short, 0=none, 1=long）",
        dtype="int32",
        scale_invariant=True,
        min_value=-1.0,
        max_value=1.0,
    ),
    FeatureSpec(
        name="trend_duration",
        category=FeatureCategory.TREND,
        description="当前趋势持续的 Bar 数量",
        dtype="int32",
        scale_invariant=True,
        min_value=0.0,
        max_value=1000.0,
    ),
    
    # === 卡尔曼滤波输出 ===
    FeatureSpec(
        name="kalman_residual_bps",
        category=FeatureCategory.KALMAN,
        description="卡尔曼残差（基点）",
        scale_invariant=True,
        min_value=-100.0,
        max_value=100.0,
    ),
    FeatureSpec(
        name="kalman_gain",
        category=FeatureCategory.KALMAN,
        description="卡尔曼增益（0-1）",
        scale_invariant=True,
        min_value=0.0,
        max_value=1.0,
    ),
    
    # === 小波分解输出 ===
    FeatureSpec(
        name="wavelet_energy_1",
        category=FeatureCategory.WAVELET,
        description="小波细节层 1 能量（最高频）",
        scale_invariant=True,
        min_value=0.0,
        max_value=1.0,
    ),
    FeatureSpec(
        name="wavelet_energy_2",
        category=FeatureCategory.WAVELET,
        description="小波细节层 2 能量",
        scale_invariant=True,
        min_value=0.0,
        max_value=1.0,
    ),
    FeatureSpec(
        name="wavelet_trend_ratio",
        category=FeatureCategory.WAVELET,
        description="趋势成分占比 = approx_energy / total_energy",
        scale_invariant=True,
        min_value=0.0,
        max_value=1.0,
    ),
    
    # === FVG 特征 (ML 友好) ===
    FeatureSpec(
        name="bullish_fvg",
        category=FeatureCategory.FVG,
        description="是否存在 Bullish FVG (0/1)",
        dtype="int32",
        scale_invariant=True,
        min_value=0.0,
        max_value=1.0,
    ),
    FeatureSpec(
        name="bearish_fvg",
        category=FeatureCategory.FVG,
        description="是否存在 Bearish FVG (0/1)",
        dtype="int32",
        scale_invariant=True,
        min_value=0.0,
        max_value=1.0,
    ),
    FeatureSpec(
        name="fvg_size_atr",
        category=FeatureCategory.FVG,
        description="FVG 大小 / ATR",
        scale_invariant=True,
        min_value=0.0,
        max_value=10.0,
    ),
    FeatureSpec(
        name="fvg_age_bars",
        category=FeatureCategory.FVG,
        description="FVG 形成后的 bar 数",
        dtype="int32",
        scale_invariant=True,
        min_value=0.0,
        max_value=100.0,
    ),
    FeatureSpec(
        name="price_to_fvg_mid",
        category=FeatureCategory.FVG,
        description="(Close - FVG_mid) / ATR",
        scale_invariant=True,
        min_value=-10.0,
        max_value=10.0,
    ),
    FeatureSpec(
        name="fvg_filled_pct",
        category=FeatureCategory.FVG,
        description="FVG 回补程度 (0-1)",
        scale_invariant=True,
        min_value=0.0,
        max_value=1.0,
    ),
    FeatureSpec(
        name="active_fvg_count",
        category=FeatureCategory.FVG,
        description="当前活跃 FVG 数量",
        dtype="int32",
        scale_invariant=True,
        min_value=0.0,
        max_value=20.0,
    ),
    FeatureSpec(
        name="nearest_fvg_distance",
        category=FeatureCategory.FVG,
        description="距离最近 FVG / ATR",
        scale_invariant=True,
        min_value=0.0,
        max_value=20.0,
    ),
    
    # === FVG 事件特征 (Event-based, 用于 LNN) ===
    FeatureSpec(
        name="fvg_event",
        category=FeatureCategory.FVG_EVENT,
        description="FVG 事件: +1=新 Bullish FVG, -1=新 Bearish FVG, 0=无事件",
        dtype="int32",
        scale_invariant=True,
        min_value=-1.0,
        max_value=1.0,
    ),
    FeatureSpec(
        name="fvg_impulse_atr",
        category=FeatureCategory.FVG_EVENT,
        description="FVG 冲击强度 = gap_size / ATR_1m (仅在事件时非零)",
        scale_invariant=True,
        min_value=0.0,
        max_value=20.0,
    ),
    FeatureSpec(
        name="fvg_location_15m",
        category=FeatureCategory.FVG_EVENT,
        description="FVG 位置 = (close_1m - mid_15m_range) / ATR_15m",
        scale_invariant=True,
        min_value=-20.0,
        max_value=20.0,
    ),
    
    # === FVG 因果跟随特征 (Causal follow-through, 用于 LNN) ===
    FeatureSpec(
        name="fvg_follow_up_3",
        category=FeatureCategory.FVG_FOLLOW,
        description="自最近 FVG 事件以来的最大上涨 / ATR (因果, 最多 3 bar)",
        scale_invariant=True,
        min_value=0.0,
        max_value=20.0,
    ),
    FeatureSpec(
        name="fvg_follow_dn_3",
        category=FeatureCategory.FVG_FOLLOW,
        description="自最近 FVG 事件以来的最大下跌 / ATR (因果, 最多 3 bar)",
        scale_invariant=True,
        min_value=-20.0,
        max_value=0.0,
    ),
    FeatureSpec(
        name="fvg_follow_bars",
        category=FeatureCategory.FVG_FOLLOW,
        description="距离最近 FVG 事件的 bar 数 (用于衰减)",
        dtype="int32",
        scale_invariant=True,
        min_value=0.0,
        max_value=100.0,
    ),
    FeatureSpec(
        name="fvg_follow_net",
        category=FeatureCategory.FVG_FOLLOW,
        description="FVG 事件后净收益 / ATR (close_now - close_at_event)",
        scale_invariant=True,
        min_value=-20.0,
        max_value=20.0,
    ),
    
    # === 15m SuperTrend 特征 ===
    FeatureSpec(
        name="st_trend_15m",
        category=FeatureCategory.SUPERTREND_15M,
        description="15m 趋势方向 (-1/0/1)",
        dtype="int32",
        scale_invariant=True,
        min_value=-1.0,
        max_value=1.0,
    ),
    FeatureSpec(
        name="st_distance_15m",
        category=FeatureCategory.SUPERTREND_15M,
        description="15m (Close - ST_line) / ATR",
        scale_invariant=True,
        min_value=-20.0,
        max_value=20.0,
    ),
    FeatureSpec(
        name="st_bars_since_flip_15m",
        category=FeatureCategory.SUPERTREND_15M,
        description="15m 趋势翻转后 bar 数",
        dtype="int32",
        scale_invariant=True,
        min_value=0.0,
        max_value=500.0,
    ),
    FeatureSpec(
        name="st_slope_15m",
        category=FeatureCategory.SUPERTREND_15M,
        description="15m ST_line 斜率 / ATR",
        scale_invariant=True,
        min_value=-5.0,
        max_value=5.0,
    ),
    FeatureSpec(
        name="st_bandwidth_15m",
        category=FeatureCategory.SUPERTREND_15M,
        description="15m ATR 带宽",
        scale_invariant=True,
        min_value=0.0,
        max_value=0.1,
    ),
    
    # === 时间/Session 特征 ===
    FeatureSpec(
        name="session",
        category=FeatureCategory.TIME,
        description="交易 Session (0=亚洲, 1=伦敦, 2=纽约, 3=重叠, 4=收盘)",
        dtype="int32",
        scale_invariant=True,
        min_value=0.0,
        max_value=4.0,
    ),
    FeatureSpec(
        name="hour_of_day",
        category=FeatureCategory.TIME,
        description="小时 (0-23)",
        dtype="int32",
        scale_invariant=True,
        min_value=0.0,
        max_value=23.0,
    ),
    FeatureSpec(
        name="minute_of_hour",
        category=FeatureCategory.TIME,
        description="分钟 (0-59)",
        dtype="int32",
        scale_invariant=True,
        min_value=0.0,
        max_value=59.0,
    ),
    FeatureSpec(
        name="day_of_week",
        category=FeatureCategory.TIME,
        description="星期几 (0=周一, 6=周日)",
        dtype="int32",
        scale_invariant=True,
        min_value=0.0,
        max_value=6.0,
    ),
    FeatureSpec(
        name="bars_from_session_open",
        category=FeatureCategory.TIME,
        description="Session 开盘后 bar 数",
        dtype="int32",
        scale_invariant=True,
        min_value=0.0,
        max_value=1000.0,
    ),
    FeatureSpec(
        name="is_session_open",
        category=FeatureCategory.TIME,
        description="是否在 Session 开盘期",
        dtype="int32",
        scale_invariant=True,
        min_value=0.0,
        max_value=1.0,
    ),
    FeatureSpec(
        name="is_session_close",
        category=FeatureCategory.TIME,
        description="是否在 Session 收盘期",
        dtype="int32",
        scale_invariant=True,
        min_value=0.0,
        max_value=1.0,
    ),
    FeatureSpec(
        name="hour_sin",
        category=FeatureCategory.TIME,
        description="小时正弦编码",
        scale_invariant=True,
        min_value=-1.0,
        max_value=1.0,
    ),
    FeatureSpec(
        name="hour_cos",
        category=FeatureCategory.TIME,
        description="小时余弦编码",
        scale_invariant=True,
        min_value=-1.0,
        max_value=1.0,
    ),
    
    # === 交叉特征 ===
    FeatureSpec(
        name="trend_fvg_alignment",
        category=FeatureCategory.INTERACTION,
        description="趋势 × FVG 对齐度 (st_trend * (bullish - bearish))",
        scale_invariant=True,
        min_value=-1.0,
        max_value=1.0,
    ),
    FeatureSpec(
        name="trend_strength_x_fvg",
        category=FeatureCategory.INTERACTION,
        description="|st_distance| × fvg_size_atr",
        scale_invariant=True,
        min_value=0.0,
        max_value=100.0,
    ),
    FeatureSpec(
        name="trend_duration_x_vol",
        category=FeatureCategory.INTERACTION,
        description="趋势持续 × 波动率",
        scale_invariant=True,
        min_value=0.0,
        max_value=1000.0,
    ),
    FeatureSpec(
        name="trend_fvg_distance",
        category=FeatureCategory.INTERACTION,
        description="st_distance × price_to_fvg_mid",
        scale_invariant=True,
        min_value=-100.0,
        max_value=100.0,
    ),
    FeatureSpec(
        name="session_trend_interaction",
        category=FeatureCategory.INTERACTION,
        description="session × st_trend",
        scale_invariant=True,
        min_value=-4.0,
        max_value=4.0,
    ),
    FeatureSpec(
        name="st_alignment",
        category=FeatureCategory.INTERACTION,
        description="ST 对齐 = st_trend_15m × fvg_event (顺/逆趋势指示)",
        scale_invariant=True,
        min_value=-1.0,
        max_value=1.0,
    ),
    
    # === ATR 特征 ===
    FeatureSpec(
        name="atr_ratio_1m_15m",
        category=FeatureCategory.ATR,
        description="ATR 比率 = ATR_1m / ATR_15m (波动率相对强度)",
        scale_invariant=True,
        min_value=0.0,
        max_value=5.0,
    ),
    FeatureSpec(
        name="atr_1m",
        category=FeatureCategory.ATR,
        description="1m ATR (用于归一化)",
        scale_invariant=True,
        min_value=0.0,
        max_value=100.0,
    ),
    FeatureSpec(
        name="atr_15m",
        category=FeatureCategory.ATR,
        description="15m ATR (用于归一化)",
        scale_invariant=True,
        min_value=0.0,
        max_value=500.0,
    ),
    FeatureSpec(
        name="mid_15m_range",
        category=FeatureCategory.ATR,
        description="15m bar 的 mid-range = (high + low) / 2",
        scale_invariant=False,
        min_value=0.0,
        max_value=100000.0,
    ),
]


@dataclass
class FeatureSchema:
    """
    v4.0 统一特征 Schema
    
    提供：
    - 特征名称与索引的双向映射
    - Schema 版本与哈希追踪
    - 特征验证与归一化
    - 序列化与反序列化
    
    使用方式：
    ```python
    # 创建默认 Schema
    schema = FeatureSchema.default()
    
    # 获取特征名称列表（有序）
    names = schema.feature_names
    
    # 按名称获取索引
    idx = schema.get_index("vpin")
    
    # 验证特征向量
    is_valid = schema.validate_features(features)
    
    # 保存/加载 Schema
    schema.save("model_dir/schema.json")
    schema = FeatureSchema.load("model_dir/schema.json")
    ```
    """
    
    specs: list[FeatureSpec] = field(default_factory=list)
    version: str = "4.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 缓存的索引映射
    _name_to_index: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _index_to_name: dict[int, str] = field(default_factory=dict, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """初始化索引映射"""
        self._build_index_maps()
    
    def _build_index_maps(self) -> None:
        """构建名称-索引双向映射"""
        self._name_to_index = {spec.name: i for i, spec in enumerate(self.specs)}
        self._index_to_name = {i: spec.name for i, spec in enumerate(self.specs)}
    
    @classmethod
    def default(cls) -> "FeatureSchema":
        """创建默认 v4.0 Schema"""
        return cls(specs=V4_FEATURE_SPECS.copy())
    
    @property
    def feature_names(self) -> list[str]:
        """获取有序的特征名称列表"""
        return [spec.name for spec in self.specs]
    
    @property
    def num_features(self) -> int:
        """特征数量"""
        return len(self.specs)
    
    @property
    def schema_hash(self) -> str:
        """
        计算 Schema 哈希（用于一致性检查）
        
        基于特征名称、顺序、类型计算 MD5
        """
        content = json.dumps(
            [{"name": s.name, "dtype": s.dtype, "idx": i} 
             for i, s in enumerate(self.specs)],
            sort_keys=True,
        )
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def get_index(self, name: str) -> int:
        """
        按名称获取特征索引
        
        Args:
            name: 特征名称
            
        Returns:
            特征在向量中的索引
            
        Raises:
            KeyError: 特征名称不存在
        """
        if name not in self._name_to_index:
            raise KeyError(f"Feature '{name}' not in schema. Available: {self.feature_names}")
        return self._name_to_index[name]
    
    def get_name(self, index: int) -> str:
        """
        按索引获取特征名称
        
        Args:
            index: 特征索引
            
        Returns:
            特征名称
        """
        if index not in self._index_to_name:
            raise IndexError(f"Index {index} out of range [0, {self.num_features})")
        return self._index_to_name[index]
    
    def get_spec(self, name: str) -> FeatureSpec:
        """
        获取特征规格
        
        Args:
            name: 特征名称
            
        Returns:
            FeatureSpec 对象
        """
        idx = self.get_index(name)
        return self.specs[idx]
    
    def get_dtype_array(self) -> list[tuple[str, str]]:
        """获取 numpy 结构化数组的 dtype 定义"""
        return [(spec.name, spec.dtype) for spec in self.specs]
    
    def validate_features(
        self, 
        features: NDArray[np.float32] | Sequence[float],
        strict: bool = False,
    ) -> tuple[bool, list[str]]:
        """
        验证特征向量
        
        Args:
            features: 特征向量
            strict: 是否严格模式（范围检查）
            
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # 检查长度
        if len(features) != self.num_features:
            errors.append(
                f"Feature length mismatch: expected {self.num_features}, got {len(features)}"
            )
            return False, errors
        
        # 检查每个特征
        for i, (spec, value) in enumerate(zip(self.specs, features)):
            if np.isnan(value):
                errors.append(f"Feature '{spec.name}' (idx={i}) is NaN")
            elif np.isinf(value):
                errors.append(f"Feature '{spec.name}' (idx={i}) is Inf")
            elif strict and not spec.validate(value):
                errors.append(
                    f"Feature '{spec.name}' (idx={i}) value {value:.4f} out of range "
                    f"[{spec.min_value}, {spec.max_value}]"
                )
        
        return len(errors) == 0, errors
    
    def clip_features(
        self, 
        features: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        裁剪特征值到有效范围
        
        Args:
            features: 特征向量
            
        Returns:
            裁剪后的特征向量
        """
        clipped = features.copy()
        for i, spec in enumerate(self.specs):
            clipped[i] = np.clip(clipped[i], spec.min_value, spec.max_value)
        return clipped
    
    def fill_defaults(self) -> NDArray[np.float32]:
        """
        创建默认特征向量（所有值为默认值）
        """
        return np.array([spec.default_value for spec in self.specs], dtype=np.float32)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "schema_hash": self.schema_hash,
            "num_features": self.num_features,
            "specs": [spec.to_dict() for spec in self.specs],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSchema":
        """从字典创建"""
        specs = [FeatureSpec.from_dict(s) for s in data["specs"]]
        schema = cls(
            specs=specs,
            version=data.get("version", "4.0.0"),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )
        return schema
    
    def save(self, path: str | Path) -> None:
        """
        保存 Schema 到 JSON 文件
        
        Args:
            path: 文件路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str | Path) -> "FeatureSchema":
        """
        从 JSON 文件加载 Schema
        
        Args:
            path: 文件路径
            
        Returns:
            FeatureSchema 对象
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def is_compatible(self, other: "FeatureSchema") -> bool:
        """
        检查两个 Schema 是否兼容
        
        兼容条件：
        - 相同的特征名称
        - 相同的顺序
        - 相同的数据类型
        
        Args:
            other: 另一个 Schema
            
        Returns:
            是否兼容
        """
        return self.schema_hash == other.schema_hash
    
    def __repr__(self) -> str:
        return (
            f"FeatureSchema(version={self.version}, "
            f"num_features={self.num_features}, "
            f"hash={self.schema_hash})"
        )


# ============================================================================
# Bar Schema 定义
# ============================================================================

@dataclass
class BarSchema:
    """
    Event Bar 的 Schema 定义
    
    统一 VolumeBar / TickImbalanceBar 的输出格式
    """
    
    FIELDS = [
        ("open_time_us", "int64"),      # Bar 开始时间（微秒）
        ("close_time_us", "int64"),     # Bar 结束时间（微秒）
        ("open", "float64"),            # 开盘价
        ("high", "float64"),            # 最高价
        ("low", "float64"),             # 最低价
        ("close", "float64"),           # 收盘价
        ("tick_count", "int32"),        # Tick 数量
        ("volume", "float64"),          # 累积成交量（可能为合成）
        ("imbalance", "int32"),         # 买卖失衡
        ("buy_count", "int32"),         # 买入 Tick 数
        ("sell_count", "int32"),        # 卖出 Tick 数
        ("spread_sum", "float64"),      # 点差累积
        ("duration_ms", "float64"),     # Bar 持续时间（毫秒）
    ]
    
    @classmethod
    def get_dtype(cls) -> list[tuple[str, str]]:
        """获取 numpy 结构化数组 dtype"""
        return cls.FIELDS.copy()
    
    @classmethod
    def field_names(cls) -> list[str]:
        """获取字段名称列表"""
        return [f[0] for f in cls.FIELDS]


# ============================================================================
# Label Schema 定义
# ============================================================================

@dataclass  
class LabelSchema:
    """
    Triple Barrier 标签的 Schema 定义
    """
    
    FIELDS = [
        ("bar_idx", "int64"),           # Bar 索引
        ("primary_direction", "int32"), # Primary 信号方向 (-1, 0, 1)
        ("barrier_hit", "int32"),       # 触碰的栏栅 (1=上, -1=下, 0=垂直)
        ("return_pct", "float32"),      # 实现的收益率（百分比）
        ("meta_label", "int32"),        # 元标签 (0=失败, 1=成功)
        ("holding_bars", "int32"),      # 持仓 Bar 数
        ("entry_price", "float64"),     # 入场价格
        ("exit_price", "float64"),      # 出场价格
        ("stop_loss", "float64"),       # 止损价格
        ("take_profit", "float64"),     # 止盈价格
    ]
    
    @classmethod
    def get_dtype(cls) -> list[tuple[str, str]]:
        """获取 numpy 结构化数组 dtype"""
        return cls.FIELDS.copy()
    
    @classmethod
    def field_names(cls) -> list[str]:
        """获取字段名称列表"""
        return [f[0] for f in cls.FIELDS]
