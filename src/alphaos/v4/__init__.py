"""
AlphaOS v4.0 - 完整重构版本

基于研究文档的全新架构：
- 信息驱动采样（Volume Bars / Imbalance Bars）
- 降噪预处理（MODWT + Kalman）
- 微观结构特征（含 VPIN、trend_deviation）
- Primary 信号引擎（PivotSuperTrend + FVG）
- Triple Barrier 元标签
- CfC/LNN 编码器 + XGBoost Meta Head
- 统一的 FeatureSchema 确保训练/推理一致性

核心设计原则：
1. 尺度不变性 - 所有特征使用百分比/对数收益率
2. 事件时间 - 以 Bar 计数而非物理时间
3. Sim2Real 鲁棒性 - Volume=0 回退机制
4. Schema 一致性 - 训练与推理特征完全对齐
"""

__version__ = "4.0.0"

# Schemas
from alphaos.v4.schemas import (
    FeatureSchema,
    FeatureSpec,
    FeatureCategory,
    BarSchema,
    LabelSchema,
)

# Sampling
from alphaos.v4.sampling import (
    SamplingConfig,
    SamplingMode,
    UnifiedSampler,
    EventBarStream,
)

# Denoising
from alphaos.v4.denoise import (
    DenoiseConfig,
    DenoiseMode,
    DenoisePipeline,
    DenoiseResult,
)

# Features
from alphaos.v4.features import (
    FeatureConfig,
    FeaturePipelineV4,
    FeatureResult,
    VPINCalculator,
    VPINConfig,
)

# Labeling
from alphaos.v4.labeling import (
    TripleBarrierConfig,
    TripleBarrierLabeler,
    BarrierEvent,
    BarrierType,
    MetaLabelGenerator,
    MetaLabelConfig,
)

# Primary Engine
from alphaos.v4.primary import (
    PrimaryEngineConfig,
    PrimaryEngineV4,
    PrimarySignalV4,
)

# Training
from alphaos.v4.training import (
    TrainingConfig,
    V4TrainingPipeline,
    CPCVSplitter,
    PurgedKFold,
    EmbargoConfig,
)

# Inference
from alphaos.v4.inference import (
    InferenceConfig,
    InferenceEngineV4,
    InferenceResult,
)

# Models (CfC + XGBoost)
from alphaos.v4.models import (
    CfCConfig,
    CfCCell,
    CfCEncoder,
    ModelBundle,
)

__all__ = [
    "__version__",
    # Schemas
    "FeatureSchema",
    "FeatureSpec",
    "FeatureCategory",
    "BarSchema",
    "LabelSchema",
    # Sampling
    "SamplingConfig",
    "SamplingMode",
    "UnifiedSampler",
    "EventBarStream",
    # Denoising
    "DenoiseConfig",
    "DenoiseMode",
    "DenoisePipeline",
    "DenoiseResult",
    # Features
    "FeatureConfig",
    "FeaturePipelineV4",
    "FeatureResult",
    "VPINCalculator",
    "VPINConfig",
    # Labeling
    "TripleBarrierConfig",
    "TripleBarrierLabeler",
    "BarrierEvent",
    "BarrierType",
    "MetaLabelGenerator",
    "MetaLabelConfig",
    # Primary
    "PrimaryEngineConfig",
    "PrimaryEngineV4",
    "PrimarySignalV4",
    # Training
    "TrainingConfig",
    "V4TrainingPipeline",
    "CPCVSplitter",
    "PurgedKFold",
    "EmbargoConfig",
    # Inference
    "InferenceConfig",
    "InferenceEngineV4",
    "InferenceResult",
    # Models
    "CfCConfig",
    "CfCCell",
    "CfCEncoder",
    "ModelBundle",
]
