"""
v4.0 训练模块

提供完整的训练流程：
- CPCV (Combinatorial Purged Cross-Validation)
- Purging & Embargo（防止数据泄露）
- 训练入口脚本

参考：降噪LNN特征提取与信号过滤.md Section 6
"""

from alphaos.v4.training.cpcv import (
    CPCVSplitter,
    PurgedKFold,
    EmbargoConfig,
)
from alphaos.v4.training.pipeline import (
    TrainingConfig,
    V4TrainingPipeline,
)

__all__ = [
    "CPCVSplitter",
    "PurgedKFold",
    "EmbargoConfig",
    "TrainingConfig",
    "V4TrainingPipeline",
]
