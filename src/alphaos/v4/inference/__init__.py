"""
v4.0 推理模块

实时推理引擎：
- Tick → Bar → Feature → Prediction
- 状态管理
- Schema 验证

参考：降噪LNN特征提取与信号过滤.md Section 7
"""

from alphaos.v4.inference.engine import (
    InferenceEngineV4,
    InferenceConfig,
    InferenceResult,
)

__all__ = [
    "InferenceEngineV4",
    "InferenceConfig",
    "InferenceResult",
]
