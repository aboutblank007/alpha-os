"""
v4.0 降噪预处理模块

提供可配置的降噪管道：
- MODWT 小波变换（离线训练数据）
- 卡尔曼滤波（在线/实时）
- 组合降噪策略

参考：降噪LNN特征提取与信号过滤.md Section 3
"""

from alphaos.v4.denoise.pipeline import (
    DenoiseConfig,
    DenoiseMode,
    DenoisePipeline,
    DenoiseResult,
)

__all__ = [
    "DenoiseConfig",
    "DenoiseMode",
    "DenoisePipeline",
    "DenoiseResult",
]
