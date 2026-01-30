"""
v4.0 标签生成模块

实现 Triple Barrier Method 和 Meta-Labeling：
- 动态三栏栅（基于波动率自适应）
- Primary 信号事件过滤
- 元标签生成（训练 meta-model）
- 多 Horizon 标签（ML 友好）

参考：降噪LNN特征提取与信号过滤.md Section 5
"""

from alphaos.v4.labeling.triple_barrier import (
    TripleBarrierConfig,
    TripleBarrierLabeler,
    BarrierEvent,
    BarrierType,
)
from alphaos.v4.labeling.meta_labels import (
    MetaLabelGenerator,
    MetaLabelConfig,
)
from alphaos.v4.labeling.multi_horizon import (
    MultiHorizonLabeler,
    MultiHorizonConfig,
    MultiHorizonLabelsResult,
    compute_simple_labels,
)

__all__ = [
    "TripleBarrierConfig",
    "TripleBarrierLabeler",
    "BarrierEvent",
    "BarrierType",
    "MetaLabelGenerator",
    "MetaLabelConfig",
    # 多 Horizon 标签
    "MultiHorizonLabeler",
    "MultiHorizonConfig",
    "MultiHorizonLabelsResult",
    "compute_simple_labels",
]
