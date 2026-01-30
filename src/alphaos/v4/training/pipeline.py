"""
v4.0 训练管道

完整的端到端训练流程：
1. 数据加载与采样（Volume Bars）
2. 降噪预处理
3. 特征计算
4. Primary 信号生成
5. Meta-Labeling
6. 模型训练（CfC + XGBoost）
7. 模型评估与保存

参考：降噪LNN特征提取与信号过滤.md Section 6

v4.0 更新：
- 实现完整的 CfC 编码器训练
- XGBoost 输入：concat([current_features, cfc_hidden_state])
- 保存完整模型包：cfc_encoder.pt, xgb_model.json, cfc_config.json
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import json

import yaml

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from alphaos.core.logging import get_logger
from alphaos.core.types import Tick
from alphaos.v4.schemas import FeatureSchema
from alphaos.v4.sampling import SamplingConfig, SamplingMode, VolumeSource, UnifiedSampler
from alphaos.v4.denoise import DenoiseConfig, DenoisePipeline
from alphaos.v4.features import FeatureConfig, FeaturePipelineV4, ThermodynamicsConfig
from alphaos.v4.primary import PrimaryEngineConfig, PrimaryEngineV4
from alphaos.v4.labeling import (
    TripleBarrierConfig,
    MetaLabelConfig,
    MetaLabelGenerator,
    MultiHorizonConfig,
)
from alphaos.v4.training.cpcv import PurgedKFold, CPCVSplitter, EmbargoConfig
from alphaos.v4.models import CfCConfig, CfCEncoder, ModelBundle

logger = get_logger(__name__)


def compute_feature_list_hash(feature_names: list[str]) -> str:
    """
    计算特征名称列表的稳定 hash（用于校验 schema/mask 一致性）
    
    Args:
        feature_names: 特征名称列表（顺序敏感）
        
    Returns:
        8 字符的 sha256 截断哈希
    """
    # 使用 json.dumps 确保序列化一致性（不排序，保持顺序）
    content = json.dumps(feature_names, sort_keys=False, ensure_ascii=True)
    return hashlib.sha256(content.encode()).hexdigest()[:8]


def compute_schema_mask_combo_hash(
    schema_hash: str,
    lnn_mask_hash: str,
    xgb_mask_hash: str,
) -> str:
    """
    计算 schema + masks 的组合哈希（快速一致性检查）
    
    Args:
        schema_hash: FeatureSchema 的 hash
        lnn_mask_hash: LNN 特征列表的 hash
        xgb_mask_hash: XGB 特征列表的 hash
        
    Returns:
        8 字符的 sha256 截断哈希
    """
    content = f"{schema_hash}:{lnn_mask_hash}:{xgb_mask_hash}"
    return hashlib.sha256(content.encode()).hexdigest()[:8]


def compute_quantile_evaluation_metrics(
    y_true: NDArray,
    y_prob: NDArray,
    ts_phase: NDArray | None = None,
    sample_weights: NDArray | None = None,
    quantiles: list[int] = [99, 95, 90, 85],
    fixed_threshold: float = 0.5,
) -> dict:
    """
    计算基于分位数的评估指标（替代固定 0.5 阈值）
    
    核心理念：
    - Accuracy 在不平衡数据集中不可信（只需预测多数类即可达到高准确率）
    - 固定阈值 0.5 对 meta-model conditioner 不合适
    - 使用 Recall@TopQuantile 和 PR-AUC 更能反映实际表现
    
    Args:
        y_true: 真实标签 [n_samples]
        y_prob: 预测概率 [n_samples]
        ts_phase: 可选的市场相位标签（0=frozen, 1=laminar, 2=turbulent, 3=transition）
        sample_weights: 样本权重（用于“成本加权”的 PR-AUC / Recall@TopQuantile；None 表示等权）
        quantiles: 要评估的分位数（如 [99, 95, 90, 85] 对应 top 1%, 5%, 10%, 15%）
        
    Returns:
        包含所有评估指标的字典
    """
    from sklearn.metrics import (
        precision_recall_curve,
        average_precision_score,
        roc_auc_score,
    )
    
    metrics = {}
    
    # === 权重（成本）=== 
    # 说明：这里的 sample_weights 代表“样本重要性/成本权重”，不是交易笔数。
    # 用于计算 weighted precision/recall，使 Recall@TopQuantile 从“交易数口径”切换到“成本口径”。
    if sample_weights is None:
        w = np.ones(len(y_true), dtype=np.float64)
    else:
        if len(sample_weights) != len(y_true):
            raise ValueError(
                f"sample_weights 长度不匹配：len(sample_weights)={len(sample_weights)} vs len(y_true)={len(y_true)}"
            )
        w = np.asarray(sample_weights, dtype=np.float64)
        # 防止出现负权重（负权重会导致指标无意义）
        if np.any(w < 0):
            raise ValueError("sample_weights 存在负值，无法用于成本加权评估。")
    
    # === 1. 概率分布摘要 ===
    prob_percentiles = [50, 75, 90, 95, 99]
    metrics["prob_distribution"] = {
        f"P{p}": float(np.percentile(y_prob, p)) for p in prob_percentiles
    }
    metrics["prob_distribution"]["mean"] = float(np.mean(y_prob))
    metrics["prob_distribution"]["std"] = float(np.std(y_prob))
    metrics["prob_distribution"]["min"] = float(np.min(y_prob))
    metrics["prob_distribution"]["max"] = float(np.max(y_prob))
    
    # === 2. PR-AUC（比 ROC-AUC 更适合不平衡数据）===
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        # 成本加权版本（等权时与原值一致）
        metrics["weighted_roc_auc"] = float(roc_auc_score(y_true, y_prob, sample_weight=w))
        metrics["weighted_pr_auc"] = float(average_precision_score(y_true, y_prob, sample_weight=w))
    else:
        metrics["roc_auc"] = 0.5
        metrics["pr_auc"] = float(np.mean(y_true))  # baseline
        metrics["weighted_roc_auc"] = 0.5
        # 加权 baseline：正类权重占比
        w_pos = float(np.sum(w * (y_true == 1)))
        w_sum = float(np.sum(w)) if len(w) else 0.0
        metrics["weighted_pr_auc"] = (w_pos / w_sum) if w_sum > 0 else float(np.mean(y_true))
    
    # === 3. Recall@TopQuantile ===
    # 对于每个分位数，计算 "如果只交易 top X% 概率的样本，Recall 是多少"
    metrics["recall_at_quantile"] = {}
    metrics["precision_at_quantile"] = {}
    metrics["n_selected_at_quantile"] = {}
    # 成本加权版本（非交易数口径）
    metrics["weighted_recall_at_quantile"] = {}
    metrics["weighted_precision_at_quantile"] = {}
    metrics["weight_selected_at_quantile"] = {}
    
    n_positive = int(np.sum(y_true))
    n_total = len(y_true)
    
    for q in quantiles:
        threshold = np.percentile(y_prob, q)
        y_pred_q = (y_prob >= threshold).astype(int)
        
        # 计算在该阈值下的 Precision/Recall
        tp = np.sum((y_pred_q == 1) & (y_true == 1))
        fp = np.sum((y_pred_q == 1) & (y_true == 0))
        fn = np.sum((y_pred_q == 0) & (y_true == 1))
        
        precision_q = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_q = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        n_selected = int(np.sum(y_pred_q))
        
        metrics["recall_at_quantile"][f"P{q}"] = float(recall_q)
        metrics["precision_at_quantile"][f"P{q}"] = float(precision_q)
        metrics["n_selected_at_quantile"][f"P{q}"] = n_selected
        
        # 成本加权 Precision/Recall（权重口径）
        sel = (y_pred_q == 1)
        y_pos = (y_true == 1)
        
        w_tp = float(np.sum(w * (sel & y_pos)))
        w_fp = float(np.sum(w * (sel & (~y_pos))))
        w_fn = float(np.sum(w * ((~sel) & y_pos)))
        
        w_precision_q = (w_tp / (w_tp + w_fp)) if (w_tp + w_fp) > 0 else 0.0
        w_recall_q = (w_tp / (w_tp + w_fn)) if (w_tp + w_fn) > 0 else 0.0
        w_selected = float(np.sum(w * sel))
        
        metrics["weighted_recall_at_quantile"][f"P{q}"] = float(w_recall_q)
        metrics["weighted_precision_at_quantile"][f"P{q}"] = float(w_precision_q)
        metrics["weight_selected_at_quantile"][f"P{q}"] = float(w_selected)
    
    # === 4. 固定阈值指标（用于对比）===
    y_pred_fixed = (y_prob > fixed_threshold).astype(int)
    tp_fixed = np.sum((y_pred_fixed == 1) & (y_true == 1))
    fp_fixed = np.sum((y_pred_fixed == 1) & (y_true == 0))
    fn_fixed = np.sum((y_pred_fixed == 0) & (y_true == 1))
    
    metrics["fixed_threshold"] = {
        "threshold": float(fixed_threshold),
        "precision": float(tp_fixed / (tp_fixed + fp_fixed)) if (tp_fixed + fp_fixed) > 0 else 0.0,
        "recall": float(tp_fixed / (tp_fixed + fn_fixed)) if (tp_fixed + fn_fixed) > 0 else 0.0,
        "n_selected": int(np.sum(y_pred_fixed)),
        # 成本加权版本（非交易数口径）
        "weighted_precision": float(np.sum(w * ((y_pred_fixed == 1) & (y_true == 1))))
        / float(np.sum(w * (y_pred_fixed == 1)))
        if float(np.sum(w * (y_pred_fixed == 1))) > 0
        else 0.0,
        "weighted_recall": float(np.sum(w * ((y_pred_fixed == 1) & (y_true == 1))))
        / float(np.sum(w * (y_true == 1)))
        if float(np.sum(w * (y_true == 1))) > 0
        else 0.0,
        "weight_selected": float(np.sum(w * (y_pred_fixed == 1))),
    }
    
    # === 5. 分相位指标（如果提供了 ts_phase）===
    # 始终打印所有相位（包括 skipped），以确保推理端 phase gating 不是"半盲"
    if ts_phase is not None and len(ts_phase) == len(y_true):
        phase_names = ["FROZEN", "LAMINAR", "TURBULENT", "TRANSITION"]
        metrics["by_phase"] = {}
        
        for phase_id, phase_name in enumerate(phase_names):
            mask = (ts_phase == phase_id)
            n_phase = int(np.sum(mask))
            
            if n_phase < 10:  # 样本太少则跳过（但仍记录）
                metrics["by_phase"][phase_name] = {
                    "n_samples": n_phase,
                    "skip_reason": "insufficient_samples (<10)"
                }
                continue
            
            y_true_phase = y_true[mask]
            y_prob_phase = y_prob[mask]
            w_phase = w[mask]
            n_pos_phase = int(np.sum(y_true_phase))
            
            phase_metrics = {
                "n_samples": n_phase,
                "n_positive": n_pos_phase,
                "positive_rate": float(n_pos_phase / n_phase) if n_phase > 0 else 0.0,
            }
            
            # PR-AUC（排序质量核心指标）
            if len(np.unique(y_true_phase)) > 1:
                phase_metrics["pr_auc"] = float(average_precision_score(y_true_phase, y_prob_phase))
                # 加权 PR-AUC
                phase_metrics["weighted_pr_auc"] = float(
                    average_precision_score(y_true_phase, y_prob_phase, sample_weight=w_phase)
                )
            else:
                phase_metrics["pr_auc"] = float(np.mean(y_true_phase))
                phase_metrics["weighted_pr_auc"] = float(np.mean(y_true_phase))
            
            # 使用传入的 quantiles 列表（而非硬编码 [95, 90]）
            for q in quantiles:
                if q == 99 and n_phase < 100:
                    # P99 需要至少 100 样本才有意义
                    phase_metrics[f"recall_P{q}"] = None
                    phase_metrics[f"weighted_recall_P{q}"] = None
                    continue
                    
                threshold = np.percentile(y_prob_phase, q)
                y_pred_phase = (y_prob_phase >= threshold).astype(int)
                tp = np.sum((y_pred_phase == 1) & (y_true_phase == 1))
                fn = np.sum((y_pred_phase == 0) & (y_true_phase == 1))
                recall_q = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                phase_metrics[f"recall_P{q}"] = float(recall_q)
                
                # 成本加权 Recall
                sel = (y_pred_phase == 1)
                y_pos = (y_true_phase == 1)
                w_tp = float(np.sum(w_phase * (sel & y_pos)))
                w_pos = float(np.sum(w_phase * y_pos))
                phase_metrics[f"weighted_recall_P{q}"] = (w_tp / w_pos) if w_pos > 0 else 0.0
            
            metrics["by_phase"][phase_name] = phase_metrics
    
    # === 6. Funnel 摘要 ===
    metrics["funnel"] = {
        "n_total_samples": n_total,
        "n_positive_labels": n_positive,
        "positive_rate": float(n_positive / n_total) if n_total > 0 else 0.0,
        # 成本口径（便于把“交易数”替换成“成本”）
        "total_weight": float(np.sum(w)) if len(w) else 0.0,
        "positive_weight": float(np.sum(w * (y_true == 1))) if len(w) else 0.0,
    }
    
    return metrics


def format_quantile_metrics_log(metrics: dict) -> str:
    """格式化分位数评估指标为易读的日志字符串"""
    lines = []
    
    # 概率分布
    prob = metrics.get("prob_distribution", {})
    lines.append(f"  Prob distribution: P50={prob.get('P50', 0):.3f}, P90={prob.get('P90', 0):.3f}, P95={prob.get('P95', 0):.3f}, P99={prob.get('P99', 0):.3f}")
    
    # AUC
    lines.append(
        f"  ROC-AUC={metrics.get('roc_auc', 0):.4f}, PR-AUC={metrics.get('pr_auc', 0):.4f} "
        f"(weighted: ROC-AUC={metrics.get('weighted_roc_auc', 0):.4f}, PR-AUC={metrics.get('weighted_pr_auc', 0):.4f})"
    )
    
    # Recall@Quantile
    recall_q = metrics.get("recall_at_quantile", {})
    prec_q = metrics.get("precision_at_quantile", {})
    n_sel = metrics.get("n_selected_at_quantile", {})
    w_recall_q = metrics.get("weighted_recall_at_quantile", {})
    w_prec_q = metrics.get("weighted_precision_at_quantile", {})
    w_sel = metrics.get("weight_selected_at_quantile", {})
    q_strs = []
    for q in ["P99", "P95", "P90", "P85"]:
        if q in recall_q:
            q_strs.append(
                f"{q}: R={recall_q[q]:.2%}/P={prec_q.get(q, 0):.2%}/N={n_sel.get(q, 0)}"
                f" | wR={w_recall_q.get(q, 0):.2%}/wP={w_prec_q.get(q, 0):.2%}/wW={w_sel.get(q, 0):.1f}"
            )
    if q_strs:
        lines.append(f"  Recall@Quantile: {', '.join(q_strs)}")
    
    # 固定阈值对比
    fixed = metrics.get("fixed_threshold", {})
    fixed_value = fixed.get("threshold", 0.5)
    lines.append(
        f"  Fixed@{fixed_value:.2f}: R={fixed.get('recall', 0):.2%}, "
        f"P={fixed.get('precision', 0):.2%}, N={fixed.get('n_selected', 0)}"
        f" | wR={fixed.get('weighted_recall', 0):.2%}, "
        f"wP={fixed.get('weighted_precision', 0):.2%}, wW={fixed.get('weight_selected', 0):.1f}"
    )
    
    # 分相位（始终打印所有相位，包括 skipped，确保推理端 phase gating 不是半盲）
    by_phase = metrics.get("by_phase", {})
    if by_phase:
        lines.append("  By phase (for inference gate alignment):")
        phase_order = ["FROZEN", "LAMINAR", "TURBULENT", "TRANSITION"]
        for phase in phase_order:
            pm = by_phase.get(phase, {})
            n = pm.get("n_samples", 0)
            
            if "skip_reason" in pm:
                # 样本不足，仍打印原因
                lines.append(f"    {phase}: n={n} ({pm['skip_reason']})")
            else:
                # 正常打印 PR-AUC + Recall@Quantiles
                pr_auc = pm.get("pr_auc", 0)
                r95 = pm.get("recall_P95")
                r90 = pm.get("recall_P90")
                r85 = pm.get("recall_P85")
                
                recall_parts = []
                if r95 is not None:
                    recall_parts.append(f"R@P95={r95:.1%}")
                if r90 is not None:
                    recall_parts.append(f"R@P90={r90:.1%}")
                if r85 is not None:
                    recall_parts.append(f"R@P85={r85:.1%}")
                
                recall_str = ", ".join(recall_parts) if recall_parts else "N/A"
                lines.append(f"    {phase}: n={n}, PR-AUC={pr_auc:.4f}, {recall_str}")
    
    return "\n".join(lines)


def extract_fvg_event_t0_anchors(
    fvg_event: NDArray,
    audit_rising_edge: bool = True,
) -> tuple[NDArray, dict]:
    """
    提取 fvg_event 的 t0 事件起始 bar 索引（严格 Anchor）
    
    FVGEventCalculator 设计上是脉冲事件：fvg_event 仅在事件那根 bar 非零。
    但为了防御"状态化"情况，可选执行 rising-edge 审计。
    
    Args:
        fvg_event: fvg_event 特征序列 (n_bars,)，值为 +1/-1/0
        audit_rising_edge: 是否审计 rising-edge（检测是否有连续非零）
        
    Returns:
        anchor_indices: t0 事件索引 (n_events,)
        audit_info: 审计信息（anchor_count, rising_edge_count, ratio_diff）
    """
    # 默认：直接取非零位置
    anchor_indices = np.where(fvg_event != 0)[0]
    
    audit_info = {
        "anchor_count": len(anchor_indices),
        "rising_edge_count": len(anchor_indices),
        "ratio_diff": 0.0,
        "is_pulse": True,
    }
    
    if audit_rising_edge and len(anchor_indices) > 0:
        # Rising-edge：(fvg_event != 0) & (prev_fvg_event == 0)
        prev_fvg = np.roll(fvg_event, 1)
        prev_fvg[0] = 0  # 第一个 bar 没有前一个
        rising_edge_mask = (fvg_event != 0) & (prev_fvg == 0)
        rising_edge_indices = np.where(rising_edge_mask)[0]
        
        audit_info["rising_edge_count"] = len(rising_edge_indices)
        
        if len(anchor_indices) > 0:
            ratio_diff = abs(len(rising_edge_indices) - len(anchor_indices)) / len(anchor_indices)
            audit_info["ratio_diff"] = ratio_diff
            audit_info["is_pulse"] = (ratio_diff < 0.01)  # 允许 1% 误差
            
            # 如果差异明显，使用 rising-edge（更严格）
            if ratio_diff > 0.05:
                anchor_indices = rising_edge_indices
    
    return anchor_indices, audit_info


def compute_temporal_recall_batch(
    anchor_indices: NDArray,
    y_true: NDArray,
    features_full: NDArray,
    delta_ts: NDArray,
    encoder,
    xgb_model,
    sequence_length: int,
    thresholds_by_quantile: dict[int, float],
    lookahead_bars: int = 3,
    lnn_indices: list[int] | None = None,
    xgb_indices: list[int] | None = None,
    device: str = "cpu",
) -> dict:
    """
    批量计算 Temporal Recall 诊断（event + N bars）
    
    严格因果：对每个事件 t0，评估 {t0, t0+1, ..., t0+N} 的模型概率，
    只要窗口内任一 bar 超过阈值即算 temporal hit。
    
    ⚠️ 这是离线诊断工具，不改训练目标、不改实盘逻辑。
    
    Args:
        anchor_indices: t0 事件索引（与 y_true 对齐）
        y_true: 真实 meta-label（与 anchor_indices 对齐）
        features_full: 全部 bar 的特征矩阵 (n_bars, n_features)
        delta_ts: 全部 bar 的时间间隔 (n_bars,)
        encoder: CfC 编码器（已训练）
        xgb_model: XGBoost 模型（已训练）
        sequence_length: 序列长度
        thresholds_by_quantile: {quantile: threshold} 来自同 fold 的 y_prob
        lookahead_bars: 向后看的 bar 数（默认 3）
        lnn_indices: LNN 特征索引（可选）
        xgb_indices: XGB 特征索引（可选）
        device: 计算设备
        
    Returns:
        temporal_metrics: 包含各 N、各 quantile 的 temporal recall
    """
    import torch
    
    n_events = len(anchor_indices)
    n_bars = len(features_full)
    N = lookahead_bars
    
    if n_events == 0:
        return {"error": "no_events", "n_events": 0}
    
    n_positive = int(np.sum(y_true))
    if n_positive == 0:
        return {"error": "no_positive_labels", "n_events": n_events}
    
    # 特征分离
    if lnn_indices is not None and len(lnn_indices) > 0:
        features_lnn = features_full[:, lnn_indices]
    else:
        features_lnn = features_full
    
    if xgb_indices is not None and len(xgb_indices) > 0:
        features_xgb = features_full[:, xgb_indices]
    else:
        features_xgb = features_full
    
    # === 1. 构造候选索引矩阵 ===
    # candidates[i, j] = anchor_indices[i] + j, 对应 event i 的 t0+j bar
    offsets = np.arange(0, N + 1)  # [0, 1, 2, ..., N]
    candidates = anchor_indices[:, None] + offsets[None, :]  # (n_events, N+1)
    
    # 标记越界
    valid_mask = (candidates >= 0) & (candidates < n_bars)  # (n_events, N+1)
    
    # === 2. Flatten 有效候选索引 ===
    flat_candidates = candidates.flatten()  # (n_events * (N+1),)
    flat_valid = valid_mask.flatten()
    
    # 只处理有效的候选
    valid_flat_indices = flat_candidates[flat_valid]
    n_valid = len(valid_flat_indices)
    
    if n_valid == 0:
        return {"error": "no_valid_candidates", "n_events": n_events}
    
    # === 3. 批量构建序列（复用 _build_sequences 逻辑）===
    seq_len = sequence_length
    lnn_dim = features_lnn.shape[1]
    
    # 为每个有效候选 bar 构建序列
    seq_X = np.zeros((n_valid, seq_len, lnn_dim), dtype=np.float32)
    seq_dt = np.ones((n_valid, seq_len), dtype=np.float32)
    
    for i, bar_idx in enumerate(valid_flat_indices):
        start_idx = max(0, bar_idx - seq_len + 1)
        end_idx = bar_idx + 1
        actual_len = end_idx - start_idx
        
        seq_X[i, -actual_len:, :] = features_lnn[start_idx:end_idx]
        seq_dt[i, -actual_len:] = delta_ts[start_idx:end_idx]
    
    # === 4. 批量提取 CfC hidden states ===
    if encoder is not None:
        try:
            encoder.eval()
            torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
            
            X_tensor = torch.tensor(seq_X, dtype=torch.float32, device=torch_device)
            dt_tensor = torch.tensor(seq_dt, dtype=torch.float32, device=torch_device)
            
            with torch.no_grad():
                hidden_output = encoder(X_tensor, dt_tensor)
                if isinstance(hidden_output, tuple):
                    hidden_output = hidden_output[0]
                hidden_np = hidden_output.cpu().numpy()
        except Exception as e:
            # 如果 CfC 失败，使用零向量
            hidden_np = np.zeros((n_valid, 64), dtype=np.float32)
    else:
        hidden_np = np.zeros((n_valid, 64), dtype=np.float32)
    
    # === 5. 组合 XGB 输入并批量预测 ===
    xgb_base = features_xgb[valid_flat_indices]  # (n_valid, xgb_dim)
    X_combined = np.concatenate([xgb_base, hidden_np], axis=1)
    
    try:
        flat_probs = xgb_model.predict_proba(X_combined)[:, 1]
    except Exception as e:
        return {"error": f"xgb_predict_failed: {e}", "n_events": n_events}
    
    # === 6. Reshape 回 (n_events, N+1) ===
    candidate_probs = np.full((n_events, N + 1), np.nan, dtype=np.float64)
    
    # 填充有效位置
    flat_idx = 0
    for i in range(n_events):
        for j in range(N + 1):
            if valid_mask[i, j]:
                candidate_probs[i, j] = flat_probs[flat_idx]
                flat_idx += 1
    
    # === 7. 计算 Temporal Recall ===
    results = {
        "n_events": n_events,
        "n_positive": n_positive,
        "lookahead_bars": N,
        "quantiles": list(thresholds_by_quantile.keys()),
    }
    
    positive_mask = (y_true == 1)
    
    for q, threshold in thresholds_by_quantile.items():
        results[f"threshold_P{q}"] = float(threshold)
        
        for w in range(N + 1):  # window size 0..N
            # 窗口内最大概率
            window_probs = candidate_probs[:, :w + 1]
            window_max = np.nanmax(window_probs, axis=1)
            
            # Temporal hit: 窗口内任一 bar 超过阈值
            hit = (window_max >= threshold)
            
            # Temporal Recall: 正样本中的命中率
            temporal_hits = hit & positive_mask
            temporal_recall = np.sum(temporal_hits) / n_positive if n_positive > 0 else 0.0
            
            # Temporal Precision: 命中中的正样本率
            n_selected = np.sum(hit)
            temporal_precision = np.sum(temporal_hits) / n_selected if n_selected > 0 else 0.0
            
            results[f"temporal_recall_P{q}_N{w}"] = float(temporal_recall)
            results[f"temporal_precision_P{q}_N{w}"] = float(temporal_precision)
            results[f"n_selected_P{q}_N{w}"] = int(n_selected)
    
    return results


def format_temporal_recall_log(temporal_metrics: dict, quantiles: list[int] = [95, 90, 85]) -> str:
    """
    格式化 Temporal Recall 诊断为标准日志输出
    
    输出格式：
    Temporal Recall Diagnosis (event + N bars)
      N=0:  R@P95=16.9%, R@P90=31.0%, R@P85=42.8%
      N=1:  R@P95=23.4%, R@P90=39.7%, R@P85=51.2%
      ...
    """
    if "error" in temporal_metrics:
        return f"  Temporal Recall skipped: {temporal_metrics['error']}"
    
    N = temporal_metrics.get("lookahead_bars", 3)
    lines = ["Temporal Recall Diagnosis (event + N bars)"]
    
    for w in range(N + 1):
        q_strs = []
        for q in quantiles:
            key = f"temporal_recall_P{q}_N{w}"
            if key in temporal_metrics:
                r = temporal_metrics[key]
                q_strs.append(f"R@P{q}={r:.1%}")
        
        if q_strs:
            lines.append(f"  N={w}:  {', '.join(q_strs)}")
    
    return "\n".join(lines)


@dataclass
class FeatureMaskConfig:
    """
    LNN/XGB 特征分离配置
    
    支持 LNN(CfC) 和 XGB 使用不同的特征子集：
    - LNN 侧重时序演化特征（FVG_event, impulse, follow-through, volatility）
    - XGB 侧重筛选特征（ST_alignment, session, location）
    
    Args:
        lnn_feature_names: LNN(CfC) 使用的特征名称列表
        xgb_feature_names: XGB 使用的特征名称列表
        use_split: 是否启用特征分离（False 时两者使用全量特征）
    """
    lnn_feature_names: list[str] = field(default_factory=list)
    xgb_feature_names: list[str] = field(default_factory=list)
    use_split: bool = False
    
    def to_dict(self) -> dict:
        return {
            "lnn_feature_names": self.lnn_feature_names,
            "xgb_feature_names": self.xgb_feature_names,
            "use_split": self.use_split,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FeatureMaskConfig":
        return cls(
            lnn_feature_names=data.get("lnn_feature_names", []),
            xgb_feature_names=data.get("xgb_feature_names", []),
            use_split=data.get("use_split", False),
        )
    
    def get_lnn_indices(self, schema_feature_names: list[str]) -> list[int]:
        """获取 LNN 特征在 schema 中的索引"""
        if not self.use_split or not self.lnn_feature_names:
            return list(range(len(schema_feature_names)))
        return [schema_feature_names.index(name) for name in self.lnn_feature_names 
                if name in schema_feature_names]
    
    def get_xgb_indices(self, schema_feature_names: list[str]) -> list[int]:
        """获取 XGB 特征在 schema 中的索引"""
        if not self.use_split or not self.xgb_feature_names:
            return list(range(len(schema_feature_names)))
        return [schema_feature_names.index(name) for name in self.xgb_feature_names 
                if name in schema_feature_names]
    
    def validate(self, schema_feature_names: list[str]) -> tuple[bool, str]:
        """
        验证特征名称是否都存在于 schema 中
        
        Returns:
            (is_valid, error_message)
        """
        if not self.use_split:
            return True, ""
        
        schema_set = set(schema_feature_names)
        
        missing_lnn = [n for n in self.lnn_feature_names if n not in schema_set]
        if missing_lnn:
            return False, f"LNN features not in schema: {missing_lnn}"
        
        missing_xgb = [n for n in self.xgb_feature_names if n not in schema_set]
        if missing_xgb:
            return False, f"XGB features not in schema: {missing_xgb}"
        
        return True, ""
    
    @classmethod
    def default_split(cls) -> "FeatureMaskConfig":
        """
        返回推荐的默认特征分离配置
        
        LNN: 时序演化特征（事件、冲击、跟随、波动）
        XGB: 筛选特征（对齐、位置、时间、ATR比率）
        """
        return cls(
            lnn_feature_names=[
                # 价格/微观结构连续特征
                "log_return", "log_return_zscore", "spread_bps",
                "delta_t_log", "tick_intensity", "ofi_count", "ofi_weighted",
                "kyle_lambda_pct", "pdi",
                # 波动率
                "micro_volatility_pct", "realized_volatility_pct",
                # 热力学
                "market_temperature", "market_entropy", "ts_phase",
                # VPIN
                "vpin", "vpin_zscore",
                # 趋势
                "trend_deviation", "trend_direction", "trend_duration",
                # FVG 事件型特征 (新增)
                "fvg_event", "fvg_impulse_atr",
                # FVG 因果跟随 (新增)
                "fvg_follow_up_3", "fvg_follow_dn_3",
                # ST 压力
                "st_distance_15m",
            ],
            xgb_feature_names=[
                # ST 对齐
                "st_alignment", "st_trend_15m",
                # FVG 位置
                "fvg_location_15m",
                # 时间/Session
                "session", "hour_of_day", "day_of_week",
                "is_session_open", "is_session_close",
                # ATR 比率 (新增)
                "atr_ratio_1m_15m",
                # 趋势持续
                "st_bars_since_flip_15m",
            ],
            use_split=True,
        )


@dataclass
class TrainingConfig:
    """
    v4.0 训练配置
    
    整合所有子模块的配置。
    """
    # 采样配置
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    
    # 降噪配置
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    
    # 特征配置
    features: FeatureConfig = field(default_factory=FeatureConfig)
    
    # Primary Engine 配置
    primary: PrimaryEngineConfig = field(default_factory=PrimaryEngineConfig)
    
    # Meta-Labeling 配置
    meta_label: MetaLabelConfig = field(default_factory=MetaLabelConfig)

    # 多 Horizon 标签配置
    multi_horizon_labels: MultiHorizonConfig = field(default_factory=MultiHorizonConfig)

    # 置信度门控配置（用于训练产物推荐）
    confidence_gate: dict = field(default_factory=dict)
    
    # 特征分离配置 (LNN vs XGB)
    feature_mask: FeatureMaskConfig = field(default_factory=FeatureMaskConfig)
    
    # 交叉验证配置
    cv_n_splits: int = 5
    cv_embargo_pct: float = 0.01
    use_cpcv: bool = False
    cpcv_n_test_splits: int = 2
    
    # CfC 模型配置
    cfc_hidden_size: int = 64
    cfc_num_layers: int = 2
    cfc_dropout: float = 0.1
    cfc_use_layer_norm: bool = True
    cfc_min_tau: float = 0.1
    cfc_max_tau: float = 10.0
    cfc_lr: float = 1e-3
    cfc_weight_decay: float = 1e-5
    
    # XGBoost 配置
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_tree_method: str = "hist"
    xgb_eval_metric: str = "auc"
    xgb_max_bin: int = 256
    xgb_early_stopping_rounds: int = 10
    
    # 训练参数
    sequence_length: int = 100
    
    # 事件锚点模式（用于构造“事件中心”的因果序列）
    # - "fvg_event": 以 FeatureSchema.fvg_event != 0 的 bar 作为 t0（推荐）
    # - "primary_engine": 以 PrimaryEngine 触发的 bar 作为 t0（向后兼容）
    event_anchor_mode: str = "fvg_event"
    batch_size: int = 2048           # RTX 3090/4090 推荐 2048-4096
    epochs: int = 50
    early_stopping_patience: int = 10
    
    # 两阶段训练（先训练 CfC，再训练 XGBoost）
    train_cfc: bool = True
    freeze_cfc_for_xgb: bool = True
    
    # ========== 性能优化参数 ==========
    # DataLoader 多进程加载（16核心推荐 8-12）
    num_workers: int = 8
    pin_memory: bool = True          # CUDA 锁页内存，加速 H2D 传输
    
    # 混合精度训练（RTX 3090 推荐 BF16）
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "bfloat16"  # "bfloat16" | "float16"
    
    # torch.compile 优化（PyTorch 2.0+）
    use_torch_compile: bool = False  # 首次会编译较慢，但后续更快
    compile_mode: str = "reduce-overhead"  # "default" | "reduce-overhead" | "max-autotune"
    
    # XGBoost GPU 高级参数
    xgb_device: str = "cuda"         # "cuda" | "cpu"
    xgb_sampling_method: str = "gradient_based"  # GPU 专用梯度采样
    xgb_grow_policy: str = "lossguide"  # 按损失变化生长叶节点
    
    # CfC 隐状态提取批大小（可独立调大）
    cfc_encoding_batch_size: int = 8192
    
    # 输出路径
    output_dir: str = "models/v4"
    
    # ========== 评估配置 ==========
    # 分位数阈值（用于 Recall@TopQuantile 评估）
    eval_quantiles: list[int] = field(default_factory=lambda: [99, 95, 90, 85])
    
    # Temporal Recall 诊断
    eval_temporal_recall_enabled: bool = True
    eval_temporal_recall_lookahead: int = 3
    
    # 分相位评估
    eval_phase_evaluation: bool = True

    # 固定阈值评估（用于对比）
    eval_fixed_threshold: float = 0.5
    
    @staticmethod
    def _build_feature_config(features_data: dict) -> FeatureConfig:
        thermo_data = features_data.get("thermodynamics", {}) if isinstance(features_data, dict) else {}
        core_fields = {
            k: v
            for k, v in (features_data or {}).items()
            if k not in ("vpin_config", "denoise_config", "thermodynamics")
        }
        return FeatureConfig(
            **core_fields,
            thermodynamics=ThermodynamicsConfig.from_dict(thermo_data),
        )
    
    def to_dict(self) -> dict:
        return {
            "sampling": self.sampling.to_dict(),
            "denoise": self.denoise.to_dict(),
            "features": self.features.to_dict(),
            "primary": self.primary.to_dict(),
            "meta_label": self.meta_label.to_dict(),
            "multi_horizon_labels": {
                "horizons": self.multi_horizon_labels.horizons,
                "threshold_bps": self.multi_horizon_labels.threshold_bps,
                "use_log_returns": self.multi_horizon_labels.use_log_returns,
            },
            "confidence_gate": self.confidence_gate,
            "feature_mask": self.feature_mask.to_dict(),
            "cv_n_splits": self.cv_n_splits,
            "cv_embargo_pct": self.cv_embargo_pct,
            "use_cpcv": self.use_cpcv,
            "cpcv_n_test_splits": self.cpcv_n_test_splits,
            # CfC 配置
            "cfc_hidden_size": self.cfc_hidden_size,
            "cfc_num_layers": self.cfc_num_layers,
            "cfc_dropout": self.cfc_dropout,
            "cfc_use_layer_norm": self.cfc_use_layer_norm,
            "cfc_min_tau": self.cfc_min_tau,
            "cfc_max_tau": self.cfc_max_tau,
            "cfc_lr": self.cfc_lr,
            "cfc_weight_decay": self.cfc_weight_decay,
            # XGBoost 配置
            "xgb_n_estimators": self.xgb_n_estimators,
            "xgb_max_depth": self.xgb_max_depth,
            "xgb_learning_rate": self.xgb_learning_rate,
            "xgb_tree_method": self.xgb_tree_method,
            "xgb_eval_metric": self.xgb_eval_metric,
            "xgb_max_bin": self.xgb_max_bin,
            "xgb_early_stopping_rounds": self.xgb_early_stopping_rounds,
            # 训练参数
            "sequence_length": self.sequence_length,
            "event_anchor_mode": self.event_anchor_mode,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "train_cfc": self.train_cfc,
            "freeze_cfc_for_xgb": self.freeze_cfc_for_xgb,
            # 性能优化参数
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "use_mixed_precision": self.use_mixed_precision,
            "mixed_precision_dtype": self.mixed_precision_dtype,
            "use_torch_compile": self.use_torch_compile,
            "compile_mode": self.compile_mode,
            "xgb_device": self.xgb_device,
            "xgb_sampling_method": self.xgb_sampling_method,
            "xgb_grow_policy": self.xgb_grow_policy,
            "cfc_encoding_batch_size": self.cfc_encoding_batch_size,
            "output_dir": self.output_dir,
            # 评估配置
            "eval_quantiles": self.eval_quantiles,
            "eval_temporal_recall_enabled": self.eval_temporal_recall_enabled,
            "eval_temporal_recall_lookahead": self.eval_temporal_recall_lookahead,
            "eval_phase_evaluation": self.eval_phase_evaluation,
            "eval_fixed_threshold": self.eval_fixed_threshold,
        }
    
    def save(self, path: str | Path) -> None:
        """保存配置到 JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        """从 JSON 加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls(
            sampling=SamplingConfig.from_dict(data.get("sampling", {})),
            denoise=DenoiseConfig(**data.get("denoise", {})) if data.get("denoise") else DenoiseConfig(),
            features=cls._build_feature_config(data.get("features", {})),
            primary=PrimaryEngineConfig(**data.get("primary", {})),
            feature_mask=FeatureMaskConfig.from_dict(data.get("feature_mask", {})),
            cv_n_splits=data.get("cv_n_splits", 5),
            cv_embargo_pct=data.get("cv_embargo_pct", 0.01),
            use_cpcv=data.get("use_cpcv", False),
            cpcv_n_test_splits=data.get("cpcv_n_test_splits", 2),
            # CfC 配置
            cfc_hidden_size=data.get("cfc_hidden_size", 64),
            cfc_num_layers=data.get("cfc_num_layers", 2),
            cfc_dropout=data.get("cfc_dropout", 0.1),
            cfc_use_layer_norm=data.get("cfc_use_layer_norm", True),
            cfc_min_tau=data.get("cfc_min_tau", 0.1),
            cfc_max_tau=data.get("cfc_max_tau", 10.0),
            cfc_lr=data.get("cfc_lr", 1e-3),
            cfc_weight_decay=data.get("cfc_weight_decay", 1e-5),
            # XGBoost 配置
            xgb_n_estimators=data.get("xgb_n_estimators", 100),
            xgb_max_depth=data.get("xgb_max_depth", 6),
            xgb_learning_rate=data.get("xgb_learning_rate", 0.1),
            # 训练参数
            sequence_length=data.get("sequence_length", 100),
            event_anchor_mode=data.get("event_anchor_mode", "fvg_event"),
            batch_size=data.get("batch_size", 32),
            epochs=data.get("epochs", 50),
            early_stopping_patience=data.get("early_stopping_patience", 10),
            train_cfc=data.get("train_cfc", True),
            freeze_cfc_for_xgb=data.get("freeze_cfc_for_xgb", True),
            output_dir=data.get("output_dir", "models/v4"),
        )
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """
        从 YAML 文件加载配置
        
        Args:
            path: YAML 文件路径
            
        Returns:
            TrainingConfig 实例
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        return cls.from_yaml_dict(data)
    
    @classmethod
    def from_yaml_dict(cls, data: dict) -> "TrainingConfig":
        """
        从 YAML 配置字典创建 TrainingConfig
        
        支持 v4 YAML 配置格式：
        - sampling: 采样配置
        - denoise: 降噪配置
        - features: 特征配置
        - primary: Primary Engine 配置
        - meta_labeling: Meta-Labeling 配置
        - cfc: CfC 模型配置
        - xgboost: XGBoost 配置
        - training: 训练配置
        
        Args:
            data: YAML 配置字典
            
        Returns:
            TrainingConfig 实例
        """
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
        
        # === 解析 meta_labeling 配置 ===
        meta_data = _require_section(data, "meta_labeling")
        _require_keys(meta_data, ["triple_barrier", "min_signals", "sample_weight_method", "time_decay_half_life"], "meta_labeling")
        tb_data = meta_data.get("triple_barrier", {})
        _require_keys(
            tb_data,
            [
                "upper_multiplier",
                "lower_multiplier",
                "vertical_bars",
                "volatility_window",
                "volatility_type",
                "ewma_lambda",
                "min_barrier_pct",
                "use_log_returns",
            ],
            "meta_labeling.triple_barrier",
        )
        tb_config = TripleBarrierConfig(
            upper_multiplier=tb_data.get("upper_multiplier", 2.0),
            lower_multiplier=tb_data.get("lower_multiplier", 2.0),
            vertical_bars=tb_data.get("vertical_bars", 20),
            volatility_window=tb_data.get("volatility_window", 20),
            volatility_type=tb_data.get("volatility_type", "realized"),
            ewma_lambda=tb_data.get("ewma_lambda", 0.94),
            min_barrier_pct=tb_data.get("min_barrier_pct", 0.1),
            use_log_returns=tb_data.get("use_log_returns", True),
        )
        meta_config = MetaLabelConfig(
            triple_barrier_config=tb_config,
            min_signals=meta_data.get("min_signals", 10),
            sample_weight_method=meta_data.get("sample_weight_method", "return_based"),
            time_decay_half_life=meta_data.get("time_decay_half_life", 100),
        )
        
        # === 解析 cfc 配置 ===
        cfc_data = _require_section(data, "cfc")
        _require_keys(
            cfc_data,
            ["hidden_size", "num_layers", "dropout", "use_layer_norm", "min_tau", "max_tau", "learning_rate", "weight_decay"],
            "cfc",
        )
        
        # === 解析 xgboost 配置 ===
        xgb_data = _require_section(data, "xgboost")
        _require_keys(
            xgb_data,
            ["n_estimators", "max_depth", "learning_rate", "tree_method", "early_stopping_rounds"],
            "xgboost",
        )
        
        # === 解析 training 配置 ===
        training_data = _require_section(data, "training")
        _require_keys(
            training_data,
            [
                "sequence_length",
                "batch_size",
                "epochs",
                "early_stopping_patience",
                "cv_splits",
                "cv_embargo_pct",
                "use_cpcv",
                "cpcv_n_test_splits",
                "train_cfc",
                "freeze_cfc_for_xgb",
                "num_workers",
                "pin_memory",
                "use_mixed_precision",
                "mixed_precision_dtype",
                "use_torch_compile",
                "compile_mode",
                "xgb_device",
                "xgb_sampling_method",
                "xgb_grow_policy",
                "cfc_encoding_batch_size",
                "output_dir",
            ],
            "training",
        )
        
        # === 解析 feature_mask 配置 ===
        feature_mask_data = data.get("feature_mask", {})
        feature_mask_config = FeatureMaskConfig(
            lnn_feature_names=feature_mask_data.get("lnn_feature_names", []),
            xgb_feature_names=feature_mask_data.get("xgb_feature_names", []),
            use_split=feature_mask_data.get("use_split", False),
        )
        
        # === 解析 evaluation 配置 ===
        eval_data = _require_section(data, "evaluation")
        _require_keys(eval_data, ["quantiles", "temporal_recall", "phase_evaluation", "fixed_threshold"], "evaluation")
        temporal_recall_data = _require_section(eval_data, "temporal_recall")
        _require_keys(temporal_recall_data, ["enabled", "lookahead_bars"], "evaluation.temporal_recall")
        
        multi_horizon_data = _require_section(data, "multi_horizon_labels")
        _require_keys(multi_horizon_data, ["horizons", "threshold_bps", "use_log_returns"], "multi_horizon_labels")

        confidence_gate_data = _require_section(data, "confidence_gate")
        return cls(
            sampling=sampling_config,
            denoise=denoise_config,
            features=features_config,
            primary=primary_config,
            meta_label=meta_config,
            multi_horizon_labels=MultiHorizonConfig(
                horizons=multi_horizon_data.get("horizons", [5, 10, 20]),
                threshold_bps=multi_horizon_data.get("threshold_bps", 3.0),
                use_log_returns=multi_horizon_data.get("use_log_returns", True),
            ),
            confidence_gate=confidence_gate_data,
            feature_mask=feature_mask_config,
            # 交叉验证
            cv_n_splits=training_data.get("cv_splits", 5),
            cv_embargo_pct=training_data.get("cv_embargo_pct", 0.01),
            use_cpcv=training_data.get("use_cpcv", False),
            cpcv_n_test_splits=training_data.get("cpcv_n_test_splits", 2),
            # CfC 配置
            cfc_hidden_size=cfc_data.get("hidden_size", 64),
            cfc_num_layers=cfc_data.get("num_layers", 2),
            cfc_dropout=cfc_data.get("dropout", 0.1),
            cfc_use_layer_norm=cfc_data.get("use_layer_norm", True),
            cfc_min_tau=cfc_data.get("min_tau", 0.1),
            cfc_max_tau=cfc_data.get("max_tau", 10.0),
            cfc_lr=cfc_data.get("learning_rate", 1e-3),
            cfc_weight_decay=cfc_data.get("weight_decay", 1e-5),
            # XGBoost 配置
            xgb_n_estimators=xgb_data.get("n_estimators", 100),
            xgb_max_depth=xgb_data.get("max_depth", 6),
            xgb_learning_rate=xgb_data.get("learning_rate", 0.1),
            xgb_tree_method=xgb_data.get("tree_method", "hist"),
            xgb_eval_metric=xgb_data.get("eval_metric", "auc"),
            xgb_max_bin=xgb_data.get("max_bin", 256),
            xgb_early_stopping_rounds=xgb_data.get("early_stopping_rounds", 10),
            # 训练参数
            sequence_length=training_data.get("sequence_length", 100),
            event_anchor_mode=training_data.get("event_anchor_mode", data.get("event_anchor_mode", "fvg_event")),
            batch_size=training_data.get("batch_size", 32),
            epochs=training_data.get("epochs", 50),
            early_stopping_patience=training_data.get("early_stopping_patience", 10),
            train_cfc=training_data.get("train_cfc", True),
            freeze_cfc_for_xgb=training_data.get("freeze_cfc_for_xgb", True),
            output_dir=training_data.get("output_dir", "models/v4"),
            # 性能优化参数
            num_workers=training_data.get("num_workers", 8),
            pin_memory=training_data.get("pin_memory", True),
            use_mixed_precision=training_data.get("use_mixed_precision", True),
            mixed_precision_dtype=training_data.get("mixed_precision_dtype", "bfloat16"),
            use_torch_compile=training_data.get("use_torch_compile", False),
            compile_mode=training_data.get("compile_mode", "reduce-overhead"),
            xgb_device=training_data.get("xgb_device", "cuda"),
            xgb_sampling_method=training_data.get("xgb_sampling_method", "gradient_based"),
            xgb_grow_policy=training_data.get("xgb_grow_policy", "lossguide"),
            cfc_encoding_batch_size=training_data.get("cfc_encoding_batch_size", 8192),
            # 评估配置
            eval_quantiles=eval_data.get("quantiles", [99, 95, 90, 85]),
            eval_temporal_recall_enabled=temporal_recall_data.get("enabled", True),
            eval_temporal_recall_lookahead=temporal_recall_data.get("lookahead_bars", 3),
            eval_phase_evaluation=eval_data.get("phase_evaluation", True),
            eval_fixed_threshold=eval_data.get("fixed_threshold", 0.5),
        )


@dataclass
class V4TrainingPipeline:
    """
    v4.0 训练管道
    
    完整的端到端训练流程。
    
    使用方式：
    ```python
    config = TrainingConfig()
    pipeline = V4TrainingPipeline(config)
    
    # 从 CSV 加载数据
    pipeline.load_tick_data("data/XAUUSD_Ticks.csv")
    
    # 执行完整训练
    results = pipeline.train()
    
    # 保存模型（自动保存 cfc_encoder.pt + xgb_model.json）
    pipeline.save_model("models/v4/run_001")
    ```
    """
    config: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 内部组件
    _schema: FeatureSchema = field(default_factory=FeatureSchema.default, init=False)
    _sampler: UnifiedSampler | None = field(default=None, init=False)
    _denoise_pipeline: DenoisePipeline | None = field(default=None, init=False)
    _feature_pipeline: FeaturePipelineV4 | None = field(default=None, init=False)
    _primary_engine: PrimaryEngineV4 | None = field(default=None, init=False)
    _meta_label_generator: MetaLabelGenerator | None = field(default=None, init=False)
    
    # 模型组件
    _cfc_encoder: CfCEncoder | None = field(default=None, init=False)
    _cfc_config: CfCConfig | None = field(default=None, init=False)
    _xgb_model: Any = field(default=None, init=False)
    _device: torch.device = field(default=None, init=False)
    
    # 数据
    _ticks: list[Tick] = field(default_factory=list, init=False)
    _bars: list = field(default_factory=list, init=False)
    _features: NDArray | None = field(default=None, init=False)
    _labels: NDArray | None = field(default=None, init=False)
    _sample_weights: NDArray | None = field(default=None, init=False)
    _delta_ts: NDArray | None = field(default=None, init=False)  # 时间间隔
    _event_indices: NDArray | None = field(default=None, init=False)  # 事件索引
    _last_funnel_metrics: dict[str, Any] | None = field(default=None, init=False)  # 对齐审计用
    
    # 特征分离索引
    _lnn_indices: list[int] = field(default_factory=list, init=False)
    _xgb_indices: list[int] = field(default_factory=list, init=False)
    
    def __post_init__(self) -> None:
        """初始化组件"""
        self._init_components()
        self._init_device()
    
    def _init_device(self) -> None:
        """初始化计算设备"""
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            logger.info("Using CUDA device")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
            logger.info("Using MPS device")
        else:
            self._device = torch.device("cpu")
            logger.info("Using CPU device")
    
    def _init_components(self) -> None:
        """初始化所有组件"""
        # 采样器
        self._sampler = UnifiedSampler(self.config.sampling)
        
        # 降噪管道
        self._denoise_pipeline = DenoisePipeline(self.config.denoise)
        
        # 特征管道
        feature_config = FeatureConfig(
            **{k: v for k, v in self.config.features.__dict__.items() 
               if not k.startswith("_") and k != "denoise_config"},
            denoise_config=self.config.denoise,
        )
        self._feature_pipeline = FeaturePipelineV4(feature_config, self._schema)
        
        # Primary 引擎
        self._primary_engine = PrimaryEngineV4(self.config.primary)
        
        # Meta-Label 生成器
        self._meta_label_generator = MetaLabelGenerator(self.config.meta_label)
        
        # 验证特征分离配置
        if self.config.feature_mask.use_split:
            is_valid, err_msg = self.config.feature_mask.validate(self._schema.feature_names)
            if not is_valid:
                raise ValueError(f"Feature mask validation failed: {err_msg}")
            
            # 计算特征索引
            self._lnn_indices = self.config.feature_mask.get_lnn_indices(self._schema.feature_names)
            self._xgb_indices = self.config.feature_mask.get_xgb_indices(self._schema.feature_names)
            lnn_input_dim = len(self._lnn_indices)
            
            logger.info(
                "Feature split enabled",
                lnn_features=len(self._lnn_indices),
                xgb_features=len(self._xgb_indices),
            )
        else:
            # 不分离时使用全量特征
            self._lnn_indices = list(range(self._schema.num_features))
            self._xgb_indices = list(range(self._schema.num_features))
            lnn_input_dim = self._schema.num_features
        
        # CfC 编码器配置
        # time_constant_hint 描述 τ 范围对应的时间尺度语义（便于调参/解释）
        self._cfc_config = CfCConfig(
            input_dim=lnn_input_dim,  # 使用 LNN 特征数量
            hidden_dim=self.config.cfc_hidden_size,
            num_layers=self.config.cfc_num_layers,
            dropout=self.config.cfc_dropout,
            use_layer_norm=self.config.cfc_use_layer_norm,
            min_tau=self.config.cfc_min_tau,
            max_tau=self.config.cfc_max_tau,
            time_constant_hint="microstructure_reaction_1m_to_5m",  # 默认：捕获 1-5 分钟微观结构反应
        )
        
        logger.info(
            "V4TrainingPipeline initialized",
            schema_hash=self._schema.schema_hash,
            n_features=self._schema.num_features,
            lnn_input_dim=lnn_input_dim,
            cfc_hidden_dim=self._cfc_config.hidden_dim,
        )
    
    def load_tick_data(
        self,
        path: str | Path,
        max_ticks: int | None = None,
    ) -> int:
        """
        从 CSV/Parquet 文件加载 Tick 数据（优化版：使用 polars/pandas 向量化）
        
        支持的格式：
        - CSV: time_msc/Time (EET), bid/Bid, ask/Ask
        - Parquet: 推荐，比 CSV 快 10x
        
        性能提升：
        - polars（首选，比 pandas 快 2-5x）或 pandas + pyarrow
        - 向量化时间解析
        - 批量创建 Tick 对象
        
        Args:
            path: CSV/Parquet 文件路径
            max_ticks: 最大加载数量（用于测试）
            
        Returns:
            加载的 Tick 数量
        """
        import numpy as np
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        logger.info(f"Loading data from {path}...")
        
        # 尝试使用 polars（最快）
        try:
            return self._load_tick_data_polars(path, max_ticks)
        except ImportError:
            pass
        
        # 回退到 pandas
        return self._load_tick_data_pandas(path, max_ticks)
    
    def _load_tick_data_polars(
        self,
        path: Path,
        max_ticks: int | None = None,
    ) -> int:
        """使用 polars 加载数据（最快）"""
        import polars as pl
        import numpy as np
        
        # 根据文件扩展名选择加载方式
        if path.suffix == ".parquet":
            df = pl.read_parquet(path)
        else:
            # CSV 格式 - polars 多线程解析
            df = pl.read_csv(
                path,
                n_rows=max_ticks,
                ignore_errors=True,
            )
        
        if max_ticks and len(df) > max_ticks:
            df = df.head(max_ticks)
        
        # 统一列名（支持多种格式）
        col_mapping = {
            "Time (EET)": "time",
            "Time": "time", 
            "time_msc": "time",
            "timestamp": "time",
            "Bid": "bid",
            "Ask": "ask",
        }
        for old_name, new_name in col_mapping.items():
            if old_name in df.columns:
                df = df.rename({old_name: new_name})
        
        # 过滤无效数据
        df = df.filter((pl.col("bid") > 0) & (pl.col("ask") > 0))
        
        # 向量化时间解析
        time_col = df["time"]
        if time_col.dtype == pl.Utf8 or time_col.dtype == pl.String:
            # 字符串时间格式 - MT5 格式: "2025.01.08 01:00:01.141"
            try:
                timestamp_ms = (
                    df.select(
                        pl.col("time")
                        .str.to_datetime(format="%Y.%m.%d %H:%M:%S%.f", strict=False)
                        .dt.timestamp("ms")
                        .alias("ts")
                    )["ts"]
                )
            except Exception:
                # 尝试其他格式
                timestamp_ms = (
                    df.select(
                        pl.col("time")
                        .str.to_datetime(strict=False)
                        .dt.timestamp("ms")
                        .alias("ts")
                    )["ts"]
                )
        else:
            # 已经是数值类型
            timestamp_ms = time_col.cast(pl.Int64)
        
        # 转换为 numpy 数组
        timestamps_us = (timestamp_ms.to_numpy() * 1000).astype(np.int64)  # ms -> us
        bids = df["bid"].to_numpy().astype(np.float64)
        asks = df["ask"].to_numpy().astype(np.float64)
        
        # 批量创建 Tick 对象
        self._ticks = [
            Tick(
                timestamp_us=int(ts),
                bid=float(b),
                ask=float(a),
                bid_volume=0.0,
                ask_volume=0.0,
                last=0.0,
                last_volume=0.0,
                flags=0,
            )
            for ts, b, a in zip(timestamps_us, bids, asks)
        ]
        
        logger.info(f"Loaded {len(self._ticks)} ticks from {path} (polars)")
        return len(self._ticks)
    
    def _load_tick_data_pandas(
        self,
        path: Path,
        max_ticks: int | None = None,
    ) -> int:
        """使用 pandas 加载数据（备选）"""
        import pandas as pd
        import numpy as np
        
        # 根据文件扩展名选择加载方式
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            if max_ticks:
                df = df.head(max_ticks)
        else:
            # CSV 格式 - 使用 pyarrow 引擎（比 c 引擎快 2-3x）
            try:
                df = pd.read_csv(
                    path,
                    nrows=max_ticks,
                    engine="pyarrow",
                    dtype_backend="pyarrow",
                )
            except Exception:
                # fallback to c engine if pyarrow not available
                df = pd.read_csv(path, nrows=max_ticks)
        
        # 统一列名（支持多种格式）
        col_mapping = {
            "Time (EET)": "time",
            "Time": "time", 
            "time_msc": "time",
            "timestamp": "time",
            "Bid": "bid",
            "Ask": "ask",
        }
        df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
        
        # 过滤无效数据
        df = df[(df["bid"] > 0) & (df["ask"] > 0)].copy()
        
        # 向量化时间解析
        time_col = df["time"]
        if time_col.dtype == "object" or str(time_col.dtype).startswith("string"):
            # 字符串时间格式
            sample = str(time_col.iloc[0]) if len(time_col) > 0 else ""
            
            if sample.replace(".", "").replace("-", "").isdigit() and "." not in sample:
                # 纯数字毫秒时间戳
                timestamp_ms = pd.to_numeric(time_col, errors="coerce").astype("int64")
            elif "." in sample and sample.count(".") >= 2:
                # MT5 格式: "2025.01.08 01:00:01.141"
                try:
                    timestamp_ms = (
                        pd.to_datetime(time_col, format="%Y.%m.%d %H:%M:%S.%f", errors="coerce")
                        .astype("int64") // 1_000_000
                    )
                except Exception:
                    # 无毫秒版本
                    timestamp_ms = (
                        pd.to_datetime(time_col, format="%Y.%m.%d %H:%M:%S", errors="coerce")
                        .astype("int64") // 1_000_000
                    )
            else:
                # ISO 格式或其他
                timestamp_ms = (
                    pd.to_datetime(time_col, errors="coerce")
                    .astype("int64") // 1_000_000
                )
        else:
            # 已经是数值类型
            timestamp_ms = time_col.astype("int64")
        
        # 转换为 numpy 数组
        timestamps_us = (timestamp_ms * 1000).values.astype(np.int64)  # ms -> us
        bids = df["bid"].values.astype(np.float64)
        asks = df["ask"].values.astype(np.float64)
        
        # 批量创建 Tick 对象
        self._ticks = [
            Tick(
                timestamp_us=int(ts),
                bid=float(b),
                ask=float(a),
                bid_volume=0.0,
                ask_volume=0.0,
                last=0.0,
                last_volume=0.0,
                flags=0,
            )
            for ts, b, a in zip(timestamps_us, bids, asks)
        ]
        
        logger.info(f"Loaded {len(self._ticks)} ticks from {path} (pandas)")
        return len(self._ticks)
    
    def _validate_primary_meta_xgb_funnel_alignment(
        self,
        *,
        n_bars: int,
        primary_signal_tuples: list[tuple[int, int]],
        meta_result: Any,
    ) -> dict[str, Any]:
        """
        Primary → MetaLabel → XGB 的 funnel 数值对齐校验。
        
        目标：
        - 明确每一层的样本数（可审计）
        - 在“数组长度/索引越界/重复事件”等 silent bug 出现时直接失败
        
        Args:
            n_bars: bar 总数
            primary_signal_tuples: Primary 信号列表 [(bar_idx, direction), ...]
            meta_result: MetaLabelGenerator.generate() 的返回值（MetaLabelResult）
        
        Returns:
            funnel_metrics: 包含各阶段样本数量与基本分布的字典
        """
        # === 1) Primary 层统计 ===
        n_primary = len(primary_signal_tuples)
        if n_primary == 0:
            return {
                "n_bars": int(n_bars),
                "n_primary_signals": 0,
                "n_meta_labels": 0,
                "n_cfc_samples": 0,
                "n_xgb_samples": 0,
                "n_positive_labels": 0,
                "positive_rate": 0.0,
            }
        
        primary_indices = np.array([s[0] for s in primary_signal_tuples], dtype=np.int64)
        primary_dirs = np.array([s[1] for s in primary_signal_tuples], dtype=np.int32)
        
        # 索引合法性（越界是硬错误）
        if np.any(primary_indices < 0) or np.any(primary_indices >= n_bars):
            bad_min = int(np.min(primary_indices))
            bad_max = int(np.max(primary_indices))
            raise ValueError(
                f"Primary 信号索引越界：min={bad_min}, max={bad_max}, n_bars={n_bars}。"
                "这会导致 MetaLabel/XGB 特征抽取错位，必须先修复 PrimaryEngine 或采样器。"
            )
        
        # 方向合法性（0 方向不应作为 meta-label 输入）
        if np.any((primary_dirs != 1) & (primary_dirs != -1)):
            unique_dirs = sorted({int(x) for x in np.unique(primary_dirs)})
            raise ValueError(
                f"Primary 信号方向非法：unique={unique_dirs}（期望仅包含 1/-1）。"
            )
        
        # 重复事件检查（同一 bar 多次触发通常是 bug；会在训练中重复计权）
        n_unique_bars = int(len(np.unique(primary_indices)))
        n_duplicates = int(n_primary - n_unique_bars)
        if n_duplicates > 0:
            raise ValueError(
                f"检测到重复 Primary 事件：n_primary={n_primary}, n_unique_bars={n_unique_bars}。"
                "同一 bar 多次触发会导致样本重复计权，需先定位 PrimaryEngine 触发逻辑。"
            )
        
        # === 2) MetaLabel 层对齐校验 ===
        event_indices = getattr(meta_result, "event_indices", None)
        meta_labels = getattr(meta_result, "meta_labels", None)
        sample_weights = getattr(meta_result, "sample_weights", None)
        primary_directions = getattr(meta_result, "primary_directions", None)
        events = getattr(meta_result, "events", None)
        
        if event_indices is None or meta_labels is None or sample_weights is None or primary_directions is None:
            raise ValueError("MetaLabelResult 缺少关键字段（event_indices/meta_labels/sample_weights/primary_directions）")
        
        n_events = int(len(event_indices))
        if not (len(meta_labels) == n_events == len(sample_weights) == len(primary_directions)):
            raise ValueError(
                "MetaLabel 输出长度不一致："
                f"len(event_indices)={len(event_indices)}, len(meta_labels)={len(meta_labels)}, "
                f"len(sample_weights)={len(sample_weights)}, len(primary_directions)={len(primary_directions)}"
            )
        
        # meta_result 必须来自 primary_signal_tuples（允许未来实现过滤：此处只要求子集关系）
        if n_events > 0 and (not np.all(np.isin(event_indices, primary_indices))):
            raise ValueError(
                "MetaLabel event_indices 不是 Primary 信号的子集："
                "这意味着 Primary → MetaLabel 的索引映射被破坏，训练样本会错位。"
            )
        
        # event_indices 合法性（越界是硬错误）
        if n_events > 0 and (np.any(event_indices < 0) or np.any(event_indices >= n_bars)):
            bad_min = int(np.min(event_indices))
            bad_max = int(np.max(event_indices))
            raise ValueError(
                f"MetaLabel 事件索引越界：min={bad_min}, max={bad_max}, n_bars={n_bars}"
            )
        
        # events 列表长度对齐（如果存在）
        if events is not None and len(events) not in (0, n_events):
            raise ValueError(
                f"MetaLabel events 长度异常：len(events)={len(events)}，期望为 0 或 {n_events}"
            )
        
        n_pos = int(np.sum(meta_labels))
        pos_rate = float(np.mean(meta_labels)) if n_events > 0 else 0.0
        
        return {
            "n_bars": int(n_bars),
            "n_primary_signals": int(n_primary),
            "n_meta_labels": int(n_events),
            # 下游阶段（应与 n_meta_labels 对齐）
            "n_cfc_samples": int(n_events),
            "n_xgb_samples": int(n_events),
            "n_positive_labels": n_pos,
            "positive_rate": pos_rate,
            # 辅助审计字段
            "primary_long": int(np.sum(primary_dirs == 1)),
            "primary_short": int(np.sum(primary_dirs == -1)),
            "event_index_min": int(np.min(event_indices)) if n_events > 0 else None,
            "event_index_max": int(np.max(event_indices)) if n_events > 0 else None,
        }
    
    def process_data(self) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """
        处理数据：采样 → 特征 → 信号 → 标签
        
        Returns:
            (features, labels, sample_weights, delta_ts, event_indices)
        """
        if not self._ticks:
            raise ValueError("No tick data loaded. Call load_tick_data() first.")
        
        logger.info("Starting data processing pipeline...")
        
        # 1. 采样：Tick → Event Bars
        logger.info("Step 1: Sampling ticks to event bars...")
        bar_stream = self._sampler.process_ticks(self._ticks)
        self._bars = bar_stream.bars
        logger.info(f"Generated {len(self._bars)} bars from {len(self._ticks)} ticks")
        
        if len(self._bars) < 100:
            raise ValueError(f"Not enough bars ({len(self._bars)}). Need at least 100.")
        
        # 2. Primary 信号生成
        # 说明：v4.0 默认使用“事件中心”锚点（FVG_event）构造训练样本。
        # - event_anchor_mode="fvg_event": 以 fvg_event!=0 的 bar 作为 t0（推荐）
        # - event_anchor_mode="primary_engine": 向后兼容，仍使用 PrimaryEngine 的触发 bar 作为 t0
        primary_signals = []
        if self.config.event_anchor_mode == "primary_engine":
            logger.info("Step 2: Generating primary signals (primary_engine anchor)...")
            primary_signals = self._primary_engine.generate_signals_batch(self._bars)
            logger.info(f"Generated {len(primary_signals)} primary signals")
            if len(primary_signals) < 10:
                logger.warning(
                    f"Very few primary signals ({len(primary_signals)}). Consider adjusting config."
                )
        else:
            logger.info("Step 2: Skipping PrimaryEngine signals (fvg_event anchor enabled)")
        
        # 3. 特征计算
        logger.info("Step 3: Computing features...")
        
        # 获取趋势信息用于特征计算
        supertrend_directions = []
        supertrend_lines = []
        supertrend_durations = []
        
        # 重新处理以获取每个 bar 的趋势状态
        temp_engine = PrimaryEngineV4(self.config.primary)
        for bar in self._bars:
            temp_engine.update(bar)
            state = temp_engine.current_state
            if state:
                supertrend_directions.append(
                    1 if state.direction.name == "LONG" else 
                    (-1 if state.direction.name == "SHORT" else 0)
                )
                supertrend_lines.append(state.supertrend_line)
                supertrend_durations.append(state.trend_duration)
            else:
                supertrend_directions.append(0)
                supertrend_lines.append(0.0)
                supertrend_durations.append(0)
        
        feature_result = self._feature_pipeline.compute_batch(
            self._bars,
            supertrend_directions=supertrend_directions,
            supertrend_lines=supertrend_lines,
            supertrend_durations=supertrend_durations,
        )
        self._features = feature_result.features
        logger.info(f"Computed features: shape={self._features.shape}")
        
        # 计算时间间隔 delta_t（用于 CfC）
        # 从 bar 的 duration_ms 获取，或使用 bar 时间差
        delta_ts = np.zeros(len(self._bars), dtype=np.float32)
        for i, bar in enumerate(self._bars):
            if i == 0:
                delta_ts[i] = bar.duration_ms / 1000.0 if bar.duration_ms > 0 else 1.0
            else:
                # 使用 bar 持续时间
                delta_ts[i] = bar.duration_ms / 1000.0 if bar.duration_ms > 0 else 1.0
        # 归一化 delta_t（防止极端值）
        delta_ts = np.clip(delta_ts, 0.001, 100.0)
        self._delta_ts = delta_ts
        
        # 4. Meta-Labeling
        logger.info("Step 4: Generating meta-labels...")
        closes = np.array([bar.close for bar in self._bars], dtype=np.float64)
        
        # 构造 (bar_idx, direction) 事件列表（决定 t0 锚点）
        if self.config.event_anchor_mode == "primary_engine":
            signal_tuples = [s.to_tuple() for s in primary_signals]
        else:
            # 事件中心：以 fvg_event 发生的 bar 作为 t0
            fvg_event_idx = self._schema.get_index("fvg_event")
            fvg_event = self._features[:, fvg_event_idx].astype(np.int32)

            # 严格事件锚点提取（防御 fvg_event 被状态化成 regime flag）
            # - 正常情况下 fvg_event 应为脉冲：仅在事件 bar 非零
            # - 若出现连续非零，这里会自动切换为 rising-edge（更严格的 t0）
            anchor_indices, audit_info = extract_fvg_event_t0_anchors(
                fvg_event,
                audit_rising_edge=True,
            )
            event_indices = anchor_indices.astype(np.int64)
            directions = np.sign(fvg_event[event_indices]).astype(np.int32)
            signal_tuples = list(zip(event_indices.tolist(), directions.tolist()))
            
            logger.info(
                "Using fvg_event as event anchor",
                n_fvg_events=int(len(event_indices)),
                long=int(np.sum(directions == 1)),
                short=int(np.sum(directions == -1)),
                anchor_count=int(audit_info.get("anchor_count", 0)),
                rising_edge_count=int(audit_info.get("rising_edge_count", 0)),
                ratio_diff=float(audit_info.get("ratio_diff", 0.0)),
                is_pulse=bool(audit_info.get("is_pulse", True)),
            )
        
        meta_result = self._meta_label_generator.generate(closes, signal_tuples)
        
        # Primary → MetaLabel → XGB funnel 对齐校验（失败即停止，避免 silent misalignment）
        self._last_funnel_metrics = self._validate_primary_meta_xgb_funnel_alignment(
            n_bars=len(self._bars),
            primary_signal_tuples=signal_tuples,
            meta_result=meta_result,
        )
        logger.info(
            "Funnel alignment (Primary → MetaLabel → XGB) OK",
            **self._last_funnel_metrics,
        )
        
        # 提取训练数据
        event_indices = meta_result.event_indices
        self._labels = meta_result.meta_labels
        self._sample_weights = meta_result.sample_weights
        self._event_indices = event_indices  # 保存用于 train()
        
        # 提取对应的特征和 delta_t
        X = self._features[event_indices]
        y = self._labels
        weights = self._sample_weights
        dt = delta_ts[event_indices]
        
        logger.info(
            f"Training data prepared: X={X.shape}, y={y.shape}, "
            f"positive_rate={np.mean(y):.2%}"
        )
        
        return X, y, weights, dt, event_indices
    
    def get_cv_splitter(
        self,
        event_times: NDArray | None = None,
        holding_periods: NDArray | None = None,
    ) -> PurgedKFold | CPCVSplitter:
        """
        获取交叉验证分割器
        
        Args:
            event_times: 事件时间索引
            holding_periods: 持仓周期
            
        Returns:
            交叉验证分割器
        """
        embargo_config = EmbargoConfig(embargo_pct=self.config.cv_embargo_pct)
        
        if self.config.use_cpcv:
            return CPCVSplitter(
                n_splits=self.config.cv_n_splits,
                n_test_splits=self.config.cpcv_n_test_splits,
                embargo_config=embargo_config,
            )
        else:
            return PurgedKFold(
                n_splits=self.config.cv_n_splits,
                embargo_config=embargo_config,
            )
    
    def _build_sequences(
        self,
        features: NDArray,
        delta_ts: NDArray,
        labels: NDArray,
        weights: NDArray,
        indices: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        构建 CfC 训练序列
        
        每个样本变成 (sequence_length, n_features) 的序列。
        
        Args:
            features: 全部特征 [n_bars, n_features]
            delta_ts: 时间间隔 [n_bars]
            labels: 事件标签 [n_events]
            weights: 样本权重 [n_events]
            indices: 事件在 features 中的索引 [n_events]
            
        Returns:
            seq_features: [n_events, seq_len, n_features]
            seq_delta_ts: [n_events, seq_len]
            seq_labels: [n_events]
            seq_weights: [n_events]
        """
        seq_len = self.config.sequence_length
        n_events = len(indices)
        n_features = features.shape[1]
        
        seq_features = np.zeros((n_events, seq_len, n_features), dtype=np.float32)
        seq_delta_ts = np.ones((n_events, seq_len), dtype=np.float32)
        
        for i, idx in enumerate(indices):
            # 获取该事件之前的 seq_len 个 bar
            start_idx = max(0, idx - seq_len + 1)
            end_idx = idx + 1
            actual_len = end_idx - start_idx
            
            # 右对齐填充
            seq_features[i, -actual_len:, :] = features[start_idx:end_idx]
            seq_delta_ts[i, -actual_len:] = delta_ts[start_idx:end_idx]
        
        return seq_features, seq_delta_ts, labels, weights
    
    def _train_cfc_encoder(
        self,
        seq_features: NDArray,
        seq_delta_ts: NDArray,
        labels: NDArray,
        weights: NDArray,
        val_features: NDArray | None = None,
        val_delta_ts: NDArray | None = None,
        val_labels: NDArray | None = None,
    ) -> tuple[CfCEncoder, dict]:
        """
        训练 CfC 编码器（使用 meta-label 二分类目标）
        
        Returns:
            encoder: 训练好的 CfC 编码器
            metrics: 训练指标
        """
        from alphaos.v4.models.cfc import MetaClassifier
        
        # 创建 CfC 编码器
        encoder = CfCEncoder(self._cfc_config).to(self._device)
        classifier = MetaClassifier(
            self._cfc_config.hidden_dim,
            self._cfc_config.dropout,
        ).to(self._device)
        
        # 优化器
        optimizer = optim.AdamW(
            list(encoder.parameters()) + list(classifier.parameters()),
            lr=self.config.cfc_lr,
            weight_decay=self.config.cfc_weight_decay,
        )
        
        # 损失函数（带样本权重）
        # 使用 BCEWithLogitsLoss 以兼容混合精度训练（autocast）
        # MetaClassifier 输出 logits，BCEWithLogitsLoss 内部应用 Sigmoid
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # 转换为 Tensor（保持在 CPU，让 DataLoader 按需传输）
        X_train = torch.tensor(seq_features, dtype=torch.float32)
        dt_train = torch.tensor(seq_delta_ts, dtype=torch.float32)
        y_train = torch.tensor(labels, dtype=torch.float32)
        w_train = torch.tensor(weights, dtype=torch.float32)
        
        # 创建 DataLoader（启用多进程加载和锁页内存）
        dataset = TensorDataset(X_train, dt_train, y_train, w_train)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers if self._device.type == "cuda" else 0,
            pin_memory=self.config.pin_memory and self._device.type == "cuda",
            persistent_workers=self.config.num_workers > 0 and self._device.type == "cuda",
            prefetch_factor=2 if self.config.num_workers > 0 and self._device.type == "cuda" else None,
        )
        
        # 混合精度训练（BF16 推荐，不需要 GradScaler）
        use_amp = self.config.use_mixed_precision and self._device.type == "cuda"
        amp_dtype = torch.bfloat16 if self.config.mixed_precision_dtype == "bfloat16" else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
        
        if use_amp:
            logger.info(f"Mixed precision training enabled: {self.config.mixed_precision_dtype}")
        
        # 验证集
        if val_features is not None:
            X_val = torch.tensor(val_features, dtype=torch.float32, device=self._device)
            dt_val = torch.tensor(val_delta_ts, dtype=torch.float32, device=self._device)
            y_val = torch.tensor(val_labels, dtype=torch.float32, device=self._device)
        
        # 训练
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.config.epochs):
            encoder.train()
            classifier.train()
            
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_x, batch_dt, batch_y, batch_w in dataloader:
                # 移动到 GPU（DataLoader 使用 pin_memory 加速）
                batch_x = batch_x.to(self._device, non_blocking=True)
                batch_dt = batch_dt.to(self._device, non_blocking=True)
                batch_y = batch_y.to(self._device, non_blocking=True)
                batch_w = batch_w.to(self._device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
                
                # 混合精度前向传播
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    # 前向传播
                    hidden, _ = encoder(batch_x, batch_dt)
                    pred = classifier(hidden)
                    
                    # 计算加权损失
                    loss = criterion(pred, batch_y)
                    weighted_loss = (loss * batch_w).mean()
                
                # 反向传播（如果用 FP16 需要 scaler，BF16 不需要）
                if use_amp and amp_dtype == torch.float16:
                    scaler.scale(weighted_loss).backward()
                else:
                    weighted_loss.backward()
                
                # 梯度裁剪和优化器更新
                if use_amp and amp_dtype == torch.float16:
                    # FP16: 使用 scaler
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(classifier.parameters()),
                        max_norm=1.0,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # BF16 或 FP32: 正常流程
                    torch.nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(classifier.parameters()),
                        max_norm=1.0,
                    )
                    optimizer.step()
                
                epoch_loss += weighted_loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / max(1, n_batches)
            
            # 验证
            if val_features is not None:
                encoder.eval()
                classifier.eval()
                
                with torch.no_grad():
                    val_hidden, _ = encoder(X_val, dt_val)
                    val_pred = classifier(val_hidden)
                    val_loss = criterion(val_pred, y_val).mean().item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        'encoder': encoder.state_dict(),
                        'classifier': classifier.state_dict(),
                    }
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(
                        f"CfC Epoch {epoch + 1}/{self.config.epochs}: "
                        f"train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}"
                    )
                
                # 早停
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(
                        f"CfC Epoch {epoch + 1}/{self.config.epochs}: loss={avg_loss:.4f}"
                    )
        
        # 恢复最佳权重
        if best_state is not None:
            encoder.load_state_dict(best_state['encoder'])
        
        encoder.eval()
        
        metrics = {
            "final_train_loss": avg_loss,
            "best_val_loss": best_val_loss if val_features is not None else None,
            "epochs_trained": epoch + 1,
        }
        
        return encoder, metrics
    
    def _extract_cfc_hidden_states(
        self,
        encoder: CfCEncoder,
        seq_features: NDArray,
        seq_delta_ts: NDArray,
    ) -> NDArray:
        """
        从训练好的 CfC 编码器提取隐状态
        
        ⚠️ SEMANTIC CONSTRAINT (架构约束)
        =================================
        CfC hidden states represent LATENT SYSTEM STATES, not interpretable features.
        
        返回的 hidden_states 应该：
          ✅ 直接 concat 到 xgb_features 后面送入 XGB
          ✅ 作为系统"记忆"表示
        
        ❌ 禁止对 hidden_states：
          - 做标准化（Normalization/StandardScaler）
          - 做 SHAP / 特征重要性分析当"因子"解读
          - 将某个维度当作有物理意义的指标
        
        CfC 回答的是 "what happens next?"（时序动态建模）
        
        Args:
            encoder: CfC 编码器
            seq_features: [n_samples, seq_len, n_features]
            seq_delta_ts: [n_samples, seq_len]
            
        Returns:
            hidden_states: [n_samples, hidden_dim]
        """
        encoder.eval()
        
        # 使用更大的批大小进行推理（提取隐状态）
        batch_size = self.config.cfc_encoding_batch_size
        
        # 混合精度推理
        use_amp = self.config.use_mixed_precision and self._device.type == "cuda"
        amp_dtype = torch.bfloat16 if self.config.mixed_precision_dtype == "bfloat16" else torch.float16
        
        hidden_states = []
        
        with torch.no_grad():
            for i in range(0, len(seq_features), batch_size):
                # 按批次移动到 GPU（避免一次性占用太多显存）
                batch_x = torch.tensor(
                    seq_features[i:i+batch_size], 
                    dtype=torch.float32, 
                    device=self._device
                )
                batch_dt = torch.tensor(
                    seq_delta_ts[i:i+batch_size],
                    dtype=torch.float32,
                    device=self._device
                )
                
                # 混合精度推理
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    hidden, _ = encoder(batch_x, batch_dt)
                
                hidden_states.append(hidden.float().cpu().numpy())  # 转回 FP32
        
        return np.concatenate(hidden_states, axis=0)
    
    def train(self) -> dict:
        """
        执行完整训练流程
        
        两阶段训练：
        1. 训练 CfC 编码器（使用 meta-label 目标）
        2. 冻结 CfC，提取隐状态，训练 XGBoost
        
        Returns:
            训练结果字典
        """
        # 处理数据（返回 event_indices 以避免重复调用 generate_signals_batch）
        X_all, y_all, weights_all, dt_all, original_indices = self.process_data()
        
        # === funnel 数值对齐（二次校验：数组长度必须一致）===
        n_events = int(len(y_all))
        if not (
            X_all.shape[0] == n_events
            and len(weights_all) == n_events
            and len(dt_all) == n_events
            and len(original_indices) == n_events
        ):
            raise ValueError(
                "训练数据对齐失败："
                f"X_all={X_all.shape}, len(y_all)={len(y_all)}, len(weights_all)={len(weights_all)}, "
                f"len(dt_all)={len(dt_all)}, len(event_indices)={len(original_indices)}"
            )

        # 空数据检查
        if X_all.shape[0] == 0:
            logger.error(
                "No training samples produced (empty X). "
                "Likely cause: Primary engine generated 0 signals. "
                "Adjust PrimaryEngineConfig (e.g. require_fvg=False, lower thresholds) or verify data.",
            )
            return {
                "fold_results": [],
                "avg_results": {},
                "config": self.config.to_dict(),
                "schema_hash": self._schema.schema_hash,
                "n_samples": 0,
                "n_features": int(X_all.shape[1]) if X_all.ndim == 2 else 0,
                "positive_rate": 0.0,
            }
        
        logger.info("Starting two-stage model training...")
        
        # 特征分离：提取 LNN 和 XGB 特征子集
        if self.config.feature_mask.use_split:
            features_lnn = self._features[:, self._lnn_indices]
            features_xgb = self._features[:, self._xgb_indices]
            logger.info(
                f"Feature split: LNN={features_lnn.shape[1]}, XGB={features_xgb.shape[1]}"
            )
        else:
            features_lnn = self._features
            features_xgb = self._features
        
        # 构建序列（使用 LNN 特征）
        logger.info("Building training sequences for CfC...")
        seq_features, seq_delta_ts, seq_labels, seq_weights = self._build_sequences(
            features_lnn, self._delta_ts, y_all, weights_all, original_indices
        )
        logger.info(f"Sequence shape: {seq_features.shape}")
        
        if int(seq_features.shape[0]) != n_events:
            raise ValueError(
                f"CfC 序列样本数不一致：seq_features.shape[0]={seq_features.shape[0]} vs n_events={n_events}"
            )
        
        # XGB 用的特征（按 event_indices 抽取）
        xgb_base_features = features_xgb[original_indices]
        logger.info(f"XGB base features shape: {xgb_base_features.shape}")
        
        if int(xgb_base_features.shape[0]) != n_events:
            raise ValueError(
                f"XGB 样本数不一致：xgb_base_features.shape[0]={xgb_base_features.shape[0]} vs n_events={n_events}"
            )
        
        # 获取 CV 分割器
        cv = self.get_cv_splitter()
        
        # 存储结果
        fold_results = []
        best_encoder = None
        best_xgb = None
        
        # 交叉验证训练
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(seq_features, seq_labels)):
            logger.info(f"Training fold {fold_idx + 1}/{cv.get_n_splits()}...")
            
            # 分割数据
            seq_X_train = seq_features[train_idx]
            seq_dt_train = seq_delta_ts[train_idx]
            y_train = seq_labels[train_idx]
            w_train = seq_weights[train_idx]
            
            seq_X_test = seq_features[test_idx]
            seq_dt_test = seq_delta_ts[test_idx]
            y_test = seq_labels[test_idx]
            w_test = seq_weights[test_idx]

            # 单类标签检查
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                fold_results.append({
                    "fold": fold_idx,
                    "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                    "error": "single_class_labels",
                    "train_pos_rate": float(np.mean(y_train)) if len(y_train) else 0.0,
                    "test_pos_rate": float(np.mean(y_test)) if len(y_test) else 0.0,
                })
                logger.warning(
                    "Skipping fold due to single-class labels",
                    fold=fold_idx,
                )
                continue
            
            try:
                # 阶段 1：训练 CfC 编码器
                if self.config.train_cfc:
                    logger.info(f"Fold {fold_idx + 1}: Training CfC encoder...")
                    encoder, cfc_metrics = self._train_cfc_encoder(
                        seq_X_train, seq_dt_train, y_train, w_train,
                        val_features=seq_X_test,
                        val_delta_ts=seq_dt_test,
                        val_labels=y_test,
                    )
                else:
                    # 不训练 CfC，使用随机初始化
                    encoder = CfCEncoder(self._cfc_config).to(self._device)
                    encoder.eval()
                    cfc_metrics = {}
                
                # 阶段 2：提取隐状态并训练 XGBoost
                # ⚠️ SEMANTIC CONSTRAINT: hidden_states 是 latent system state，
                #    不做标准化，不做 SHAP 解读当因子
                logger.info(f"Fold {fold_idx + 1}: Extracting CfC hidden states...")
                hidden_train = self._extract_cfc_hidden_states(encoder, seq_X_train, seq_dt_train)
                hidden_test = self._extract_cfc_hidden_states(encoder, seq_X_test, seq_dt_test)
                
                # 组合特征：XGB 特征子集 + CfC 隐状态
                # 使用特征分离时，XGB 使用独立的特征集
                # ⚠️ hidden_states 直接 concat，不做任何 preprocessing
                X_train_combined = np.concatenate([
                    xgb_base_features[train_idx],  # XGB 特征（按 event_indices 抽取）
                    hidden_train,                   # CfC latent state（不可解释）
                ], axis=1)
                X_test_combined = np.concatenate([
                    xgb_base_features[test_idx],  # XGB 特征（按 event_indices 抽取）
                    hidden_test,                   # CfC latent state（不可解释）
                ], axis=1)
                
                logger.info(f"Combined features shape: {X_train_combined.shape}")
                
                # 训练 XGBoost
                # ⚠️ SEMANTIC CONSTRAINT: XGB 是 FILTER / CONDITIONER，不是 DECISION MAKER
                # - CfC 回答 "what happens next?"
                # - XGB 回答 "is this a tradable instance?"
                # - 最终决策还需配合 filters（warmup, signal, phase, risk）
                logger.info(f"Fold {fold_idx + 1}: Training XGBoost...")
                import xgboost as xgb
                
                # 设备选择（从配置读取）
                xgb_device = self.config.xgb_device
                if xgb_device == "cuda" and not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU for XGBoost")
                    xgb_device = "cpu"
                
                # GPU 专用参数
                xgb_params = {
                    "n_estimators": self.config.xgb_n_estimators,
                    "max_depth": self.config.xgb_max_depth,
                    "learning_rate": self.config.xgb_learning_rate,
                    "tree_method": self.config.xgb_tree_method,
                    "device": xgb_device,
                    "eval_metric": self.config.xgb_eval_metric,
                    "early_stopping_rounds": self.config.xgb_early_stopping_rounds,
                    "random_state": 42,
                    "n_jobs": -1,  # 使用所有 CPU 核心
                }
                
                # GPU 特有优化
                if xgb_device == "cuda":
                    xgb_params["sampling_method"] = self.config.xgb_sampling_method
                    xgb_params["grow_policy"] = self.config.xgb_grow_policy
                    xgb_params["max_bin"] = self.config.xgb_max_bin
                
                xgb_model = xgb.XGBClassifier(**xgb_params)
                
                # 抑制 XGBoost GPU 设备警告（eval_set 内部评估时的已知行为）
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")
                    xgb_model.fit(
                        X_train_combined, y_train,
                        sample_weight=w_train,
                        eval_set=[(X_test_combined, y_test)],
                        verbose=False,
                    )
                
                # 评估（显式使用 DMatrix 避免 GPU 设备警告）
                if xgb_device == "cuda":
                    # GPU 模式：使用 DMatrix 确保数据正确传输
                    dtest = xgb.DMatrix(X_test_combined, label=y_test)
                    y_prob = xgb_model.get_booster().predict(dtest)
                    y_pred = (y_prob > self.config.eval_fixed_threshold).astype(int)
                else:
                    # CPU 模式：使用标准 sklearn API
                    y_pred = xgb_model.predict(X_test_combined)
                    y_prob = xgb_model.predict_proba(X_test_combined)[:, 1]
                
                from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
                
                # === 传统指标（向后兼容）===
                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                
                # === 分位数评估指标（核心改进）===
                # 提取 ts_phase 用于分相位评估
                ts_phase_test = None
                try:
                    ts_phase_idx = self._schema.get_index("ts_phase")
                    # 从原始特征中提取（不是 combined，因为 combined 包含 hidden）
                    ts_phase_test = self._features[original_indices[test_idx], ts_phase_idx].astype(int)
                except (KeyError, ValueError):
                    pass  # ts_phase 不存在时跳过分相位评估
                
                quantile_metrics = compute_quantile_evaluation_metrics(
                    y_true=y_test,
                    y_prob=y_prob,
                    ts_phase=ts_phase_test,
                    sample_weights=w_test,
                    quantiles=self.config.eval_quantiles,
                    fixed_threshold=self.config.eval_fixed_threshold,
                )
                
                # === Temporal Recall 诊断（event + N bars）===
                # 仅在配置启用时执行
                temporal_metrics = {}
                if self.config.eval_temporal_recall_enabled:
                    try:
                        # 构建 fold-consistent quantile thresholds
                        eval_quantiles = self.config.eval_quantiles if self.config.eval_quantiles else [95, 90, 85]
                        thresholds_by_quantile = {
                            q: float(np.percentile(y_prob, q)) for q in eval_quantiles
                        }
                        
                        # 获取 test 子集的 anchor indices（原始 bar 索引）
                        anchor_indices_test = original_indices[test_idx]
                        
                        temporal_metrics = compute_temporal_recall_batch(
                            anchor_indices=anchor_indices_test,
                            y_true=y_test,
                            features_full=self._features,
                            delta_ts=self._delta_ts,
                            encoder=encoder,
                            xgb_model=xgb_model,
                            sequence_length=self.config.sequence_length,
                            thresholds_by_quantile=thresholds_by_quantile,
                            lookahead_bars=self.config.eval_temporal_recall_lookahead,
                            lnn_indices=self._lnn_indices if self.config.feature_mask.use_split else None,
                            xgb_indices=self._xgb_indices if self.config.feature_mask.use_split else None,
                            device=str(self._device),
                        )
                    except Exception as e:
                        logger.warning(f"Temporal recall diagnosis failed: {e}")
                        temporal_metrics = {"error": str(e)}
                
                fold_result = {
                    "fold": fold_idx,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    # 传统指标（向后兼容）
                    "accuracy": accuracy,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "cfc_metrics": cfc_metrics,
                    # 新增：分位数评估指标
                    "quantile_metrics": quantile_metrics,
                    # 新增：Temporal Recall 诊断
                    "temporal_recall": temporal_metrics,
                }
                fold_results.append(fold_result)
                
                # 日志输出：传统指标 + 分位数指标
                logger.info(
                    f"Fold {fold_idx + 1}: accuracy={accuracy:.4f}, "
                    f"auc={auc:.4f}, precision={precision:.4f}, recall={recall:.4f}"
                )
                logger.info(
                    f"Fold {fold_idx + 1} quantile evaluation:\n"
                    f"{format_quantile_metrics_log(quantile_metrics)}"
                )
                
                # Temporal Recall 日志输出
                if temporal_metrics and "error" not in temporal_metrics:
                    eval_quantiles = self.config.eval_quantiles if self.config.eval_quantiles else [95, 90, 85]
                    logger.info(
                        f"Fold {fold_idx + 1} {format_temporal_recall_log(temporal_metrics, eval_quantiles)}"
                    )
                
                # 保存最佳模型（按 AUC）
                if best_encoder is None or auc > max([r.get("auc", 0) for r in fold_results[:-1]], default=0):
                    best_encoder = encoder
                    best_xgb = xgb_model
                
            except ImportError as e:
                logger.error(f"Import error: {e}. Skipping fold.")
                fold_results.append({"fold": fold_idx, "error": str(e)})
            except Exception as e:
                logger.error(f"Error in fold {fold_idx + 1}: {e}")
                fold_results.append({"fold": fold_idx, "error": str(e)})
        
        # 汇总结果
        valid_results = [r for r in fold_results if "accuracy" in r]
        if valid_results:
            avg_results = {
                # 传统指标
                "avg_accuracy": np.mean([r["accuracy"] for r in valid_results]),
                "avg_auc": np.mean([r["auc"] for r in valid_results]),
                "avg_precision": np.mean([r["precision"] for r in valid_results]),
                "avg_recall": np.mean([r["recall"] for r in valid_results]),
                "std_accuracy": np.std([r["accuracy"] for r in valid_results]),
                "std_auc": np.std([r["auc"] for r in valid_results]),
            }
            
            # 汇总分位数指标（PR-AUC 和 Recall@各分位数）
            results_with_quantile = [r for r in valid_results if "quantile_metrics" in r]
            if results_with_quantile:
                avg_results["avg_pr_auc"] = np.mean([
                    r["quantile_metrics"]["pr_auc"] for r in results_with_quantile
                ])
                avg_results["std_pr_auc"] = np.std([
                    r["quantile_metrics"]["pr_auc"] for r in results_with_quantile
                ])
                # 成本加权 PR-AUC（非交易数口径）
                avg_results["avg_weighted_pr_auc"] = np.mean([
                    r["quantile_metrics"].get("weighted_pr_auc", 0.0) for r in results_with_quantile
                ])
                avg_results["std_weighted_pr_auc"] = np.std([
                    r["quantile_metrics"].get("weighted_pr_auc", 0.0) for r in results_with_quantile
                ])
                
                # 各分位数的平均 Recall
                for q in ["P99", "P95", "P90", "P85"]:
                    recalls = [
                        r["quantile_metrics"]["recall_at_quantile"].get(q, 0)
                        for r in results_with_quantile
                    ]
                    avg_results[f"avg_recall_{q}"] = np.mean(recalls)
                    avg_results[f"std_recall_{q}"] = np.std(recalls)
                    
                    # 成本加权 Recall（非交易数口径）
                    w_recalls = [
                        r["quantile_metrics"]["weighted_recall_at_quantile"].get(q, 0)
                        for r in results_with_quantile
                    ]
                    avg_results[f"avg_weighted_recall_{q}"] = np.mean(w_recalls)
                    avg_results[f"std_weighted_recall_{q}"] = np.std(w_recalls)
            
            # 日志输出
            logger.info(
                f"Cross-validation results: "
                f"accuracy={avg_results['avg_accuracy']:.4f}±{avg_results['std_accuracy']:.4f}, "
                f"auc={avg_results['avg_auc']:.4f}±{avg_results['std_auc']:.4f}"
            )
            
            # 新增：分位数指标汇总日志
            if "avg_pr_auc" in avg_results:
                logger.info(
                    f"Quantile metrics summary: "
                    f"PR-AUC={avg_results['avg_pr_auc']:.4f}±{avg_results.get('std_pr_auc', 0):.4f}, "
                    f"wPR-AUC={avg_results.get('avg_weighted_pr_auc', 0):.4f}±{avg_results.get('std_weighted_pr_auc', 0):.4f}"
                )
                recall_summary = ", ".join([
                    f"R@{q}={avg_results.get(f'avg_recall_{q}', 0):.2%}"
                    for q in ["P95", "P90", "P85"]
                ])
                w_recall_summary = ", ".join([
                    f"wR@{q}={avg_results.get(f'avg_weighted_recall_{q}', 0):.2%}"
                    for q in ["P95", "P90", "P85"]
                ])
                logger.info(f"Recall@Quantile: {recall_summary} | weighted: {w_recall_summary}")
            
            # === Temporal Recall 汇总 ===
            results_with_temporal = [r for r in valid_results if "temporal_recall" in r and "error" not in r.get("temporal_recall", {})]
            if results_with_temporal:
                eval_quantiles = self.config.eval_quantiles if self.config.eval_quantiles else [95, 90, 85]
                lookahead = self.config.eval_temporal_recall_lookahead if self.config.eval_temporal_recall_lookahead else 3
                
                # 汇总各 N、各 quantile 的 temporal recall
                for q in eval_quantiles:
                    for w in range(lookahead + 1):
                        key = f"temporal_recall_P{q}_N{w}"
                        temporal_recalls = [
                            r["temporal_recall"].get(key, 0.0)
                            for r in results_with_temporal
                            if key in r.get("temporal_recall", {})
                        ]
                        if temporal_recalls:
                            avg_results[f"avg_{key}"] = np.mean(temporal_recalls)
                            avg_results[f"std_{key}"] = np.std(temporal_recalls)
                
                # Temporal Recall Summary 日志（矩阵输出）
                logger.info("Temporal Recall Summary (event + N bars):")
                for w in range(lookahead + 1):
                    q_strs = []
                    for q in eval_quantiles:
                        key = f"avg_temporal_recall_P{q}_N{w}"
                        if key in avg_results:
                            q_strs.append(f"R@P{q}={avg_results[key]:.1%}")
                    if q_strs:
                        logger.info(f"  N={w}: {', '.join(q_strs)}")
        else:
            avg_results = {}
        
        # 保存最佳模型
        self._cfc_encoder = best_encoder
        self._xgb_model = best_xgb
        
        # Funnel metrics（追踪样本在各阶段的数量）
        # ⚠️ 统计口径与 process_data 对齐校验保持一致
        n_bars = len(self._bars) if self._bars else 0
        n_primary = (
            int(self._last_funnel_metrics.get("n_primary_signals", n_events))
            if self._last_funnel_metrics is not None
            else n_events
        )
        funnel_metrics = {
            "n_bars": int(n_bars),
            "n_primary_signals": int(n_primary),  # = meta-label 输入口径
            "n_meta_labels": int(n_events),  # = meta-label 输出/训练样本口径
            "n_cfc_samples": int(n_events),
            "n_xgb_samples": int(n_events),
            "n_positive_labels": int(np.sum(y_all)),
            "positive_rate": float(np.mean(y_all)) if n_events > 0 else 0.0,
            # 各阶段转化率
            "primary_to_bar_rate": float(n_primary / n_bars) if n_bars > 0 else 0.0,
            "meta_to_primary_rate": float(n_events / n_primary) if n_primary > 0 else 0.0,
            "positive_to_meta_rate": float(np.mean(y_all)) if n_events > 0 else 0.0,
        }
        
        logger.info(
            f"Funnel: {funnel_metrics['n_bars']} bars → "
            f"{funnel_metrics['n_primary_signals']} primary → "
            f"{funnel_metrics['n_meta_labels']} meta → "
            f"{funnel_metrics['n_positive_labels']} positive ({funnel_metrics['positive_rate']:.1%})"
        )
        
        return {
            "fold_results": fold_results,
            "avg_results": avg_results,
            "config": self.config.to_dict(),
            "schema_hash": self._schema.schema_hash,
            "n_samples": len(y_all),
            "n_features": X_all.shape[1],
            "positive_rate": float(np.mean(y_all)),
            # 新增：Funnel 指标
            "funnel_metrics": funnel_metrics,
        }
    
    def compute_temporal_recall_diagnosis(
        self,
        event_indices: NDArray,
        y_true: NDArray,
        lookahead_bars: int = 3,
        quantiles: list[int] = [95, 90],
    ) -> dict:
        """
        计算 Temporal Recall 诊断（离线分析用，不参与训练）
        
        目的：检测 "事件抓住了，但时间窗口错过了" 的问题
        
        对于每个事件 index t：
        - 检查 t, t+1, t+2, ..., t+lookahead_bars 的模型概率
        - 如果窗口内任一 bar 超过分位阈值，则算作 temporal hit
        
        ⚠️ 这是纯诊断功能，保持因果性：
        - 只使用 t..t+N 的特征（不使用未来 label）
        - 不修改训练目标或样本权重
        
        Args:
            event_indices: 事件索引（来自 Primary gate）
            y_true: 真实 meta-label（与 event_indices 对齐）
            lookahead_bars: 向后看的 bar 数（默认 3）
            quantiles: 要评估的分位数阈值
            
        Returns:
            temporal_recall_metrics: 包含各窗口大小、各分位数的 temporal recall
        """
        if self._cfc_encoder is None or self._xgb_model is None:
            logger.warning("No trained model available for temporal recall diagnosis")
            return {}
        
        if self._features is None or self._delta_ts is None:
            logger.warning("No features available for temporal recall diagnosis")
            return {}
        
        logger.info(f"Computing temporal recall diagnosis (lookahead={lookahead_bars} bars)...")
        
        import xgboost as xgb
        
        # 特征分离
        if self.config.feature_mask.use_split:
            features_lnn = self._features[:, self._lnn_indices]
            features_xgb = self._features[:, self._xgb_indices]
        else:
            features_lnn = self._features
            features_xgb = self._features
        
        n_bars = len(self._features)
        n_events = len(event_indices)
        n_positive = int(np.sum(y_true))
        
        # 为每个事件构建候选窗口 (t, t+1, ..., t+lookahead)
        # 注意：避免越界
        candidate_probs = []  # [n_events, lookahead+1]
        
        for event_idx, orig_idx in enumerate(event_indices):
            window_probs = []
            for offset in range(lookahead_bars + 1):
                bar_idx = orig_idx + offset
                if bar_idx >= n_bars:
                    # 越界时使用 NaN
                    window_probs.append(np.nan)
                    continue
                
                # 构建该 bar 的序列（复用现有逻辑）
                seq_len = self.config.sequence_length
                start_idx = max(0, bar_idx - seq_len + 1)
                end_idx = bar_idx + 1
                actual_len = end_idx - start_idx
                
                seq_features = np.zeros((1, seq_len, features_lnn.shape[1]), dtype=np.float32)
                seq_delta_ts = np.ones((1, seq_len), dtype=np.float32)
                
                seq_features[0, -actual_len:, :] = features_lnn[start_idx:end_idx]
                seq_delta_ts[0, -actual_len:] = self._delta_ts[start_idx:end_idx]
                
                # 提取 CfC hidden state
                hidden = self._extract_cfc_hidden_states(
                    self._cfc_encoder, seq_features, seq_delta_ts
                )
                
                # 组合 XGB 输入
                xgb_input = np.concatenate([
                    features_xgb[bar_idx:bar_idx+1],
                    hidden,
                ], axis=1)
                
                # 预测概率
                try:
                    prob = self._xgb_model.predict_proba(xgb_input)[0, 1]
                except Exception:
                    prob = np.nan
                
                window_probs.append(prob)
            
            candidate_probs.append(window_probs)
        
        candidate_probs = np.array(candidate_probs)  # [n_events, lookahead+1]
        
        # 计算 temporal recall
        results = {
            "lookahead_bars": lookahead_bars,
            "n_events": n_events,
            "n_positive": n_positive,
        }
        
        for q in quantiles:
            # 计算整体分位阈值（基于所有预测概率）
            valid_probs = candidate_probs[~np.isnan(candidate_probs)]
            threshold = np.percentile(valid_probs, q)
            
            results[f"threshold_P{q}"] = float(threshold)
            
            # 对于每个窗口大小，计算 temporal recall
            for window_size in range(1, lookahead_bars + 2):  # 1 到 lookahead+1
                # 检查窗口内是否有任一概率超过阈值
                window_probs = candidate_probs[:, :window_size]
                window_max = np.nanmax(window_probs, axis=1)
                
                # Temporal hit: 正样本且窗口内有超过阈值的预测
                positive_mask = (y_true == 1)
                temporal_hits = (window_max >= threshold) & positive_mask
                temporal_recall = np.sum(temporal_hits) / n_positive if n_positive > 0 else 0.0
                
                # 也计算 temporal precision
                selected = (window_max >= threshold)
                temporal_precision = np.sum(temporal_hits) / np.sum(selected) if np.sum(selected) > 0 else 0.0
                
                results[f"temporal_recall_P{q}_window{window_size}"] = float(temporal_recall)
                results[f"temporal_precision_P{q}_window{window_size}"] = float(temporal_precision)
                results[f"n_selected_P{q}_window{window_size}"] = int(np.sum(selected))
        
        # 对比：原始 point-in-time recall
        point_probs = candidate_probs[:, 0]  # 仅事件时刻
        for q in quantiles:
            threshold = np.percentile(valid_probs, q)
            positive_mask = (y_true == 1)
            point_hits = (point_probs >= threshold) & positive_mask
            point_recall = np.sum(point_hits) / n_positive if n_positive > 0 else 0.0
            results[f"point_recall_P{q}"] = float(point_recall)
        
        # 日志输出
        logger.info("Temporal Recall Diagnosis:")
        for q in quantiles:
            point_r = results[f"point_recall_P{q}"]
            temp_r_max = results[f"temporal_recall_P{q}_window{lookahead_bars + 1}"]
            improvement = (temp_r_max - point_r) / point_r * 100 if point_r > 0 else 0
            logger.info(
                f"  P{q}: Point={point_r:.1%} → Window{lookahead_bars+1}={temp_r_max:.1%} "
                f"(+{improvement:.1f}%)"
            )
        
        return results
    
    def save_model(self, path: str | Path) -> None:
        """
        保存完整模型包
        
        保存文件：
        - cfc_encoder.pt: CfC 编码器权重
        - cfc_config.json: CfC 配置
        - xgb_model.json: XGBoost 模型
        - schema.json: 特征 Schema
        - config.json: 训练配置
        - bundle_meta.json: 元数据
        
        Args:
            path: 保存目录
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存 CfC 编码器
        if self._cfc_encoder is not None:
            torch.save(
                self._cfc_encoder.state_dict(),
                path / "cfc_encoder.pt"
            )
            logger.info(f"Saved CfC encoder to {path / 'cfc_encoder.pt'}")
        else:
            logger.warning("No CfC encoder to save (training may have failed)")
        
        # 保存 CfC 配置
        if self._cfc_config is not None:
            with open(path / "cfc_config.json", "w", encoding="utf-8") as f:
                json.dump(self._cfc_config.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 保存 XGBoost 模型
        if self._xgb_model is not None:
            self._xgb_model.save_model(str(path / "xgb_model.json"))
            logger.info(f"Saved XGBoost model to {path / 'xgb_model.json'}")
        else:
            logger.warning("No XGBoost model to save (training may have failed)")
        
        # 保存 Feature Schema
        self._schema.save(path / "schema.json")
        
        # 保存训练配置
        self.config.save(path / "config.json")
        
        # 保存元数据
        # 计算 mask hashes（用于推理端校验，防止 silent diverge）
        lnn_mask_hash = compute_feature_list_hash(self.config.feature_mask.lnn_feature_names)
        xgb_mask_hash = compute_feature_list_hash(self.config.feature_mask.xgb_feature_names)
        schema_mask_combo_hash = compute_schema_mask_combo_hash(
            self._schema.schema_hash, lnn_mask_hash, xgb_mask_hash
        )
        
        meta = {
            "schema_hash": self._schema.schema_hash,
            "feature_names": self._schema.feature_names,
            "hidden_dim": self._cfc_config.hidden_dim if self._cfc_config else 0,
            "num_layers": self._cfc_config.num_layers if self._cfc_config else 0,
            "n_features": self._schema.num_features,
            "sequence_length": self.config.sequence_length,
            "event_anchor_mode": self.config.event_anchor_mode,
            # 特征分离配置
            "feature_mask": self.config.feature_mask.to_dict(),
            "lnn_feature_names": self.config.feature_mask.lnn_feature_names,
            "xgb_feature_names": self.config.feature_mask.xgb_feature_names,
            "use_feature_split": self.config.feature_mask.use_split,
            "lnn_input_dim": len(self.config.feature_mask.get_lnn_indices(self._schema.feature_names)) if self.config.feature_mask.use_split else self._schema.num_features,
            "xgb_input_dim": len(self.config.feature_mask.get_xgb_indices(self._schema.feature_names)) if self.config.feature_mask.use_split else self._schema.num_features,
            # Mask hashes（用于推理端校验，防止 research/live diverge）
            "lnn_mask_hash": lnn_mask_hash,
            "xgb_mask_hash": xgb_mask_hash,
            "schema_mask_combo_hash": schema_mask_combo_hash,
            # 评估配置（便于 research/live 对齐）
            "evaluation_config": {
                "quantiles": self.config.eval_quantiles,
                "temporal_recall_enabled": self.config.eval_temporal_recall_enabled,
                "temporal_recall_lookahead": self.config.eval_temporal_recall_lookahead,
                "phase_evaluation": self.config.eval_phase_evaluation,
            },
            # 推荐的 confidence_gate 配置（推理端使用）
            # Cold Start Semantics (Hybrid 3-Stage, 语义冻结):
            # - Stage 1 HARD_FROZEN: buffer < min_required → 禁止交易
            # - Stage 2 FIXED_FALLBACK: min_required <= buffer < full_buffer → 使用 fixed_fallback_threshold
            # - Stage 3 ROLLING_QUANTILE: buffer >= full_buffer → 使用 rolling_quantile(q_by_phase)
            "recommended_confidence_gate": self.config.confidence_gate,
        }
        with open(path / "bundle_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        logger.info(
            f"Bundle hashes: schema={self._schema.schema_hash[:8]}, "
            f"lnn_mask={lnn_mask_hash}, xgb_mask={xgb_mask_hash}, "
            f"combo={schema_mask_combo_hash}"
        )
        
        logger.info(f"Model bundle saved to {path}")
    
    def save_schema(self, path: str | Path) -> None:
        """保存 Feature Schema"""
        self._schema.save(path)
    
    @property
    def schema(self) -> FeatureSchema:
        """获取 Feature Schema"""
        return self._schema
    
    @property
    def features(self) -> NDArray | None:
        """获取计算的特征"""
        return self._features
    
    @property
    def labels(self) -> NDArray | None:
        """获取标签"""
        return self._labels
    
    @property
    def cfc_encoder(self) -> CfCEncoder | None:
        """获取训练好的 CfC 编码器"""
        return self._cfc_encoder
    
    @property
    def xgb_model(self) -> Any:
        """获取训练好的 XGBoost 模型"""
        return self._xgb_model
