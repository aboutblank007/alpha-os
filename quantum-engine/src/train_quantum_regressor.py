#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案 C 量子电路回归训练：预测 y_hat = target_next_close_change

核心优化（基于量子计算生产环境方案.md）：
- float64 精度：金融微观结构必需
- 严格角度归一化：[-π, π] 映射，充分利用 Bloch 球状态空间
- 12 qubits 甜点区：M2 Pro CPU 最优配置
- 分特征定制预处理：rsi/ema_spread/volume_shock 等各有专属变换
- 目标值 [-0.9, 0.9]：留出缓冲防止量子测量饱和
- lightning.qubit + adjoint 微分：CPU 双精度模拟 + 快速梯度
"""

from __future__ import annotations

import argparse
import json
from loguru import logger
import math
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# ================== 方案 C 核心：强制 float64 ==================
import torch
torch.set_default_dtype(torch.float64)

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pennylane as qml

# ================== M2 Pro 多核优化 ==================
# OMP 线程亲和性
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("KMP_BLOCKTIME", "0")
# PennyLane 批量并行：使用多进程执行 batch 内的量子电路
# os.environ.setdefault("PENNYLANE_NUM_THREADS", "8")

# logger = logging.getLogger("quantum-engine.train")


TARGET_COL = "target_next_close_change"
TS_COL = "timestamp"
SYMBOL_COL = "symbol"

# 方案 C 附录 7.1 规范的特征分类
PHYSICAL_COLS = {"rsi", "wick_ratio"}
# 趋势特征：改用相对比例
EMA_RATIO_COLS = {"ema_fast", "ema_slow"}
# 压力特征
PRESSURE_COLS = {"dom_pressure_proxy"}
VOLUME_SHOCK_COLS = {"volume_shock"}
# 其他数值特征采用通用处理


@dataclass(frozen=True)
class TrainConfig:
    data: str
    outdir: str
    symbol: str
    qubits: int
    layers: int
    batch_size: int
    lr: float
    weight_decay: float
    epochs: int
    val_ratio: float
    timestamp_format: str
    max_train_rows: int
    max_seconds: int
    patience: int
    min_delta: float
    seed: int


@dataclass(frozen=True)
class TrainReport:
    started_at: str
    finished_at: str
    duration_sec: float
    train_rows_total: int
    train_rows_used: int
    val_rows: int
    feature_cols: List[str]
    qubits: int
    layers: int
    device: str
    backend: str
    best_val_loss: float
    best_epoch: int


class QuantumFeatureTransformer:
    """
    方案 C 特征工程器：按附录 7.1 规范对不同特征定制预处理到量子友好区间。
    
    特征处理规范：
    - rsi: x / 100 * π → [0, π]
    - ema_spread: RobustScaler → MinMax(-1,1) * π → [-π, π]
    - volume_shock: ln(1+x) → StandardScaler → 约 [-2, 2]
    - 其他数值特征: RobustScaler → clip(-3,3) → * (π/3) → [-π, π]
    """
    
    def __init__(self, n_qubits: int, feature_cols: List[str], seed: int = 42):
        self.n_qubits = n_qubits
        self.feature_cols = feature_cols
        self.seed = seed
        
        # 按特征类型分组
        self._physical_idx: List[int] = []
        self._ema_ratio_idx: List[int] = []
        self._pressure_idx: List[int] = []
        self._volume_shock_idx: List[int] = []
        self._general_idx: List[int] = []
        
        # 查找 close 列索引以计算 EMA ratio
        self._close_idx = -1
        for i, col in enumerate(feature_cols):
            if col.lower() == "close":
                self._close_idx = i
                break
                
        for i, col in enumerate(feature_cols):
            col_lower = col.lower()
            if col_lower in PHYSICAL_COLS:
                self._physical_idx.append(i)
            elif col_lower in EMA_RATIO_COLS or "ema_spread" in col_lower:
                # ema_spread 也要通过 ratio 处理
                self._ema_ratio_idx.append(i)
            elif col_lower in PRESSURE_COLS:
                self._pressure_idx.append(i)
            elif col_lower in VOLUME_SHOCK_COLS or "volume_shock" in col_lower:
                self._volume_shock_idx.append(i)
            else:
                self._general_idx.append(i)
        
        # 各类型的 scaler
        self._ema_robust = RobustScaler()
        self._ema_minmax = MinMaxScaler(feature_range=(-1, 1))
        self._pressure_robust = RobustScaler()
        self._vol_std = StandardScaler()
        self._general_robust = RobustScaler()
        
        # PCA 降维到 n_qubits
        from sklearn.decomposition import PCA
        self._pca = PCA(n_components=n_qubits, random_state=seed)
        
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> "QuantumFeatureTransformer":
        """拟合各类型的 scaler 和 PCA。"""
        X = X.astype(np.float64)
        
        # 1. 对各类特征分别拟合 scaler
        if self._ema_ratio_idx:
            # 尝试计算 ratio: (close - ema) / ema
            # 如果 X 包含必要的列
            ema_data = X[:, self._ema_ratio_idx].copy()
            if self._close_idx >= 0:
                for j, i in enumerate(self._ema_ratio_idx):
                    col_name = self.feature_cols[i].lower()
                    if "ema" in col_name:
                        # 避免除以 0
                        denom = np.where(X[:, i] == 0, 1e-9, X[:, i])
                        ema_data[:, j] = (X[:, self._close_idx] - X[:, i]) / denom
            
            self._ema_robust.fit(ema_data)
            ema_robust_out = self._ema_robust.transform(ema_data)
            self._ema_minmax.fit(ema_robust_out)

        if self._pressure_idx:
            pressure_data = X[:, self._pressure_idx]
            self._pressure_robust.fit(pressure_data)
            
        if self._volume_shock_idx:
            vol_data = X[:, self._volume_shock_idx]
            # ln(1+x) 变换，处理负值
            vol_log = np.sign(vol_data) * np.log1p(np.abs(vol_data))
            self._vol_std.fit(vol_log)
        
        if self._general_idx:
            general_data = X[:, self._general_idx]
            self._general_robust.fit(general_data)
        
        # 2. 先做一次完整变换，再拟合 PCA
        X_transformed = self._transform_before_pca(X)
        self._pca.fit(X_transformed)
        
        self._fitted = True
        return self
    
    def _transform_before_pca(self, X: np.ndarray) -> np.ndarray:
        """应用分类预处理（PCA 之前）。"""
        X = X.astype(np.float64)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        out = np.zeros((n_samples, n_features), dtype=np.float64)
        
        # 方案 C 2.0: 所有特征都已经是 TS_Rank ([0, 1])
        # 我们只需根据列名决定是映射到 [0, pi] 还是 [-pi, pi]
        
        for i in range(n_features):
            col_name = self.feature_cols[i].lower()
            val = X[:, i]
            
            if col_name in PHYSICAL_COLS or "rsi" in col_name or "wick" in col_name:
                # [0, 1] -> [0, pi]
                out[:, i] = val * np.pi
            elif "dom_pressure" in col_name or "imbalance" in col_name or "spread" in col_name:
                # [0, 1] -> [-pi, pi] 假设 0.5 是中心
                out[:, i] = (val - 0.5) * 2.0 * np.pi
            else:
                # 默认 [0, 1] -> [-pi/2, pi/2]
                out[:, i] = (val - 0.5) * np.pi
        
        
        # 其他特征: RobustScaler → clip(-3,3) → * (π/3) → [-π, π]
        if self._general_idx:
            general_data = X[:, self._general_idx]
            general_robust_out = self._general_robust.transform(general_data)
            for j, i in enumerate(self._general_idx):
                clipped = np.clip(general_robust_out[:, j], -3, 3)
                out[:, i] = clipped * (np.pi / 3.0)
        
        return out
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """完整变换：分类预处理 + PCA 降维到 n_qubits。"""
        if not self._fitted:
            raise RuntimeError("QuantumFeatureTransformer 未 fit")
        
        X_before_pca = self._transform_before_pca(X)
        X_pca = self._pca.transform(X_before_pca)
        
        # PCA 输出需要再次归一化到 [-π, π]（防止溢出）
        X_pca_clipped = np.clip(X_pca, -np.pi, np.pi)
        return X_pca_clipped.astype(np.float64)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """拟合并变换。"""
        self.fit(X)
        return self.transform(X)
    
    def get_state(self) -> Dict[str, Any]:
        """获取可序列化的状态（用于 pickle 保存）。"""
        return {
            "n_qubits": self.n_qubits,
            "feature_cols": self.feature_cols,
            "seed": self.seed,
            "_physical_idx": self._physical_idx,
            "_ema_ratio_idx": self._ema_ratio_idx,
            "_pressure_idx": self._pressure_idx,
            "_volume_shock_idx": self._volume_shock_idx,
            "_general_idx": self._general_idx,
            "_ema_robust": self._ema_robust,
            "_ema_minmax": self._ema_minmax,
            "_pressure_robust": self._pressure_robust,
            "_vol_std": self._vol_std,
            "_general_robust": self._general_robust,
            "_pca": self._pca,
            "_fitted": self._fitted,
        }
    
    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "QuantumFeatureTransformer":
        """从状态恢复。"""
        obj = cls(
            n_qubits=state["n_qubits"],
            feature_cols=state["feature_cols"],
            seed=state["seed"],
        )
        obj._rsi_idx = state["_rsi_idx"]
        obj._ema_spread_idx = state["_ema_spread_idx"]
        obj._volume_shock_idx = state["_volume_shock_idx"]
        obj._general_idx = state["_general_idx"]
        obj._ema_robust = state["_ema_robust"]
        obj._ema_minmax = state["_ema_minmax"]
        obj._vol_std = state["_vol_std"]
        obj._general_robust = state["_general_robust"]
        obj._pca = state["_pca"]
        obj._fitted = state["_fitted"]
        return obj


class TargetScaler:
    """
    目标变量归一化器（对称版）：
    将 target 归一化到 [-0.8, 0.8]，强制 0 映射为 0。
    解决 MinMaxScaler 在数据分布偏斜时导致的零点漂移问题（即模型输出 0 被解码为 Buy/Sell）。
    """
    
    def __init__(self):
        self.scale_ = 1.0
        self._fitted = False
    
    def fit(self, y: np.ndarray) -> "TargetScaler":
        y = y.astype(np.float64)
        max_abs = np.max(np.abs(y))
        # 防止除以 0
        if max_abs == 0:
            self.scale_ = 1.0
        else:
            self.scale_ = max_abs
        self._fitted = True
        return self
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TargetScaler 未 fit")
        y = y.astype(np.float64)
        # 线性映射：x / max_abs * 0.8 -> [-0.8, 0.8]
        return (y / self.scale_) * 0.8
    
    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """逆变换：将模型预测值还原为原始尺度。"""
        if not self._fitted:
            raise RuntimeError("TargetScaler 未 fit")
        y_scaled = np.asarray(y_scaled, dtype=np.float64)
        return (y_scaled / 0.8) * self.scale_





def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="aligned CSV 路径")
    p.add_argument("--outdir", required=True, help="模型产物输出目录（建议 quantum-engine/models）")
    p.add_argument("--symbol", default="XAUUSD", help="按 symbol 过滤（默认 XAUUSD）")
    # 方案 C：默认 12 qubits（M2 Pro 甜点区）
    p.add_argument("--qubits", type=int, default=12, help="量子比特数（默认 12，M2 Pro 甜点区）")
    p.add_argument("--layers", type=int, default=3, help="电路层数（StronglyEntanglingLayers）")
    p.add_argument("--batch-size", type=int, default=64, help="batch 大小（真量子仿真很慢，别太大）")
    p.add_argument("--lr", type=float, default=3e-3, help="学习率")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    p.add_argument("--epochs", type=int, default=50, help="最大 epoch")
    p.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例（按时间尾部切分）")
    p.add_argument("--timestamp-format", default="%Y.%m.%d %H:%M", help="timestamp 解析格式")
    p.add_argument(
        "--max-train-rows",
        type=int,
        default=80000,
        help="训练最大样本数（从训练段尾部取，默认 8 万；设 0 表示不限制）",
    )
    p.add_argument(
        "--max-seconds",
        type=int,
        default=6 * 3600,
        help="训练最大耗时（秒），默认 6 小时",
    )
    p.add_argument("--patience", type=int, default=5, help="验证集 loss 连续不改善的容忍 epoch 数（早停）")
    p.add_argument("--min-delta", type=float, default=1e-4, help="判定为'改善'的最小 val loss 下降幅度")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_dataframe(
    path: Path,
    symbol: str,
    timestamp_format: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在：{path}")
    df = pd.read_csv(path)
    required = {TS_COL, SYMBOL_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列：{sorted(missing)}")

    df = df[df[SYMBOL_COL] == symbol].copy()
    if df.empty:
        raise ValueError(f"过滤 symbol={symbol} 后无数据")

    df[TS_COL] = pd.to_datetime(df[TS_COL], format=timestamp_format, errors="coerce")
    df = df.dropna(subset=[TS_COL, TARGET_COL]).sort_values(TS_COL).reset_index(drop=True)
    return df


def _select_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {TS_COL, SYMBOL_COL, TARGET_COL}
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise ValueError("未找到可用的数值特征列")
    return cols


def _apply_ts_rank(df: pd.DataFrame, feature_cols: List[str], window: int = 1440) -> pd.DataFrame:
    """
    对指定特征列执行滚动时间序列排名 (TS_Rank)。
    产生值范围 [0, 1]。
    """
    logger.info("⏳ 执行 Rolling TS_Rank (window={})", window)
    df_rank = df.copy()
    for col in feature_cols:
        # 使用 pandas 的 rolling rank
        df_rank[col] = df[col].rolling(window=window, min_periods=1).rank(pct=True)
    
    # 填充开头的 NaN (虽然 min_periods=1 已经填充，但防万一)
    df_rank[feature_cols] = df_rank[feature_cols].fillna(0.5)
    return df_rank


def _time_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    if n < 1000:
        raise ValueError(f"样本量太小：{n}（至少建议 1000）")
    split = int(math.floor(n * (1.0 - val_ratio)))
    split = max(1, min(split, n - 1))
    return X[:split], y[:split], X[split:], y[split:]


def _pick_backend() -> str:
    # 方案 C：优先 lightning.qubit（CPU 双精度 + adjoint 微分）
    try:
        _ = qml.device("lightning.qubit", wires=2)
        return "lightning.qubit"
    except Exception:
        return "default.qubit"


class QuantumRegressor(nn.Module):
    """
    方案 C 量子回归器（M2 Pro 多核优化）：
    - AngleEmbedding (rotation='Y')：严格 [-π, π] 角度嵌入
    - StronglyEntanglingLayers：捕捉非线性特征
    - PauliZ 测量：输出 [-1, 1]
    - 多线程并行：lightning.qubit 内置 OpenMP 支持
    """
    
    def __init__(self, n_qubits: int, n_layers: int, backend: str):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend

        # M2 Pro 优化：为 lightning.qubit 后端指定线程数
        # 注意：lightning.qubit 内部使用 OpenMP 并行化状态向量运算
        dev = qml.device(backend, wires=n_qubits)

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        # 方案 C：lightning.qubit 支持 adjoint differentiation
        diff_method = "adjoint" if backend == "lightning.qubit" else "parameter-shift"

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def qnode(inputs, weights):
            # inputs: (n_qubits,) 已经在 [-π, π] 区间
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_qubits)
        # TorchLayer 内部会遍历 batch，lightning.qubit 会用 OpenMP 加速每次仿真
        z = self.q_layer(x)
        y = self.head(z)
        return y.squeeze(-1)


def main() -> int:
    # 强制日志输出到终端和文件
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    args = _parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.outdir, exist_ok=True)
    logger.add(os.path.join(args.outdir, "train.log"), rotation="10 MB")
    
    logger.info("🎬 启动模型训练任务")
    cfg = TrainConfig(
        data=args.data,
        outdir=args.outdir,
        symbol=args.symbol,
        qubits=args.qubits,
        layers=args.layers,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        val_ratio=args.val_ratio,
        timestamp_format=args.timestamp_format,
        max_train_rows=args.max_train_rows,
        max_seconds=args.max_seconds,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
    )
    _seed_everything(cfg.seed)

    started = time.time()
    started_at = datetime.now().isoformat(timespec="seconds")

    data_path = Path(cfg.data).expanduser().resolve()
    outdir = Path(cfg.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    reports_dir = outdir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("📥 读取数据：{}", str(data_path))
    df = _load_dataframe(data_path, cfg.symbol, cfg.timestamp_format)
    logger.info("✅ 数据读取完成：rows={} symbol={}", len(df), cfg.symbol)
    feature_cols = _select_feature_cols(df)
    logger.info("🧩 特征列选择完成：n_features={}（示例={}）", len(feature_cols), feature_cols[:10])

    # 方案 C 2.0: 应用 TS_Rank
    df = _apply_ts_rank(df, feature_cols, window=1440)

    # 方案 C：使用 float64
    X_raw = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.5).to_numpy(dtype=np.float64)
    y_raw = df[TARGET_COL].to_numpy(dtype=np.float64)

    # 时间切分（先切，再在训练段尾部限量）
    X_train_raw, y_train_raw, X_val_raw, y_val_raw = _time_split(X_raw, y_raw, cfg.val_ratio)
    train_total = len(X_train_raw)

    if cfg.max_train_rows and train_total > cfg.max_train_rows:
        # 从训练段"尾部"取样，贴近最新分布（更符合上线）
        X_train_raw = X_train_raw[-cfg.max_train_rows:]
        y_train_raw = y_train_raw[-cfg.max_train_rows:]

    # 方案 C：使用 QuantumFeatureTransformer（分特征定制预处理 + PCA）
    feature_transformer = QuantumFeatureTransformer(
        n_qubits=cfg.qubits,
        feature_cols=feature_cols,
        seed=cfg.seed,
    )
    X_train = feature_transformer.fit_transform(X_train_raw)
    X_val = feature_transformer.transform(X_val_raw)

    # 方案 C：目标值归一化到 [-0.9, 0.9]
    target_scaler = TargetScaler()
    y_train = target_scaler.fit_transform(y_train_raw)
    y_val = target_scaler.transform(y_val_raw)

    # Torch（方案 C：强制 CPU + float64）
    device = torch.device("cpu")
    backend = _pick_backend()
    logger.info(
        "配置：symbol={} qubits={} layers={} backend={} device={} dtype=float64 train={}/{} val={} feats={}",
        cfg.symbol,
        cfg.qubits,
        cfg.layers,
        backend,
        str(device),
        len(X_train),
        train_total,
        len(X_val),
        len(feature_cols),
    )

    model = QuantumRegressor(n_qubits=cfg.qubits, n_layers=cfg.layers, backend=backend).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=1.0)  # Huber，抗噪

    train_ds = TensorDataset(
        torch.from_numpy(X_train).to(torch.float64),
        torch.from_numpy(y_train).to(torch.float64),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).to(torch.float64),
        torch.from_numpy(y_val).to(torch.float64),
    )
    # M2 Pro 优化：DataLoader 多进程加载
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False,  # CPU 训练不需要 pin_memory
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.batch_size, 
        shuffle=False,
        num_workers=num_workers,
    )

    best_val = float("inf")
    best_epoch = -1
    best_path = outdir / "quantum_regressor_best.pt"
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        if time.time() - started > cfg.max_seconds:
            logger.warning("达到 max_seconds=%d，提前停止训练。", cfg.max_seconds)
            break

        model.train()
        tr_losses: List[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses: List[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_losses.append(loss.item())

        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va = float(np.mean(va_losses)) if va_losses else float("nan")
        logger.info("epoch={} train_loss={:.6f} val_loss={:.6f}", epoch, tr, va)

        if va < (best_val - cfg.min_delta):
            best_val = va
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model_state": model.state_dict()}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                logger.warning(
                    "触发早停：patience={} min_delta={:.6g}（best_val={:.6f}）",
                    cfg.patience,
                    cfg.min_delta,
                    best_val,
                )
                break

    # 保存预处理与元信息
    artifacts = {
        "config": asdict(cfg),
        "feature_cols": feature_cols,
        "backend": backend,
        "timestamp_format": cfg.timestamp_format,
        "scheme": "C",  # 标记方案 C
    }
    with (outdir / "artifacts.json").open("w", encoding="utf-8") as f:
        json.dump(artifacts, f, ensure_ascii=False, indent=2)

    # 方案 C：保存 QuantumFeatureTransformer 和 TargetScaler
    with (outdir / "feature_transformer.pkl").open("wb") as f:
        pickle.dump(feature_transformer.get_state(), f)
    with (outdir / "target_scaler.pkl").open("wb") as f:
        pickle.dump(target_scaler, f)

    finished_at = datetime.now().isoformat(timespec="seconds")
    duration = time.time() - started
    report = TrainReport(
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=duration,
        train_rows_total=train_total,
        train_rows_used=len(X_train),
        val_rows=len(X_val),
        feature_cols=feature_cols,
        qubits=cfg.qubits,
        layers=cfg.layers,
        device=str(device),
        backend=backend,
        best_val_loss=float(best_val),
        best_epoch=int(best_epoch),
    )
    report_path = reports_dir / f"train_report_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)

    logger.info("✅ 训练完成：best_val_loss=%.6f best_epoch=%d report=%s", best_val, best_epoch, str(report_path))
    logger.info("✅ 模型产物：%s", str(outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
