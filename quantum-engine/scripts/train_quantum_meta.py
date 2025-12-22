#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量子 Meta-Labeling 模型训练：变分量子电路 (VQC) 二分类
严格遵循 docs/交易风控离场策略研究.md 规范

功能：
- 预测 QNN 的信号是否值得下注（Bet / Pass）
- 输入：qnn_prediction + 市场状态特征 (降维到 10 qubits)
- 输出：量子测量期望值 <Z>，映射到 [0, 1] 概率
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml

# ================== 核心配置：强制 float64 ==================
torch.set_default_dtype(torch.float64)

# M2 Pro 多核优化
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("PENNYLANE_NUM_THREADS", "8")

logger = logging.getLogger("quantum-engine.train_meta")

# ================== 特征处理与标签逻辑 ==================

def create_meta_labels(df: pd.DataFrame, cost_multiplier: float = 0.1) -> pd.Series:
    """创建 Meta-Label：1=正确（值得下注），0=错误"""
    pred = df["qnn_prediction"].values
    actual = df["target_next_close_change"].values
    atr = df["atr"].values
    
    direction_correct = (pred * actual) > 0
    cost = np.abs(atr) * cost_multiplier
    profit_sufficient = np.abs(actual) > cost
    
    meta_label = (direction_correct & profit_sufficient).astype(int)
    return pd.Series(meta_label, index=df.index, name="meta_label")

class QuantumFeatureTransformer:
    """遵循方案 C 规范的特征预处理"""
    def __init__(self, n_qubits: int, feature_cols: List[str], seed: int = 42):
        self.n_qubits = n_qubits
        self.feature_cols = feature_cols
        self.seed = seed
        self._fitted = False
        
        # 使用 PCA 降维到 n_qubits
        from sklearn.decomposition import PCA
        self._scaler = RobustScaler()
        self._pca = PCA(n_components=n_qubits, random_state=seed)
    
    def fit(self, X: np.ndarray) -> "QuantumFeatureTransformer":
        X = X.astype(np.float64)
        X_scaled = self._scaler.fit_transform(X)
        self._pca.fit(X_scaled)
        self._fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise RuntimeError("Transformer not fitted")
        X_scaled = self._scaler.transform(X)
        X_pca = self._pca.transform(X_scaled)
        # 映射到 [-pi, pi]
        return np.clip(X_pca, -np.pi, np.pi)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def get_state(self) -> Dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "feature_cols": self.feature_cols,
            "seed": self.seed,
            "_scaler": self._scaler,
            "_pca": self._pca,
            "_fitted": self._fitted,
        }

# ================== 量子神经网络 (VQC) ==================

class QuantumClassifier(nn.Module):
    """严格遵循文档 6.2 节定义的电路结构"""
    def __init__(self, n_qubits: int, n_layers: int, backend: str):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        dev = qml.device(backend, wires=n_qubits)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        diff_method = "adjoint" if backend == "lightning.qubit" else "parameter-shift"

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def qnode(inputs, weights):
            # 角度嵌入 (rotation='Y')
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            # 强纠缠层
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            # 测量 PauliZ(0) 期望值
            return qml.expval(qml.PauliZ(0))

        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_qubits)
        z = self.q_layer(x) # range [-1, 1]
        # 映射到 [0, 1] 概率: p = (1 + z) / 2
        p = (1.0 + z) / 2.0
        return p

# ================== 训练流程 ==================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--qubits", type=int, default=10)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--subset", type=int, default=0, help="使用前 N 条数据进行训练（0表示全量）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    logger.info("读取数据: %s", args.data)
    df = pd.read_csv(args.data)
    
    # 创建标签
    df["meta_label"] = create_meta_labels(df)
    
    # 选择特征 (同 XGBoost 版本)
    features = [
        "qnn_prediction", "atr", "adx", "rsi", "volume_shock", "volume_density",
        "spread", "tick_rate", "bid_ask_imbalance", "wick_ratio", "candle_size"
    ]
    available_features = [f for f in features if f in df.columns]
    
    # 数据采样（可选）
    if args.subset > 0:
        logger.info("应用采样: 仅使用前 %d 条数据", args.subset)
        df = df.iloc[:args.subset]
    
    if len(df) > 50000:
        logger.info("数据量巨大 (%d)，将进行进度追踪", len(df))
    X_raw = df[available_features].fillna(0).to_numpy(dtype=np.float64)
    y_raw = df["meta_label"].to_numpy(dtype=np.float64)

    # 时间切分
    split = int(len(X_raw) * (1 - args.val_ratio))
    X_train_raw, y_train_raw = X_raw[:split], y_raw[:split]
    X_val_raw, y_val_raw = X_raw[split:], y_raw[split:]

    # 特征转换
    transformer = QuantumFeatureTransformer(n_qubits=args.qubits, feature_cols=available_features)
    X_train = transformer.fit_transform(X_train_raw)
    X_val = transformer.transform(X_val_raw)

    # 2. 准备 Torch DataLoader
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_raw))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val_raw))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # 3. 初始化模型
    backend = "lightning.qubit"
    model = QuantumClassifier(n_qubits=args.qubits, n_layers=args.layers, backend=backend)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # 4. 训练循环
    best_val_auc = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for xb, yb in pbar:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        # 验证
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc="Eval"):
                pred = model(xb)
                all_preds.extend(pred.numpy())
                all_labels.extend(yb.numpy())
        
        auc = roc_auc_score(all_labels, all_preds)
        logger.info("Epoch %d: Loss=%.4f, Val AUC=%.4f", epoch, total_loss/len(train_loader), auc)
        
        if auc > best_val_auc:
            best_val_auc = auc
            torch.save(model.state_dict(), outdir / "quantum_meta_model.pt")
            logger.info("已保存最佳模型 (AUC=%.4f)", auc)

    # 保存元数据
    artifacts = {
        "feature_cols": available_features,
        "n_qubits": args.qubits,
        "n_layers": args.layers,
        "backend": backend,
        "best_auc": float(best_val_auc)
    }
    with (outdir / "meta_artifacts.json").open("w") as f:
        json.dump(artifacts, f, indent=2)
    with (outdir / "meta_transformer.pkl").open("wb") as f:
        pickle.dump(transformer.get_state(), f)

    logger.info("✅ 训练完成")

if __name__ == "__main__":
    main()
