#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线 Residual 标注脚本：批量推理训练数据，计算预测误差

功能：
1. 读取原始 CSV（含 target_next_close_change）
2. 使用训练好的 QuantumRegressor 对每行推理
3. 计算 residual = target_next_close_change - qnn_prediction
4. 输出新 CSV，附加 qnn_prediction 和 residual 列

用法：
    python add_residuals.py \
        --input ../QuantumNet_Training_Data.aligned.csv \
        --output ../QuantumNet_MetaLabel_Data.csv \
        --modeldir ../models
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# ================== 强制 float64 ==================
import torch
torch.set_default_dtype(torch.float64)

import torch.nn as nn
import pennylane as qml

# OMP 设置
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("KMP_BLOCKTIME", "0")

logger = logging.getLogger("quantum-engine.add_residuals")


# ================== 复用 infer 模块的类 ==================
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.decomposition import PCA

RSI_COLS = {"rsi"}
EMA_SPREAD_COLS = {"ema_spread"}
VOLUME_SHOCK_COLS = {"volume_shock"}


class QuantumFeatureTransformer:
    """方案 C 特征工程器：按附录 7.1 规范对不同特征定制预处理到量子友好区间。"""
    
    def __init__(self, n_qubits: int, feature_cols: List[str], seed: int = 42):
        self.n_qubits = n_qubits
        self.feature_cols = feature_cols
        self.seed = seed
        
        self._rsi_idx: List[int] = []
        self._ema_spread_idx: List[int] = []
        self._volume_shock_idx: List[int] = []
        self._general_idx: List[int] = []
        
        for i, col in enumerate(feature_cols):
            col_lower = col.lower()
            if col_lower in RSI_COLS or "rsi" in col_lower:
                self._rsi_idx.append(i)
            elif col_lower in EMA_SPREAD_COLS or "ema_spread" in col_lower:
                self._ema_spread_idx.append(i)
            elif col_lower in VOLUME_SHOCK_COLS or "volume_shock" in col_lower:
                self._volume_shock_idx.append(i)
            else:
                self._general_idx.append(i)
        
        self._ema_robust = RobustScaler()
        self._ema_minmax = MinMaxScaler(feature_range=(-1, 1))
        self._vol_std = StandardScaler()
        self._general_robust = RobustScaler()
        
        self._pca = PCA(n_components=n_qubits, random_state=seed)
        self._fitted = False
    
    def _transform_before_pca(self, X: np.ndarray) -> np.ndarray:
        """应用分类预处理（PCA 之前）。"""
        X = X.astype(np.float64)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        out = np.zeros((n_samples, n_features), dtype=np.float64)
        
        for i in self._rsi_idx:
            out[:, i] = np.clip(X[:, i], 0, 100) / 100.0 * np.pi
        
        if self._ema_spread_idx:
            ema_data = X[:, self._ema_spread_idx]
            ema_robust_out = self._ema_robust.transform(ema_data)
            ema_minmax_out = self._ema_minmax.transform(ema_robust_out)
            for j, i in enumerate(self._ema_spread_idx):
                out[:, i] = ema_minmax_out[:, j] * np.pi
        
        if self._volume_shock_idx:
            vol_data = X[:, self._volume_shock_idx]
            vol_log = np.sign(vol_data) * np.log1p(np.abs(vol_data))
            vol_std_out = self._vol_std.transform(vol_log)
            for j, i in enumerate(self._volume_shock_idx):
                out[:, i] = np.clip(vol_std_out[:, j], -3, 3)
        
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
        X_pca_clipped = np.clip(X_pca, -np.pi, np.pi)
        return X_pca_clipped.astype(np.float64)
    
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
    """目标变量归一化器：[-0.9, 0.9]"""
    
    def __init__(self):
        self._scaler = MinMaxScaler(feature_range=(-0.9, 0.9))
        self._fitted = False
    
    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """逆变换：将模型预测值还原为原始尺度。"""
        if not self._fitted:
            raise RuntimeError("TargetScaler 未 fit")
        y_scaled = np.asarray(y_scaled, dtype=np.float64).reshape(-1, 1)
        return self._scaler.inverse_transform(y_scaled).flatten()


class QuantumRegressor(nn.Module):
    """方案 C 量子回归器"""
    
    def __init__(self, n_qubits: int, n_layers: int, backend: str):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend

        dev = qml.device(backend, wires=n_qubits)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.q_layer(x)
        y = self.head(z)
        return y.squeeze(-1)


def _pick_backend(preferred: str) -> str:
    try:
        _ = qml.device(preferred, wires=2)
        return preferred
    except Exception:
        return "default.qubit"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量推理并计算 Residual")
    p.add_argument("--input", required=True, help="输入 CSV 文件路径")
    p.add_argument("--output", required=True, help="输出 CSV 文件路径")
    p.add_argument("--modeldir", default="./models", help="模型产物目录")
    p.add_argument("--batch-size", type=int, default=256, help="批量推理大小")
    return p.parse_args()


def main() -> int:
    _setup_logging()
    args = _parse_args()
    
    modeldir = Path(args.modeldir).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    
    # 1. 加载 artifacts
    artifacts_path = modeldir / "artifacts.json"
    if not artifacts_path.exists():
        logger.error("找不到 artifacts.json：%s", artifacts_path)
        return 1
    
    artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
    feature_cols = artifacts["feature_cols"]
    cfg = artifacts["config"]
    n_qubits = int(cfg["qubits"])
    n_layers = int(cfg["layers"])
    preferred_backend = artifacts.get("backend", "lightning.qubit")
    backend = _pick_backend(preferred_backend)
    scheme = artifacts.get("scheme", "legacy")
    
    logger.info("方案: %s, qubits=%d, layers=%d, backend=%s", scheme, n_qubits, n_layers, backend)
    
    # 2. 加载预处理器
    if scheme == "C":
        transformer_path = modeldir / "feature_transformer.pkl"
        with transformer_path.open("rb") as f:
            transformer_state = pickle.load(f)
        feature_transformer = QuantumFeatureTransformer.from_state(transformer_state)
        
        target_scaler_path = modeldir / "target_scaler.pkl"
        with target_scaler_path.open("rb") as f:
            target_scaler = pickle.load(f)
    else:
        logger.error("暂不支持非 C 方案")
        return 1
    
    # 3. 加载模型
    ckpt_path = modeldir / "quantum_regressor_best.pt"
    if not ckpt_path.exists():
        logger.error("找不到模型权重：%s", ckpt_path)
        return 1
    
    device = torch.device("cpu")
    model = QuantumRegressor(n_qubits=n_qubits, n_layers=n_layers, backend=backend).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    logger.info("模型加载完成")
    
    # 4. 读取数据
    logger.info("读取数据: %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("总行数: %d", len(df))
    
    # 检查必要列
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.error("缺少特征列: %s", missing_cols[:10])
        return 1
    
    target_col = "target_next_close_change"
    if target_col not in df.columns:
        logger.error("缺少目标列: %s", target_col)
        return 1
    
    # 5. 批量推理
    X_raw = df[feature_cols].values.astype(np.float64)
    y_actual = df[target_col].values.astype(np.float64)
    
    n_samples = len(df)
    batch_size = args.batch_size
    predictions = np.zeros(n_samples, dtype=np.float64)
    
    logger.info("开始批量推理 (batch_size=%d)...", batch_size)
    
    with torch.no_grad():
        for start in tqdm(range(0, n_samples, batch_size), desc="Inference"):
            end = min(start + batch_size, n_samples)
            X_batch = X_raw[start:end]
            
            # 特征变换
            X_q = feature_transformer.transform(X_batch)
            X_tensor = torch.from_numpy(X_q).to(torch.float64).to(device)
            
            # 推理
            y_scaled = model(X_tensor).cpu().numpy()
            
            # 逆变换还原原始尺度
            y_hat = target_scaler.inverse_transform(y_scaled)
            predictions[start:end] = y_hat
    
    # 6. 计算 residual
    residuals = y_actual - predictions
    
    # 7. 保存结果
    df["qnn_prediction"] = predictions
    df["residual"] = residuals
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info("✅ 完成！输出: %s", output_path)
    logger.info("统计: qnn_prediction mean=%.6f std=%.6f", predictions.mean(), predictions.std())
    logger.info("统计: residual mean=%.6f std=%.6f", residuals.mean(), residuals.std())
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
