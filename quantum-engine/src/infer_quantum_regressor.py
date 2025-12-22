#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案 C 量子回归推理：输入一行特征（JSON），输出 y_hat = 预测的 target_next_close_change

核心优化：
- float64 精度
- QuantumFeatureTransformer 分特征定制预处理
- TargetScaler 逆变换还原原始尺度
- OMP 线程亲和性

注意：本脚本只做"模型推理"，不负责交易动作/风控决策。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ================== 方案 C 核心：强制 float64 ==================
import torch
torch.set_default_dtype(torch.float64)

import torch.nn as nn
import pennylane as qml

# 设置 OMP 线程亲和性（M2 Pro 优化）
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("KMP_BLOCKTIME", "0")

logger = logging.getLogger("quantum-engine.infer")


# ================== 从训练脚本复制的 QuantumFeatureTransformer ==================
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

RSI_COLS = {"rsi"}
EMA_SPREAD_COLS = {"ema_spread"}
VOLUME_SHOCK_COLS = {"volume_shock"}


class QuantumFeatureTransformer:
    """
    方案 C 特征工程器：按附录 7.1 规范对不同特征定制预处理到量子友好区间。
    """
    
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
        
        from sklearn.decomposition import PCA
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


# ================== 量子模型 ==================

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--modeldir", required=True, help="训练产物目录（quantum-engine/models）")
    p.add_argument("--row-json", required=True, help="一行特征 JSON（键=列名，值=数字）")
    return p.parse_args()


def _pick_backend(preferred: str) -> str:
    try:
        _ = qml.device(preferred, wires=2)
        return preferred
    except Exception:
        return "default.qubit"


class QuantumRegressor(nn.Module):
    """方案 C 量子回归器"""
    
    def __init__(self, n_qubits: int, n_layers: int, backend: str):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend

        dev = qml.device(backend, wires=n_qubits)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        # 推理时使用 parameter-shift（无需 adjoint 的依赖）
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


def _build_feature_vector(feature_cols: List[str], row: Dict[str, float]) -> np.ndarray:
    """构建特征向量（float64）。"""
    vals = []
    missing = []
    for c in feature_cols:
        if c not in row:
            missing.append(c)
            vals.append(0.0)
        else:
            vals.append(float(row[c]))
    if missing:
        logger.warning("输入缺少 %d 个特征，已按 0 补齐（前 10 个）：%s", len(missing), missing[:10])
    return np.asarray(vals, dtype=np.float64).reshape(1, -1)


def main() -> int:
    _setup_logging()
    args = _parse_args()

    modeldir = Path(args.modeldir).expanduser().resolve()
    artifacts_path = modeldir / "artifacts.json"
    if not artifacts_path.exists():
        raise FileNotFoundError(f"找不到 artifacts.json：{artifacts_path}")

    artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
    feature_cols = artifacts["feature_cols"]
    cfg = artifacts["config"]
    n_qubits = int(cfg["qubits"])
    n_layers = int(cfg["layers"])
    preferred_backend = artifacts.get("backend", "lightning.qubit")
    backend = _pick_backend(preferred_backend)
    scheme = artifacts.get("scheme", "legacy")

    # 方案 C：加载 QuantumFeatureTransformer
    if scheme == "C":
        transformer_path = modeldir / "feature_transformer.pkl"
        if not transformer_path.exists():
            raise FileNotFoundError(f"找不到 feature_transformer.pkl：{transformer_path}")
        with transformer_path.open("rb") as f:
            transformer_state = pickle.load(f)
        feature_transformer = QuantumFeatureTransformer.from_state(transformer_state)
        
        target_scaler_path = modeldir / "target_scaler.pkl"
        if not target_scaler_path.exists():
            raise FileNotFoundError(f"找不到 target_scaler.pkl：{target_scaler_path}")
        with target_scaler_path.open("rb") as f:
            target_scaler = pickle.load(f)
    else:
        # 兼容旧版（StandardScaler + PCA）
        scaler = pickle.loads((modeldir / "scaler.pkl").read_bytes())
        pca = pickle.loads((modeldir / "pca.pkl").read_bytes())
        feature_transformer = None
        target_scaler = None

    ckpt_path = modeldir / "quantum_regressor_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到模型权重：{ckpt_path}")

    row = json.loads(args.row_json)
    x_raw = _build_feature_vector(feature_cols, row)

    # 特征变换
    if scheme == "C":
        x_q = feature_transformer.transform(x_raw)
    else:
        x_std = scaler.transform(x_raw)
        x_q = pca.transform(x_std).astype(np.float64)

    # 方案 C：强制 CPU + float64
    device = torch.device("cpu")
    model = QuantumRegressor(n_qubits=n_qubits, n_layers=n_layers, backend=backend).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    with torch.no_grad():
        y_scaled = model(torch.from_numpy(x_q).to(torch.float64).to(device)).cpu().numpy().reshape(-1)[0]

    # 方案 C：逆变换还原原始尺度
    if scheme == "C" and target_scaler is not None:
        y_hat = target_scaler.inverse_transform(np.array([y_scaled]))[0]
    else:
        y_hat = y_scaled

    # 输出预测值（不输出 BUY/SELL/WAIT）
    logger.info("y_hat=%.6f (原始尺度: 美元)", float(y_hat))
    print(json.dumps({"y_hat": float(y_hat)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
