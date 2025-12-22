#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q-Link Alpha 引擎

量子推理服务：
1. 订阅 Market Stream (ZMQ PULL)
2. 加载量子回归模型 + 预处理器
3. 执行量子电路推理
4. 输出 AlphaSignal 到风控引擎

基于 M2 Pro 优化：
- 绑定 P-Cores #1-7
- Float64 全链路精度
- lightning.qubit 后端
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import signal
import sys
import time
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event
from typing import Optional, Dict, Any, List

import numpy as np

# ================== M2 Pro 多核优化 ==================
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OMP_PROC_BIND", "true")
os.environ.setdefault("KMP_BLOCKTIME", "0")

import torch
torch.set_default_dtype(torch.float64)

import zmq

from protocol import (
    Ports, MessageType, TickData, AlphaSignal, OrderSide,
    Heartbeat, HEARTBEAT_INTERVAL_MS, LATENCY_THRESHOLD_MS,
)

logger = logging.getLogger("qlink.alpha")


# ================== 兼容类定义（用于 pickle 加载） ==================

class TargetScaler:
    """目标变量缩放器（兼容训练脚本的序列化）"""
    def __init__(self):
        from sklearn.preprocessing import MinMaxScaler
        self._scaler = MinMaxScaler(feature_range=(-0.9, 0.9))
        self._fitted = False
    
    def inverse_transform(self, y_scaled):
        y_scaled = np.asarray(y_scaled, dtype=np.float64).reshape(-1, 1)
        return self._scaler.inverse_transform(y_scaled).flatten()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


# ================== 模型加载 ==================

class QuantumPredictor:
    """
    量子回归预测器
    
    封装模型加载、特征变换、推理逻辑
    """
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self.feature_transformer = None
        self.target_scaler = None
        self.feature_cols: List[str] = []
        self.n_qubits = 10
        self.n_layers = 3
        self.backend = "lightning.qubit"
        
        self._load_artifacts()
    
    def _load_artifacts(self) -> None:
        """加载模型产物"""
        import json
        import pennylane as qml
        import torch.nn as nn
        
        # 1. 加载配置
        artifacts_path = self.model_dir / "artifacts.json"
        if not artifacts_path.exists():
            raise FileNotFoundError(f"找不到 artifacts.json: {artifacts_path}")
        
        with artifacts_path.open("r", encoding="utf-8") as f:
            artifacts = json.load(f)
        
        self.feature_cols = artifacts["feature_cols"]
        config = artifacts.get("config", {})
        self.n_qubits = config.get("qubits", 10)
        self.n_layers = config.get("layers", 3)
        self.backend = artifacts.get("backend", "lightning.qubit")
        
        logger.info("配置: qubits=%d, layers=%d, backend=%s, features=%d",
                    self.n_qubits, self.n_layers, self.backend, len(self.feature_cols))
        
        # 2. 加载特征变换器
        transformer_path = self.model_dir / "feature_transformer.pkl"
        if transformer_path.exists():
            with transformer_path.open("rb") as f:
                state = pickle.load(f)
            self.feature_transformer = self._restore_transformer(state)
            logger.info("特征变换器加载完成")
        
        # 3. 加载目标缩放器
        scaler_path = self.model_dir / "target_scaler.pkl"
        if scaler_path.exists():
            with scaler_path.open("rb") as f:
                scaler_state = pickle.load(f)
            self.target_scaler = self._restore_target_scaler(scaler_state)
            logger.info("目标缩放器加载完成")
        
        # 4. 加载量子模型
        ckpt_path = self.model_dir / "quantum_regressor_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"找不到模型权重: {ckpt_path}")
        
        self.model = self._build_model()
        state = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state["model_state"])
        self.model.eval()
        logger.info("量子模型加载完成")
    
    def _restore_target_scaler(self, state):
        """从状态恢复 TargetScaler"""
        from sklearn.preprocessing import MinMaxScaler
        
        class TargetScaler:
            def __init__(self):
                self._scaler = MinMaxScaler(feature_range=(-0.9, 0.9))
                self._fitted = True
            
            def inverse_transform(self, y_scaled):
                y_scaled = np.asarray(y_scaled, dtype=np.float64).reshape(-1, 1)
                return self._scaler.inverse_transform(y_scaled).flatten()
        
        # 如果 state 是 dict，从字典恢复
        if isinstance(state, dict):
            obj = TargetScaler()
            obj._scaler = state.get("_scaler", state)
            return obj
        # 如果 state 本身有 inverse_transform，直接使用
        elif hasattr(state, "inverse_transform"):
            return state
        # 如果 state 是 MinMaxScaler
        elif hasattr(state, "_scaler"):
            obj = TargetScaler()
            obj._scaler = state._scaler
            return obj
        else:
            # 尝试作为 MinMaxScaler 使用
            obj = TargetScaler()
            obj._scaler = state
            return obj
    
    def _restore_transformer(self, state: Dict[str, Any]):
        """从状态恢复 QuantumFeatureTransformer"""
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
        from sklearn.decomposition import PCA
        
        class QuantumFeatureTransformer:
            def __init__(self):
                pass
            
            def transform(self, X: np.ndarray) -> np.ndarray:
                X = X.astype(np.float64)
                n_samples = X.shape[0]
                n_features = X.shape[1]
                out = np.zeros((n_samples, n_features), dtype=np.float64)
                
                # RSI
                for i in self._rsi_idx:
                    out[:, i] = np.clip(X[:, i], 0, 100) / 100.0 * np.pi
                
                # EMA Spread
                if self._ema_spread_idx:
                    ema_data = X[:, self._ema_spread_idx]
                    ema_robust_out = self._ema_robust.transform(ema_data)
                    ema_minmax_out = self._ema_minmax.transform(ema_robust_out)
                    for j, i in enumerate(self._ema_spread_idx):
                        out[:, i] = ema_minmax_out[:, j] * np.pi
                
                # Volume Shock
                if self._volume_shock_idx:
                    vol_data = X[:, self._volume_shock_idx]
                    vol_log = np.sign(vol_data) * np.log1p(np.abs(vol_data))
                    vol_std_out = self._vol_std.transform(vol_log)
                    for j, i in enumerate(self._volume_shock_idx):
                        out[:, i] = np.clip(vol_std_out[:, j], -3, 3)
                
                # General
                if self._general_idx:
                    general_data = X[:, self._general_idx]
                    general_robust_out = self._general_robust.transform(general_data)
                    for j, i in enumerate(self._general_idx):
                        clipped = np.clip(general_robust_out[:, j], -3, 3)
                        out[:, i] = clipped * (np.pi / 3.0)
                
                # PCA
                X_pca = self._pca.transform(out)
                return np.clip(X_pca, -np.pi, np.pi).astype(np.float64)
        
        obj = QuantumFeatureTransformer()
        obj._rsi_idx = state["_rsi_idx"]
        obj._ema_spread_idx = state["_ema_spread_idx"]
        obj._volume_shock_idx = state["_volume_shock_idx"]
        obj._general_idx = state["_general_idx"]
        obj._ema_robust = state["_ema_robust"]
        obj._ema_minmax = state["_ema_minmax"]
        obj._vol_std = state["_vol_std"]
        obj._general_robust = state["_general_robust"]
        obj._pca = state["_pca"]
        return obj
    
    def _build_model(self):
        """构建量子回归模型"""
        import pennylane as qml
        import torch.nn as nn
        
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        backend = self.backend
        
        dev = qml.device(backend, wires=n_qubits)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        diff_method = "adjoint" if backend == "lightning.qubit" else "parameter-shift"
        
        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]
        
        class QuantumRegressor(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
                self.head = nn.Linear(n_qubits, 1)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                z = self.q_layer(x)
                y = self.head(z)
                return y.squeeze(-1)
        
        return QuantumRegressor()
    
    def predict(self, tick: TickData) -> AlphaSignal:
        """
        执行量子推理
        
        Args:
            tick: Tick 行情数据
        
        Returns:
            AlphaSignal: 预测信号
        """
        # 1. 构建特征向量
        feature_map = {
            "bid": tick.bid,
            "ask": tick.ask,
            "volume": float(tick.volume),
            "wick_ratio": tick.wick_ratio,
            "vol_density": tick.vol_density,
            "volume_shock": tick.vol_shock,
            "spread": float(tick.spread) if tick.spread else 0,
            "tick_rate": float(tick.tick_rate) if tick.tick_rate else 0,
            "bid_ask_imbalance": tick.bid_ask_imbalance if tick.bid_ask_imbalance else 0,
        }
        
        x_raw = np.array([
            feature_map.get(col, 0.0) for col in self.feature_cols
        ], dtype=np.float64).reshape(1, -1)
        
        # 2. 特征变换
        if self.feature_transformer:
            x_q = self.feature_transformer.transform(x_raw)
        else:
            x_q = x_raw
        
        # 3. 推理
        with torch.no_grad():
            x_tensor = torch.from_numpy(x_q).to(torch.float64)
            y_scaled = self.model(x_tensor).cpu().numpy().reshape(-1)[0]
        
        # 4. 逆变换
        if self.target_scaler:
            y_hat = self.target_scaler.inverse_transform(np.array([y_scaled]))[0]
        else:
            y_hat = y_scaled
        
        # 5. 生成信号
        direction = OrderSide.BUY if y_hat > 0 else OrderSide.SELL
        confidence = min(abs(y_hat) / 2.0, 1.0)  # 粗略置信度
        
        return AlphaSignal(
            timestamp=tick.timestamp,
            symbol=tick.symbol,
            prediction=float(y_hat),
            direction=direction,
            confidence=confidence,
            tick_data=tick,
            raw_expectation=float(y_scaled),  # 原始量子测量期望值 [-1, 1]
        )


# ================== 主服务 ==================

class AlphaEngine:
    """
    Alpha 引擎主服务
    
    职责：
    1. 订阅 Market Stream
    2. 执行量子推理
    3. 输出信号到队列（供 Risk Engine 消费）
    """
    
    def __init__(
        self,
        model_dir: Path,
        signal_queue: Queue,
        bind_addr: str = "tcp://127.0.0.1",
        signal_port: int = 5560,
        market_port: int = 5557,
    ):
        self.model_dir = model_dir
        self.signal_queue = signal_queue
        self.bind_addr = bind_addr
        self.signal_port = signal_port
        self.market_port = market_port
        
        self.predictor: Optional[QuantumPredictor] = None
        self.context: Optional[zmq.Context] = None
        self.market_socket: Optional[zmq.Socket] = None
        
        self._stop_event = Event()
        self._last_tick_time = 0
    
    def start(self) -> None:
        """启动服务"""
        logger.info("Alpha Engine 启动中...")
        
        # 1. 加载模型
        self.predictor = QuantumPredictor(self.model_dir)
        
        # 2. 初始化 ZMQ
        self.context = zmq.Context()
        
        # Market Stream (PULL) - 连接到 EA (EA bind)
        self.market_socket = self.context.socket(zmq.PULL)
        self.market_socket.setsockopt(zmq.RCVHWM, 1000)
        self.market_socket.setsockopt(zmq.LINGER, 0)
        market_addr = f"{self.bind_addr}:{self.market_port}"
        self.market_socket.connect(market_addr)
        logger.info("已连接 Market Stream: %s", market_addr)
        
        # Alpha-to-Risk (PUSH) - 内部管道，绑定本地
        self.signal_socket = self.context.socket(zmq.PUSH)
        self.signal_socket.setsockopt(zmq.SNDHWM, 100)
        self.signal_socket.setsockopt(zmq.LINGER, 0)
        signal_addr = f"tcp://127.0.0.1:{self.signal_port}"
        self.signal_socket.bind(signal_addr)
        logger.info("已绑定 Alpha-to-Risk: %s", signal_addr)
        
        # 3. 主循环
        self._run_loop()
    
    def _run_loop(self) -> None:
        """主循环：接收 Tick -> 推理 -> 输出信号"""
        poller = zmq.Poller()
        poller.register(self.market_socket, zmq.POLLIN)
        
        while not self._stop_event.is_set():
            try:
                socks = dict(poller.poll(timeout=100))
                
                if self.market_socket in socks:
                    message = self.market_socket.recv_string(zmq.NOBLOCK)
                    self._handle_message(message)
                    
            except zmq.Again:
                continue
            except Exception as e:
                logger.error("主循环异常: %s", e)
                time.sleep(0.1)
    
    def _handle_message(self, message: str) -> None:
        """处理收到的消息"""
        try:
            if message.startswith(MessageType.TICK.value):
                tick = TickData.from_csv(message)
                self._process_tick(tick)
            elif message.startswith(MessageType.HEARTBEAT.value):
                hb = Heartbeat.from_csv(message)
                logger.debug("收到心跳: %s", hb.source)
            else:
                logger.warning("未知消息类型: %s", message[:20])
        except Exception as e:
            logger.error("消息处理失败: %s - %s", e, message[:50])
    
    def _process_tick(self, tick: TickData) -> None:
        """处理 Tick 数据"""
        now = int(time.time() * 1000)
        latency = now - tick.timestamp
        
        # 延迟检查
        if latency > LATENCY_THRESHOLD_MS:
            logger.warning("Tick 延迟过高: %d ms", latency)
        
        # 推理
        try:
            signal = self.predictor.predict(tick)
            # 发送信号到 Risk Engine (通过 ZMQ)
            signal_json = signal.to_json()
            self.signal_socket.send_string(signal_json, zmq.NOBLOCK)
            logger.info(
                "信号: %s %s pred=%.4f conf=%.2f",
                signal.symbol, signal.direction.value,
                signal.prediction, signal.confidence
            )
        except Exception as e:
            logger.error("推理失败: %s", e)
        
        self._last_tick_time = now
    
    def stop(self) -> None:
        """停止服务"""
        logger.info("Alpha Engine 停止中...")
        self._stop_event.set()
        
        if self.signal_socket:
            self.signal_socket.close()
        if self.market_socket:
            self.market_socket.close()
        if self.context:
            self.context.term()


# ================== 入口 ==================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q-Link Alpha Engine")
    p.add_argument(
        "--model-dir",
        default="./models",
        help="模型目录路径",
    )
    p.add_argument(
        "--bind",
        default="tcp://127.0.0.1",
        help="ZMQ 绑定地址",
    )
    p.add_argument(
        "--signal-port",
        type=int,
        default=5560,
        help="Alpha-to-Risk 信号端口（默认 5560）",
    )
    p.add_argument(
        "--market-port",
        type=int,
        default=5557,
        help="Market Stream 端口（默认 5557）",
    )
    return p.parse_args()


def main() -> int:
    _setup_logging()
    args = _parse_args()
    
    model_dir = Path(args.model_dir).expanduser().resolve()
    signal_queue = Queue(maxsize=100)
    
    engine = AlphaEngine(
        model_dir=model_dir,
        signal_queue=signal_queue,
        bind_addr=args.bind,
        signal_port=args.signal_port,
        market_port=args.market_port,
    )
    
    # 信号处理
    def signal_handler(sig, frame):
        engine.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        engine.start()
    except KeyboardInterrupt:
        engine.stop()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
