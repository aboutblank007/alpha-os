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
    """目标变量缩放器（兼容训练脚本的序列化，对称版）"""
    def __init__(self):
        self.scale_ = 1.0
        self._fitted = False
    
    def inverse_transform(self, y_scaled):
        y_scaled = np.asarray(y_scaled, dtype=np.float64)
        return (y_scaled / 0.8) * self.scale_


def _setup_logging() -> None:
    """配置日志，同时输出到控制台和日志文件"""
    log_format = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    
    # 基础配置 (控制台)
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
    )
    
    # 文件配置
    log_dir = Path(__file__).parent.parent / "logs"
    if log_dir.exists():
        file_handler = logging.FileHandler(log_dir / "alpha_engine.log")
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        logger.info(f"✅ 日志文件已启用: {log_dir}/alpha_engine.log")


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
        
        # Alpha101 2.0: 特征滚动缓冲区 (用于 TS_Rank)
        self._feature_buffer = []
        self._buffer_max_size = 1440  # 1440 分钟/Tick
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
        
        class TargetScaler:
            def __init__(self):
                self.scale_ = 1.0
                self._fitted = True
            
            def inverse_transform(self, y_scaled):
                y_scaled = np.asarray(y_scaled, dtype=np.float64)
                return (y_scaled / 0.8) * self.scale_
        
        # 如果 state 是 dict，从字典恢复
        if isinstance(state, dict):
            obj = TargetScaler()
            obj.scale_ = state.get("scale_", 1.0)
            return obj
        # 如果 state 本身有 inverse_transform，直接使用
        elif hasattr(state, "inverse_transform"):
            return state
        # 旧版兼容：如果 state 是 MinMaxScaler
        elif hasattr(state, "_scaler"):
            # 注意：旧版 pickle 无法完美兼容到对称逻辑，这里仅做降级处理
            # 实际上由于我们重新训练了模型，这种情况不应发生
            return state
        else:
            # 尝试作为 scale_ 值使用
            obj = TargetScaler()
            if isinstance(state, (float, int)):
                 obj.scale_ = float(state)
            return obj
    
    def _restore_transformer(self, state: Dict[str, Any]):
        """从状态恢复 QuantumFeatureTransformer"""
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
        from sklearn.decomposition import PCA
        
        class QuantumFeatureTransformer:
            def __init__(self):
                pass
            
            def _apply_ts_rank_single(self, x_new: np.ndarray, buffer: List[np.ndarray]) -> np.ndarray:
                """计算最新特征在缓冲区中的百分位排名"""
                if not buffer:
                    return np.full_like(x_new, 0.5)
                
                # buffer 形状: List[np.ndarray(1, N)] -> vstack -> (n_ticks, N)
                buf_arr = np.vstack(buffer)
                # 计算比当前值小的比例
                ranks = np.zeros_like(x_new)
                for i in range(x_new.shape[1]):
                    # 避免全一致导致的问题 (x_new 形状是 1, N)
                    ranks[0, i] = np.mean(buf_arr[:, i] < x_new[0, i])
                return ranks

            def transform(self, X: np.ndarray) -> np.ndarray:
                # Alpha101 2.0: 此时 X 已经是 TS_Rank ([0, 1])
                # 注意：X 在这里是单行形状 (1, n_features)
                n_features = X.shape[1]
                out = np.zeros((1, n_features), dtype=np.float64)
                
                PHYSICAL_COLS = ["rsi", "wick_ratio", "wick"]
                
                for i in range(n_features):
                    col_name = self.feature_cols[i].lower()
                    val = X[0, i]
                    
                    if col_name in PHYSICAL_COLS or "rsi" in col_name or "wick" in col_name:
                        # [0, 1] -> [0, pi]
                        out[0, i] = val * np.pi
                    elif "dom_pressure" in col_name or "imbalance" in col_name or "spread" in col_name:
                        # [0, 1] -> [-pi, pi]
                        out[0, i] = (val - 0.5) * 2.0 * np.pi
                    else:
                        # 默认 [0, 1] -> [-pi/2, pi/2]
                        out[0, i] = (val - 0.5) * np.pi
                
                # PCA
                X_pca = self._pca.transform(out)
                return np.clip(X_pca, -np.pi, np.pi).astype(np.float64)
        
        obj = QuantumFeatureTransformer()
        obj.feature_cols = state["feature_cols"]
        obj._physical_idx = state.get("_physical_idx", state.get("_rsi_idx", []))
        obj._ema_ratio_idx = state.get("_ema_ratio_idx", state.get("_ema_spread_idx", []))
        obj._pressure_idx = state.get("_pressure_idx", [])
        obj._volume_shock_idx = state["_volume_shock_idx"]
        obj._general_idx = state["_general_idx"]
        obj._ema_robust = state["_ema_robust"]
        obj._ema_minmax = state["_ema_minmax"]
        obj._pressure_robust = state.get("_pressure_robust")
        obj._vol_std = state["_vol_std"]
        obj._general_robust = state["_general_robust"]
        obj._pca = state["_pca"]
        
        # 重新初始化 _close_idx
        obj._close_idx = -1
        if hasattr(obj, "feature_cols"):
            for i, col in enumerate(obj.feature_cols):
                if col.lower() == "close":
                    obj._close_idx = i
                    break
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
        mid_price = (tick.bid + tick.ask) / 2.0
        
        # 基础特征提取（处理别名）
        val_volume = float(tick.volume)
        val_dom = tick.dom_pressure if hasattr(tick, "dom_pressure") else 0.0
        # 如果 TickData 中没有 dom_pressure，尝试使用 dom_pressure_proxy
        if val_dom == 0.0 and hasattr(tick, "dom_pressure_proxy"):
            val_dom = tick.dom_pressure_proxy
            
        val_ema_fast = tick.ema_fast if hasattr(tick, "ema_fast") else 0.0
        val_ema_slow = tick.ema_slow if hasattr(tick, "ema_slow") else 0.0
        val_rsi = tick.rsi if hasattr(tick, "rsi") else 50.0  # 默认 50 中性
        
        # 衍生特征计算
        val_ema_spread = val_ema_fast - val_ema_slow
        
        # 特征映射表
        feature_map = {
            "open": mid_price,
            "high": mid_price,
            "low": mid_price,
            "close": mid_price,
            "tick_volume": val_volume,
            "volume": val_volume,            # 兼容旧名
            "ema_fast": val_ema_fast,
            "ema_slow": val_ema_slow,
            "ema_spread": val_ema_spread,
            "rsi": val_rsi,
            "bid": tick.bid,
            "ask": tick.ask,
            "wick_ratio": tick.wick_ratio,
            "volume_density": tick.vol_density,
            "vol_density": tick.vol_density, # 兼容旧名
            "volume_shock": tick.vol_shock,
            "vol_shock": tick.vol_shock,     # 兼容旧名
            "dom_pressure_proxy": val_dom,
            "dom_pressure": val_dom,         # 兼容旧名
            "spread": float(tick.spread) if tick.spread else 0,
            "tick_rate": float(tick.tick_rate) if tick.tick_rate else 0,
            "bid_ask_imbalance": tick.bid_ask_imbalance if tick.bid_ask_imbalance else 0,
            # 缺失的指标用 0 或中性值补全
            "atr": 0.0,
            "adx": 0.0,
            "wick_upper": 0.0,
            "wick_lower": 0.0,
            "candle_size": 0.0,
        }
        
        x_raw = np.array([
            feature_map.get(col, 0.0) for col in self.feature_cols
        ], dtype=np.float64).reshape(1, -1)
        
        # 2. Alpha101 2.0: 计算 TS_Rank
        # 更新缓冲区
        self._feature_buffer.append(x_raw.copy())
        if len(self._feature_buffer) > self._buffer_max_size:
            self._feature_buffer.pop(0)
            
        # 计算当前 x_raw 在 buffer 中的排名
        if self.feature_transformer:
            # 内部 transform 会处理 TS_Rank
            x_rank = self.feature_transformer._apply_ts_rank_single(x_raw, self._feature_buffer)
            x_q = self.feature_transformer.transform(x_rank)
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
