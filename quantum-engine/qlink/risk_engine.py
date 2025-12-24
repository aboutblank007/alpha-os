#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q-Link 风控引擎

三道防线风控服务：
1. 第一道防线：Meta-Labeling（PennyLane 量子电路）
2. 第二道防线：波动率目标 + 凯利公式
3. 第三道防线：L-VaR 流动性风控

核心优化：
- 量子推理：基于 lightning.qubit 加速
- 精度保障：强制使用 Float64
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event
from typing import Optional, Dict, Any

import numpy as np

import zmq

import torch
import pennylane as qml
import pickle

from protocol import (
    Ports, MessageType, OrderCommand, OrderAction, OrderSide, OrderType,
    AlphaSignal, RiskDecision, AccountState, Heartbeat,
    META_THRESHOLD, TARGET_VOLATILITY, MAX_POSITION_SIZE, MIN_POSITION_SIZE,
    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
    CONFIDENCE_DECAY_RATIO, OOD_THRESHOLD_SIGMA, ATR_K_HIGH_CONFIDENCE,
    ATR_K_LOW_CONFIDENCE, CONFIDENCE_HIGH_THRESHOLD, MAX_HOLDING_BARS,
    MAGIC_NUMBER, HEARTBEAT_INTERVAL_MS, HEARTBEAT_TIMEOUT_MS, TickData,
)

logger = logging.getLogger("qlink.risk")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


# ================== ATR 计算器 ==================

class ATRCalculator:
    """
    动态 ATR 计算器 (指数加权移动平均)
    
    在 Python 端计算 ATR，避免修改 MT5 EA
    """
    
    def __init__(self, period: int = 14, alpha: float = None):
        self.period = period
        self.alpha = alpha if alpha else 2.0 / (period + 1)
        
        # 每个品种的价格历史和当前 ATR
        self.high_history: Dict[str, list] = {}
        self.low_history: Dict[str, list] = {}
        self.close_history: Dict[str, list] = {}
        self.atr_values: Dict[str, float] = {}
    
    def update(self, symbol: str, high: float, low: float, close: float) -> float:
        """
        更新 ATR 值
        
        Args:
            symbol: 品种代码
            high: 最高价 (可用 ask 近似)
            low: 最低价 (可用 bid 近似)
            close: 收盘价 (可用 mid_price 近似)
            
        Returns:
            当前 ATR 值
        """
        # 初始化历史
        if symbol not in self.high_history:
            self.high_history[symbol] = []
            self.low_history[symbol] = []
            self.close_history[symbol] = []
            self.atr_values[symbol] = 0.0
        
        # 计算 True Range
        if len(self.close_history[symbol]) > 0:
            prev_close = self.close_history[symbol][-1]
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
        else:
            tr = high - low
        
        # 更新历史
        self.high_history[symbol].append(high)
        self.low_history[symbol].append(low)
        self.close_history[symbol].append(close)
        
        # 保持窗口大小
        if len(self.high_history[symbol]) > self.period * 2:
            self.high_history[symbol] = self.high_history[symbol][-self.period:]
            self.low_history[symbol] = self.low_history[symbol][-self.period:]
            self.close_history[symbol] = self.close_history[symbol][-self.period:]
        
        # 计算 EWMA ATR
        if self.atr_values[symbol] == 0:
            # 初始化：使用第一个 TR
            self.atr_values[symbol] = tr
        else:
            # 指数加权移动平均
            self.atr_values[symbol] = self.alpha * tr + (1 - self.alpha) * self.atr_values[symbol]
        
        return self.atr_values[symbol]
    
    def get(self, symbol: str) -> float:
        """获取当前 ATR 值"""
        return self.atr_values.get(symbol, 1.0)


# ================== Meta-Labeling 模型 (次级分类器) ==================

class XGBoostMetaLabeler:
    """
    XGBoost Meta-Labeling 模型
    
    作为次级分类器，预测 Alpha 信号在当前市场微观结构下的获胜概率
    输入特征: ATR, 成交量密度, 量子模型预测值, 预测置信度, DOM 压力
    """
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self.threshold = META_THRESHOLD
        
        try:
            import xgboost as xgb
            self._has_xgb = True
        except ImportError:
            self._has_xgb = False
            logger.warning("未检测到 xgboost 库, 元标记将降级回中性预测")
            
        self._load_artifacts()
    
    def _load_artifacts(self) -> None:
        """加载训练好的 XGBoost 模型"""
        if not self._has_xgb:
            return
            
        model_path = self.model_dir / "meta_labeling_xgb.json"
        if not model_path.exists():
            logger.warning("XGBoost Meta-Labeling 模型不存在: %s", model_path)
            return
            
        try:
            import xgboost as xgb
            self.model = xgb.Booster()
            self.model.load_model(str(model_path))
            logger.info("✅ XGBoost Meta-Labeling 模型加载完成")
        except Exception as e:
            logger.error("加载 XGBoost 失败: %s", e)
            self.model = None

    def predict(self, signal: AlphaSignal, market_state: Dict[str, float]) -> float:
        """预测信号可靠概率"""
        if not self.model or not self._has_xgb:
            # 降级方案：如果没有模型，使用 Alpha 信号自身的置信度
            return float(signal.confidence) if signal.confidence else 0.5
        
        try:
            import xgboost as xgb
            # 1. 提取核心特征 (对齐研究报告)
            # 特征向量: [atr_ratio, vol_density, pred_abs, confidence, dom_pressure]
            atr_ratio = market_state.get("atr", 0.0) / (signal.tick_data.mid_price + 1e-9)
            vol_density = signal.tick_data.vol_density
            pred_abs = abs(signal.prediction)
            confidence = signal.confidence
            dom_pressure = market_state.get("dom_pressure", 0.0)
            
            features = np.array([[atr_ratio, vol_density, pred_abs, confidence, dom_pressure]])
            dmatrix = xgb.DMatrix(features)
            
            # 2. 预测
            prob = self.model.predict(dmatrix)[0]
            return float(prob)
                
        except Exception as e:
            logger.error("XGBoost 推理失败: %s", e)
            return 0.5


class QuantumMetaLabeler:
    """
    量子 Meta-Labeling 模型
    
    使用 PennyLane 变分量子电路 (VQC) 预测 Alpha 信号是否可靠
    """
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.device = torch.device("cpu")
        self.model = None
        self.transformer = None
        self.features: list = []
        self.threshold = META_THRESHOLD
        
        # 强制设置 float64
        torch.set_default_dtype(torch.float64)
        
        self._load_artifacts()
    
    def _load_artifacts(self) -> None:
        """加载量子模型与预处理器"""
        model_path = self.model_dir / "quantum_meta_model.pt"
        meta_path = self.model_dir / "meta_artifacts.json"
        transformer_path = self.model_dir / "meta_transformer.pkl"
        
        if not (model_path.exists() and meta_path.exists() and transformer_path.exists()):
            logger.warning("量子 Meta-Labeling 产物不全, 将使用中性预测: %s", self.model_dir)
            return
            
        try:
            # 1. 加载元信息
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            self.features = meta.get("feature_cols", [])
            self.n_qubits = meta.get("n_qubits", 10)
            self.n_layers = meta.get("n_layers", 3)
            self.backend = meta.get("backend", "lightning.qubit")
            
            # 2. 加载预处理器 (Transformer)
            with transformer_path.open("rb") as f:
                transformer_state = pickle.load(f)
            self.transformer = self._restore_transformer(transformer_state)
            
            # 3. 初始化电路并加载权重
            dev = qml.device(self.backend, wires=self.n_qubits)
            weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
            
            @qml.qnode(dev, interface="torch", diff_method="adjoint")
            def qnode(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='Y')
                qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
                return qml.expval(qml.PauliZ(0))
            
            self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 兼容处理：如果 checkpoint 包含 q_layer. 前缀，则移除
            filtered_state = {k.replace("q_layer.", ""): v for k, v in checkpoint.items()}
            self.q_layer.load_state_dict(filtered_state)
            self.q_layer.eval()
            
            logger.info("量子 Meta-Labeling 模型加载完成: qubits=%d, layers=%d", 
                        self.n_qubits, self.n_layers)
                        
        except Exception as e:
            logger.error("加载量子 Meta-Labeling 失败: %s", e)
            self.q_layer = None

    def _restore_transformer(self, state: Dict[str, Any]):
        """从状态恢复 Transformer (避免依赖外部脚本)"""
        class QuantumFeatureTransformer:
            def __init__(self, state):
                self._scaler = state["_scaler"]
                self._pca = state["_pca"]
            def transform(self, X: np.ndarray) -> np.ndarray:
                X_scaled = self._scaler.transform(X)
                X_pca = self._pca.transform(X_scaled)
                return np.clip(X_pca, -np.pi, np.pi)
        return QuantumFeatureTransformer(state)

    def predict(self, signal: AlphaSignal, market_state: Dict[str, float]) -> float:
        """预测信号可靠概率"""
        if self.q_layer is None or self.transformer is None:
            return 0.5
        
        # 1. 构建原始特征映射
        tick = signal.tick_data
        feature_map = {
            "qnn_prediction": signal.prediction,
            "atr": market_state.get("atr", 0),
            "adx": market_state.get("adx", 0),
            "rsi": market_state.get("rsi", 50),
            "volume_shock": tick.vol_shock if hasattr(tick, 'vol_shock') else 0.0,
            "volume_density": tick.vol_density,
            "spread": float(tick.spread) if tick.spread else 0,
            "tick_rate": float(tick.tick_rate) if tick.tick_rate else 0,
            "bid_ask_imbalance": tick.bid_ask_imbalance if tick.bid_ask_imbalance else 0,
            "wick_ratio": tick.wick_ratio,
            "candle_size": market_state.get("candle_size", 0),
            "dom_pressure_proxy": market_state.get("dom_pressure", 0),
        }
        
        # 2. 转换为 numpy 并应用 Transformer
        x_raw = np.array([
            feature_map.get(f, 0.0) for f in self.features
        ], dtype=np.float64).reshape(1, -1)
        
        try:
            x_quantum = self.transformer.transform(x_raw)
            x_tensor = torch.from_numpy(x_quantum).to(torch.float64)
            
            # 3. 量子推理
            with torch.no_grad():
                z = self.q_layer(x_tensor)
                # 概率映射: p = (1 + <Z>) / 2
                prob = (1.0 + z.item()) / 2.0
                return float(prob)
                
        except Exception as e:
            logger.error("量子 Meta-Labeling 推理失败: %s", e)
            return 0.5


# ================== 仓位计算 ==================

class PositionSizer:
    """
    仓位计算器
    
    基于波动率目标 + 凯利公式
    """
    
    def __init__(
        self,
        target_vol: float = TARGET_VOLATILITY,
        max_size: float = MAX_POSITION_SIZE,
        min_size: float = MIN_POSITION_SIZE,
    ):
        self.target_vol = target_vol
        self.max_size = max_size
        self.min_size = min_size
    
    def calculate(
        self,
        meta_prob: float,
        current_vol: float,
        equity: float,
    ) -> tuple[float, float, float]:
        """
        计算仓位大小 (波动率目标仓位)
        
        公式: Position = Capital × σ_target / σ_t
        参考: docs/交易风控离场策略研究.md 3.2节
        
        Args:
            meta_prob: Meta-Labeling 概率 (用于过滤，不参与仓位计算)
            current_vol: 当前波动率（ATR 占价格比例）
            equity: 账户净值
        
        Returns:
            (position_size, confidence_weight, vol_scalar)
        """
        # 1. 波动率缩放: vol_scalar = σ_target / σ_t
        if current_vol > 0:
            vol_scalar = self.target_vol / current_vol
        else:
            vol_scalar = 1.0
        vol_scalar = np.clip(vol_scalar, 0.1, 3.0)
        
        # 2. 置信度权重 (替代凯利公式，仅用于微调)
        # 高置信度时满仓，低置信度时减仓
        confidence_weight = min(meta_prob, 1.0)
        
        # 3. 计算最终仓位: Position = base × vol_scalar × confidence_weight
        base_size = 0.1  # 基础手数
        position_size = base_size * vol_scalar * confidence_weight
        
        # 4. 裁剪
        position_size = np.clip(position_size, 0, self.max_size)
        if position_size < self.min_size:
            position_size = 0  # 低于最小仓位不交易
        
        return position_size, confidence_weight, vol_scalar


# ================== L-VaR 流动性风控 ==================

class LiquidityRiskController:
    """
    流动性风险控制器
    
    基于成交量密度计算隐含滑点成本
    """
    
    def __init__(self, base_cost: float = 0.1):
        self.base_cost = base_cost  # 基础交易成本（美元），从 0.5 降至 0.1 以适配 HFT
    
    def calculate_lvar(
        self,
        volume_density: float,
        position_size: float,
        spread: float,
    ) -> float:
        """
        计算流动性 VaR（隐含滑点成本）
        
        Args:
            volume_density: 成交量密度
            position_size: 计划仓位
            spread: 点差
        
        Returns:
            lvar: 隐含成本（美元）
        """
        # 薄盘惩罚：密度越低，滑点越大
        if volume_density > 0:
            density_factor = 1.0 / np.sqrt(volume_density)
        else:
            density_factor = 10.0  # 极端情况
        
        density_factor = np.clip(density_factor, 0.5, 10.0)
        
        # 仓位惩罚：仓位越大，市场冲击越大
        # 对于 XAUUSD，0.1-1.0 手属于极小单，系数从 0.1 降至 0.01
        size_penalty = position_size * 100 * 0.01
        
        # 点差成本
        spread_penalty = spread * density_factor
        
        # 总成本
        lvar = self.base_cost + spread_penalty + size_penalty
        
        return lvar, (self.base_cost, spread_penalty, size_penalty)


# ================== 风控引擎主服务 ==================

class RiskEngine:
    """
    风控引擎主服务
    
    职责：
    1. 接收 Alpha 信号
    2. 三道防线审查
    3. 发送交易指令到 MT5
    """
    
    def __init__(
        self,
        model_dir: Path,
        signal_queue: Queue,
        bind_addr: str = "tcp://127.0.0.1",
        signal_port: int = 5560,
        command_port: int = 5558,
    ):
        self.model_dir = model_dir
        self.signal_queue = signal_queue
        self.bind_addr = bind_addr
        self.signal_port = signal_port
        self.command_port = command_port
        
        # 组件
        self.meta_labeler: Optional[XGBoostMetaLabeler] = None
        self.quantum_meta: Optional[QuantumMetaLabeler] = None # 保留量子版本作为备选
        self.position_sizer: Optional[PositionSizer] = None
        self.lvar_controller: Optional[LiquidityRiskController] = None
        self.atr_calculator: Optional[ATRCalculator] = None
        
        # ZMQ
        self.context: Optional[zmq.Context] = None
        self.command_socket: Optional[zmq.Socket] = None
        self.state_socket: Optional[zmq.Socket] = None
        
        # 状态
        self.account_state: Optional[AccountState] = None
        self._stop_event = Event()
        self._last_state_sync = 0
        
        # 持仓追踪 (增强版)
        # {symbol: {"side", "entry_price", "entry_confidence", "entry_time", "current_sl", "lots", "bars_held"}}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self._last_entry_time: Dict[str, float] = {}  # 上次开仓时间（冷却用）
    
    def start(self) -> None:
        """启动服务"""
        logger.info("Risk Engine 启动中...")
        
        # 1. 初始化风控组件
        # 优先使用 XGBoost 元标记
        self.meta_labeler = XGBoostMetaLabeler(self.model_dir)
        self.quantum_meta = QuantumMetaLabeler(self.model_dir)
        self.position_sizer = PositionSizer()
        self.lvar_controller = LiquidityRiskController()
        self.atr_calculator = ATRCalculator(period=14)
        
        # 2. 初始化 ZMQ
        self.context = zmq.Context()
        
        # Alpha-to-Risk (PULL) - 内部管道，连接本地
        self.signal_socket = self.context.socket(zmq.PULL)
        self.signal_socket.setsockopt(zmq.RCVHWM, 100)
        self.signal_socket.setsockopt(zmq.LINGER, 0)
        self.signal_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms 超时
        signal_addr = f"tcp://127.0.0.1:{self.signal_port}"
        self.signal_socket.connect(signal_addr)
        logger.info("已连接 Alpha-to-Risk: %s", signal_addr)
        
        # Command Bus (PUSH) - Python bind，EA connect 过来
        self.command_socket = self.context.socket(zmq.PUSH)
        self.command_socket.setsockopt(zmq.SNDHWM, 1000)  # 扩容至 1000 以缓解高频信号拥堵
        self.command_socket.setsockopt(zmq.LINGER, 0)
        cmd_addr = f"tcp://0.0.0.0:{self.command_port}"
        self.command_socket.bind(cmd_addr)
        logger.info("Command Bus 已绑定: %s (EA 连接到 Mac IP:%d)", cmd_addr, self.command_port)
        
        # State Sync (REQ) - 连接到远程 MT5
        self.state_socket = self.context.socket(zmq.REQ)
        self.state_socket.setsockopt(zmq.LINGER, 0)
        self.state_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1秒超时
        state_addr = f"{self.bind_addr}:{Ports.STATE_SYNC}"
        self.state_socket.connect(state_addr)
        logger.info("State Sync 已连接: %s", state_addr)
        
        # 3. 主循环
        self._run_loop()
    
    def _run_loop(self) -> None:
        """主循环"""
        while not self._stop_event.is_set():
            try:
                # 从 ZMQ 接收信号
                try:
                    msg = self.signal_socket.recv_string()
                    signal = AlphaSignal.from_json(msg)
                    decision = self._process_signal(signal)
                    if decision:
                        if decision.action == "EXIT":
                            # 离场：只发送平仓指令
                            self._send_order(decision.close_order)
                        elif decision.action == "BET":
                            # 开仓：先发送平仓指令（如有），再发送开仓指令
                            if hasattr(decision, 'close_order') and decision.close_order:
                                self._send_order(decision.close_order)
                            self._send_order(decision.order)
                            
                            # 记录开仓时间，用于静默期
                            if decision.order.action == OrderAction.OPEN:
                                self._last_entry_time[decision.symbol] = time.time()
                except zmq.Again:
                    pass  # 超时，继续
                
                # 定期同步账户状态
                now = int(time.time() * 1000)
                if now - self._last_state_sync > 5000:
                    self._sync_account_state()
                    self._last_state_sync = now
                    
            except Exception as e:
                logger.error("主循环异常: %s", e)
                time.sleep(0.1)
    
    def _process_signal(self, signal: AlphaSignal) -> Optional[RiskDecision]:
        """
        处理 Alpha 信号，执行三道防线
        """
        symbol = signal.symbol
        tick = signal.tick_data
        
        # 动态更新 ATR（使用 bid/ask 近似 high/low/close）
        current_atr = self.atr_calculator.update(
            symbol=symbol,
            high=tick.ask,
            low=tick.bid,
            close=tick.mid_price,
        )
        
        # 市场状态（包含动态 ATR 和 OOD 检测所需的 volume_shock）
        market_state = {
            "atr": current_atr,  # 动态计算的 ATR
            "adx": 20.0,
            "rsi": 50.0,
            "candle_size": 1.0,
            "dom_pressure": 0.0,
            "pred_vol_5": 0.01,
            "pred_dev": 0.01,
            "atr_change": 0.0,
            "volume_shock": tick.vol_shock if hasattr(tick, 'vol_shock') else 0.0,
        }
        
        # ========== 检查现有持仓的离场条件 ==========
        current_position = self.positions.get(symbol)
        if current_position is not None:
            old_sl = current_position.get("current_sl")
            exit_reason = self._check_exit_conditions(signal, current_position, market_state)
            
            # 如果没有离场，检查是否需要同步移动止损 (Ratchet Stop)
            if not exit_reason:
                new_sl = current_position.get("current_sl")
                if new_sl and abs(new_sl - (old_sl or 0)) > 1e-7:
                    self._sync_sl_to_broker(symbol, new_sl)
            
            if exit_reason:
                # 生成平仓指令
                close_order = OrderCommand(
                    uuid=str(uuid.uuid4())[:8],
                    action=OrderAction.CLOSE,
                    symbol=symbol,
                    side=current_position["side"],
                    order_type=OrderType.MARKET,
                    lots=current_position["lots"],
                    magic=MAGIC_NUMBER,
                    comment=f"QLink_{exit_reason}",
                )
                logger.info("🔻 触发离场: %s %s %.2f lots 原因=%s", 
                           symbol, current_position["side"].value, 
                           current_position["lots"], exit_reason)
                
                # 清除持仓记录
                del self.positions[symbol]
                
                # 如果是信号反转，继续处理新开仓；否则只平仓
                if exit_reason != "EXIT_SIGNAL_REVERSAL":
                    return RiskDecision(
                        timestamp=signal.timestamp,
                        symbol=symbol,
                        action="EXIT",
                        meta_prob=0,
                        position_size=0,
                        kelly_fraction=0,
                        vol_scalar=1.0,
                        lvar_cost=0,
                        alpha_signal=signal,
                        close_order=close_order,
                    )
        
        # ========== 第一道防线：Meta-Labeling (Quantum) ==========
        # 优先使用 Quantum Meta-Labeling (符合 XAUUSD_Quantum_Strategic_Research.md)
        qt_prob = self.quantum_meta.predict(signal, market_state)
        
        # Shadow Mode: 记录 XGBoost 预测值用于对比
        xgb_prob = self.meta_labeler.predict(signal, market_state)
        
        # 决策使用量子概率
        meta_prob = qt_prob
        
        logger.info("Meta-Labeling (Quantum): prob=%.3f (XGB Shadow: %.3f) threshold=%.2f", 
                    qt_prob, xgb_prob, META_THRESHOLD)
        
        if meta_prob < META_THRESHOLD:
            logger.info("信号被 Meta-Labeling 拒绝: prob=%.3f < %.2f", meta_prob, META_THRESHOLD)
            return RiskDecision(
                timestamp=signal.timestamp,
                symbol=symbol,
                action="PASS",
                meta_prob=meta_prob,
                position_size=0,
                kelly_fraction=0,
                vol_scalar=1.0,
                lvar_cost=0,
                alpha_signal=signal,
            )
        
        # ========== 第二道防线：仓位计算 ==========
        equity = self.account_state.equity if self.account_state else 10000.0
        current_vol = market_state["atr"] / tick.mid_price if tick.mid_price > 0 else 0.01
        
        position_size, kelly_fraction, vol_scalar = self.position_sizer.calculate(
            meta_prob=meta_prob,
            current_vol=current_vol,
            equity=equity,
        )
        
        logger.info("仓位计算: size=%.2f conf=%.3f vol_scalar=%.2f", 
                    position_size, kelly_fraction, vol_scalar)
        
        if position_size < MIN_POSITION_SIZE:
            logger.info("仓位过小，跳过交易: size=%.4f < %.4f", position_size, MIN_POSITION_SIZE)
            return RiskDecision(
                timestamp=signal.timestamp,
                symbol=symbol,
                action="PASS",
                meta_prob=meta_prob,
                position_size=position_size,
                kelly_fraction=kelly_fraction,
                vol_scalar=vol_scalar,
                lvar_cost=0,
                alpha_signal=signal,
            )
        
        # ========== 第三道防线：L-VaR ==========
        lvar_cost, cost_breakdown = self.lvar_controller.calculate_lvar(
            volume_density=tick.vol_density,
            position_size=position_size,
            spread=tick.spread_points,
        )
        base_c, spread_c, size_c = cost_breakdown
        
        # 1. 检查静默期 (Cooldown)
        now = time.time()
        last_time = self._last_entry_time.get(signal.symbol, 0)
        # 如果是同方向信号，且距离上次开仓不足 10 秒，则跳过
        if signal.direction == OrderSide.SELL and now - last_time < 10.0:
            return RiskDecision(
                timestamp=signal.timestamp,
                symbol=symbol,
                action="PASS",
                meta_prob=meta_prob,
                position_size=0,
                kelly_fraction=0,
                vol_scalar=1.0,
                lvar_cost=0,
                alpha_signal=signal,
            )
        # 多头同理
        if signal.direction == OrderSide.BUY and now - last_time < 10.0:
            return RiskDecision(
                timestamp=signal.timestamp,
                symbol=symbol,
                action="PASS",
                meta_prob=meta_prob,
                position_size=0,
                kelly_fraction=0,
                vol_scalar=1.0,
                lvar_cost=0,
                alpha_signal=signal,
            )
        
        # 修正：将每单位预期收益转换为整个仓位的总预期收益
        contract_size = 100 if "XAU" in symbol else 1.0  # 黄金标准合约 100 盎司
        expected_return = abs(signal.prediction) * position_size * contract_size
        
        logger.info("L-VaR: cost=%.2f (base=%.2f spread=%.2f size=%.2f) expected_profit=%.4f", 
                    lvar_cost, base_c, spread_c, size_c, expected_return)
        
        if expected_return < lvar_cost:
            logger.info("预期收益低于成本，跳过交易: return=%.4f < cost=%.2f", 
                        expected_return, lvar_cost)
            return RiskDecision(
                timestamp=signal.timestamp,
                symbol=symbol,
                action="PASS",
                meta_prob=meta_prob,
                position_size=position_size,
                kelly_fraction=kelly_fraction,
                vol_scalar=vol_scalar,
                lvar_cost=lvar_cost,
                alpha_signal=signal,
            )
        
        # ========== 检查信号反转平仓 ==========
        current_position = self.positions.get(symbol)
        close_order = None
        
        if current_position is not None:
            current_side = current_position["side"]
            # 信号反转：持有多头但新信号是空头，或持有空头但新信号是多头
            if (current_side == OrderSide.BUY and signal.direction == OrderSide.SELL) or \
               (current_side == OrderSide.SELL and signal.direction == OrderSide.BUY):
                # 生成平仓指令
                close_order = OrderCommand(
                    uuid=str(uuid.uuid4())[:8],
                    action=OrderAction.CLOSE,
                    symbol=symbol,
                    side=current_side,  # 平掉当前持仓方向
                    order_type=OrderType.MARKET,
                    lots=current_position["lots"],
                    magic=MAGIC_NUMBER,
                    comment="QLink_Reversal",
                )
                logger.info("🔄 信号反转平仓: %s %s %.2f lots", symbol, current_side.value, current_position["lots"])
                # 清除持仓记录
                del self.positions[symbol]
        
        # ========== 计算 ATR 止损止盈 ==========
        atr = market_state["atr"]  # 平均真实波幅
        entry_price = tick.mid_price
        
        if signal.direction == OrderSide.BUY:
            sl_price = entry_price - atr * ATR_SL_MULTIPLIER
            tp_price = entry_price + atr * ATR_TP_MULTIPLIER
        else:
            sl_price = entry_price + atr * ATR_SL_MULTIPLIER
            tp_price = entry_price - atr * ATR_TP_MULTIPLIER
        
        # ========== 生成开仓订单 ==========
        order = OrderCommand(
            uuid=str(uuid.uuid4())[:8],
            action=OrderAction.OPEN,
            symbol=symbol,
            side=signal.direction,
            order_type=OrderType.MARKET,
            lots=round(position_size, 2),
            sl=round(sl_price, 5),
            tp=round(tp_price, 5),
            magic=MAGIC_NUMBER,
            comment=f"QLink_{signal.direction.value}",
        )
        
        # 记录持仓 (增强版，支持三层离场检测)
        self.positions[symbol] = {
            "side": signal.direction,
            "entry_price": entry_price,
            "entry_confidence": signal.confidence,  # 入场置信度
            "entry_time": signal.timestamp,         # 入场时间戳
            "current_sl": sl_price,                 # 棘轮止损初始值
            "lots": round(position_size, 2),
            "bars_held": 0,                         # 持仓 K 线计数
        }
        
        logger.info("✅ 生成订单: %s %s %.2f lots SL=%.2f TP=%.2f", 
                    symbol, order.side.value, order.lots, sl_price, tp_price)
        
        return RiskDecision(
            timestamp=signal.timestamp,
            symbol=symbol,
            action="BET",
            meta_prob=meta_prob,
            position_size=position_size,
            kelly_fraction=kelly_fraction,
            vol_scalar=vol_scalar,
            lvar_cost=lvar_cost,
            alpha_signal=signal,
            order=order,
            close_order=close_order,  # 反转平仓指令
        )
    
    def _send_order(self, order: Optional[OrderCommand]) -> None:
        """发送订单到 MT5"""
        if order is None:
            return
        
        try:
            msg = order.to_json()
            self.command_socket.send_string(msg, zmq.NOBLOCK)
            if order.action == OrderAction.MODIFY:
                logger.info("🔧 修改指令已发送: ticket=%s sl=%.5f", order.ticket, order.sl)
            else:
                logger.info("订单已发送: %s action=%s", order.uuid, order.action.value)
        except zmq.Again:
            logger.warning("订单发送缓冲区满")
        except Exception as e:
            logger.error("订单发送失败: %s", e)
    
    def _sync_sl_to_broker(self, symbol: str, new_sl: float) -> None:
        """同步止损到经纪商端的所有相关持仓"""
        if not self.account_state:
            return
            
        for p in self.account_state.positions:
            # 匹配品种和魔术数
            if p.symbol == symbol and p.magic == MAGIC_NUMBER:
                # 如果经纪商端的止损与我们计算出的棘轮止损不一致，则发起修改
                # 兼容性修复：处理 MT5 端 sl 为 None 的情况
                broker_sl = p.sl or 0.0
                if abs(broker_sl - new_sl) > 1e-5:
                    modify_order = OrderCommand(
                        uuid=str(uuid.uuid4())[:8],
                        action=OrderAction.MODIFY,
                        symbol=symbol,
                        side=p.side,
                        order_type=OrderType.MARKET,
                        lots=p.lots,
                        sl=round(new_sl, 5),
                        tp=p.tp,
                        magic=MAGIC_NUMBER,
                        ticket=p.ticket,
                    )
                    self._send_order(modify_order)
    
    def _check_exit_conditions(
        self, 
        signal: AlphaSignal, 
        position: Dict[str, Any], 
        market_state: Dict[str, float]
    ) -> Optional[str]:
        """
        三层离场检测
        
        Returns:
            离场原因字符串，或 None 表示继续持仓
        """
        symbol = signal.symbol
        tick = signal.tick_data
        
        # === 第 3 层：微观结构熔断 (最高优先级) ===
        if self._is_out_of_distribution(market_state):
            logger.warning("🚨 微观结构熔断触发: %s", symbol)
            return "EXIT_OOD_CIRCUIT_BREAKER"
        
        # === 第 2 层：信号衰减离场 ===
        current_confidence = signal.confidence
        entry_confidence = position.get("entry_confidence", 0.5)
        
        # 置信度下降超过 20%
        if current_confidence < entry_confidence * CONFIDENCE_DECAY_RATIO:
            logger.info("📉 信号衰减离场: 置信度从 %.2f 下降到 %.2f", 
                       entry_confidence, current_confidence)
            return "EXIT_SIGNAL_DECAY"
        
        # 信号方向翻转
        if (position["side"] == OrderSide.BUY and signal.direction == OrderSide.SELL) or \
           (position["side"] == OrderSide.SELL and signal.direction == OrderSide.BUY):
            logger.info("🔄 信号反转离场: %s -> %s", 
                       position["side"].value, signal.direction.value)
            return "EXIT_SIGNAL_REVERSAL"
        
        # 时间障碍：持仓超过最大 K 线数
        bars_held = position.get("bars_held", 0) + 1
        if bars_held >= MAX_HOLDING_BARS:
            logger.info("⏰ 时间障碍离场: 持仓 %d 根 K 线", bars_held)
            return "EXIT_TIME_BARRIER"
        
        # 更新持仓 K 线计数
        position["bars_held"] = bars_held
        
        # === 第 1 层：波动率自适应棘轮止损 ===
        atr = market_state.get("atr", 1.0)
        
        # 动态 ATR 乘数
        k_factor = ATR_K_HIGH_CONFIDENCE if current_confidence > CONFIDENCE_HIGH_THRESHOLD else ATR_K_LOW_CONFIDENCE
        
        # 棘轮止损 (只能向有利方向移动)
        if position["side"] == OrderSide.BUY:
            candidate_sl = tick.mid_price - k_factor * atr
            new_sl = max(position.get("current_sl", -np.inf), candidate_sl)
            position["current_sl"] = new_sl
            
            if tick.mid_price <= new_sl:
                logger.info("🛑 棘轮止损触发 (多头): 价格 %.2f <= 止损 %.2f", 
                           tick.mid_price, new_sl)
                return "EXIT_TRAILING_STOP"
        else:
            candidate_sl = tick.mid_price + k_factor * atr
            new_sl = min(position.get("current_sl", np.inf), candidate_sl)
            position["current_sl"] = new_sl
            
            if tick.mid_price >= new_sl:
                logger.info("🛑 棘轮止损触发 (空头): 价格 %.2f >= 止损 %.2f", 
                           tick.mid_price, new_sl)
                return "EXIT_TRAILING_STOP"
        
        return None  # 继续持仓
    
    def _is_out_of_distribution(self, market_state: Dict[str, float]) -> bool:
        """
        微观结构熔断检测 (OOD 检测)
        
        如果关键指标超出训练分布 3σ，认为市场进入异常状态
        """
        volume_shock = market_state.get("volume_shock", 0)
        
        # 检查 volume_shock 是否超出 3σ
        if abs(volume_shock) > OOD_THRESHOLD_SIGMA:
            return True
        
        return False
    
    def _sync_account_state(self) -> None:
        """同步账户状态"""
        try:
            self.state_socket.send_string(MessageType.STATE_REQ.value, zmq.NOBLOCK)
            
            # 等待响应（非阻塞 + 超时）
            if self.state_socket.poll(timeout=1000):
                response = self.state_socket.recv_string()
                if response.startswith("{"):
                    self.account_state = AccountState.from_json(response)
                    logger.debug("账户同步: equity=%.2f", self.account_state.equity)
        except Exception as e:
            logger.debug("账户同步失败: %s", e)
    
    def stop(self) -> None:
        """停止服务"""
        logger.info("Risk Engine 停止中...")
        self._stop_event.set()
        
        if self.signal_socket:
            self.signal_socket.close()
        if self.command_socket:
            self.command_socket.close()
        if self.state_socket:
            self.state_socket.close()
        if self.context:
            self.context.term()


# ================== 入口 ==================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q-Link Risk Engine")
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
        "--command-port",
        type=int,
        default=5558,
        help="Command Bus 端口（默认 5558）",
    )
    return p.parse_args()


def main() -> int:
    _setup_logging()
    args = _parse_args()
    
    model_dir = Path(args.model_dir).expanduser().resolve()
    signal_queue = Queue(maxsize=100)
    
    engine = RiskEngine(
        model_dir=model_dir,
        signal_queue=signal_queue,
        bind_addr=args.bind,
        signal_port=args.signal_port,
        command_port=args.command_port,
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
