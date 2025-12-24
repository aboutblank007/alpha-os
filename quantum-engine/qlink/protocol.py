#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q-Link 协议定义

基于 ZeroMQ 的量子-经典双优侧车通信协议。
定义端口、消息格式、序列化器等核心组件。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger("qlink.protocol")

# ================== 端口配置 ==================

class Ports:
    """Q-Link 通道端口配置"""
    MARKET_STREAM = 5557  # MT5 -> Python (PUSH/PULL)
    COMMAND_BUS = 5558    # Python -> MT5 (PUSH/PULL)
    STATE_SYNC = 5559     # 双向 (REQ/REP)
    ALPHA_TO_RISK = 5560  # Alpha -> Risk 内部管道 (PUSH/PULL)


# ================== 消息类型 ==================

class MessageType(str, Enum):
    """消息类型枚举"""
    TICK = "TICK"           # 行情数据
    ORDER = "ORDER"         # 订单指令
    HEARTBEAT = "HB"        # 心跳
    STATE_REQ = "STATE_REQ" # 状态请求
    STATE_RES = "STATE_RES" # 状态响应


class OrderAction(str, Enum):
    """订单动作"""
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    MODIFY = "MODIFY"


class OrderSide(str, Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """订单类型"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


# ================== 数据结构 ==================

@dataclass
class TickData:
    """
    Tick 行情数据（紧凑 CSV 格式）
    
    CSV 格式: TICK,timestamp,symbol,bid,ask,volume,wick_ratio,vol_density,vol_shock,ema_fast,ema_slow,rsi,dom_pressure
    """
    timestamp: int          # 毫秒时间戳
    symbol: str             # 交易品种
    bid: float              # 买价
    ask: float              # 卖价
    volume: int             # 成交量
    wick_ratio: float       # 影线比率
    vol_density: float      # 成交量密度
    vol_shock: float        # 成交量冲击
    ema_fast: float         # 快速均线 (e.g. EMA14)
    ema_slow: float         # 慢速均线 (e.g. EMA50)
    rsi: float              # RSI 指标
    dom_pressure: float     # DOM 压力代理值
    
    # 额外元数据
    spread: Optional[int] = None
    tick_rate: Optional[int] = None
    bid_ask_imbalance: Optional[float] = None
    
    @classmethod
    def from_csv(cls, csv_line: str) -> "TickData":
        """解析 CSV 格式的 Tick 数据"""
        parts = csv_line.strip().split(",")
        if len(parts) < 13 or parts[0] != MessageType.TICK.value:
            raise ValueError(f"Invalid TICK format: {csv_line}")
        
        return cls(
            timestamp=int(parts[1]),
            symbol=parts[2],
            bid=float(parts[3]),
            ask=float(parts[4]),
            volume=int(parts[5]),
            wick_ratio=float(parts[6]),
            vol_density=float(parts[7]),
            vol_shock=float(parts[8]),
            ema_fast=float(parts[9]),
            ema_slow=float(parts[10]),
            rsi=float(parts[11]),
            dom_pressure=float(parts[12]),
            spread=int(parts[13]) if len(parts) > 13 else None,
            tick_rate=int(parts[14]) if len(parts) > 14 else None,
            bid_ask_imbalance=float(parts[15]) if len(parts) > 15 else None,
        )
    
    def to_csv(self) -> str:
        """序列化为 CSV 格式"""
        base = (
            f"{MessageType.TICK.value},{self.timestamp},{self.symbol},"
            f"{self.bid:.5f},{self.ask:.5f},{self.volume},"
            f"{self.wick_ratio:.4f},{self.vol_density:.4f},{self.vol_shock:.4f},"
            f"{self.ema_fast:.5f},{self.ema_slow:.5f},{self.rsi:.4f},{self.dom_pressure:.4f}"
        )
        if self.spread is not None:
            base += f",{self.spread},{self.tick_rate},{self.bid_ask_imbalance:.5f}"
        return base
    
    @property
    def mid_price(self) -> float:
        """中间价"""
        return (self.bid + self.ask) / 2
    
    @property
    def spread_points(self) -> float:
        """点差"""
        return self.ask - self.bid


@dataclass
class OrderCommand:
    """
    订单指令（JSON 格式）
    """
    uuid: str               # 唯一标识
    action: OrderAction     # 动作
    symbol: str             # 品种
    side: OrderSide         # 方向
    order_type: OrderType   # 类型
    lots: float             # 手数
    sl: Optional[float] = None      # 止损价
    tp: Optional[float] = None      # 止盈价
    price: Optional[float] = None   # 限价（LIMIT/STOP 类型）
    magic: int = 999001     # 魔术数
    comment: str = ""       # 注释
    ticket: Optional[int] = None    # 订单/持仓票据号 (用于 MODIFY/CLOSE)
    
    def to_json(self) -> str:
        """序列化为 JSON"""
        return json.dumps({
            "uuid": self.uuid,
            "action": self.action.value,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.order_type.value,
            "lots": self.lots,
            "sl": self.sl,
            "tp": self.tp,
            "price": self.price,
            "magic": self.magic,
            "comment": self.comment,
            "ticket": self.ticket,
        }, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "OrderCommand":
        """从 JSON 反序列化"""
        data = json.loads(json_str)
        return cls(
            uuid=data["uuid"],
            action=OrderAction(data["action"]),
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            order_type=OrderType(data["type"]),
            lots=data["lots"],
            sl=data.get("sl"),
            tp=data.get("tp"),
            price=data.get("price"),
            magic=data.get("magic", 999001),
            comment=data.get("comment", ""),
        )


@dataclass
class Position:
    """持仓信息"""
    ticket: int             # 订单号
    symbol: str             # 品种
    side: OrderSide         # 方向
    lots: float             # 手数
    open_price: float       # 开仓价
    current_price: float    # 当前价
    profit: float           # 浮盈
    sl: Optional[float] = None
    tp: Optional[float] = None
    magic: int = 0
    comment: str = ""


@dataclass
class AccountState:
    """
    账户状态（JSON 格式）
    """
    balance: float          # 余额
    equity: float           # 净值
    margin_used: float      # 已用保证金
    margin_free: float      # 可用保证金
    positions: List[Position] = field(default_factory=list)
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    
    def to_json(self) -> str:
        """序列化为 JSON"""
        return json.dumps({
            "balance": self.balance,
            "equity": self.equity,
            "margin_used": self.margin_used,
            "margin_free": self.margin_free,
            "positions": [asdict(p) for p in self.positions],
            "timestamp": self.timestamp,
        }, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "AccountState":
        """从 JSON 反序列化"""
        data = json.loads(json_str)
        positions = [
            Position(
                ticket=p["ticket"],
                symbol=p["symbol"],
                side=OrderSide(p["side"]),
                lots=p["lots"],
                open_price=p["open_price"],
                current_price=p["current_price"],
                profit=p["profit"],
                sl=p.get("sl"),
                tp=p.get("tp"),
                magic=p.get("magic", 0),
                comment=p.get("comment", ""),
            )
            for p in data.get("positions", [])
        ]
        return cls(
            balance=data["balance"],
            equity=data["equity"],
            margin_used=data.get("margin_used", 0),
            margin_free=data["margin_free"],
            positions=positions,
            timestamp=data.get("timestamp", int(time.time() * 1000)),
        )


@dataclass
class AlphaSignal:
    """
    Alpha 引擎输出的预测信号
    """
    timestamp: int              # 时间戳
    symbol: str                 # 品种
    prediction: float           # 预测值（价格变动）
    direction: OrderSide        # 方向
    confidence: float           # 置信度 [0, 1]
    tick_data: TickData         # 原始 Tick 数据
    
    # 量子模型内部状态
    q_state_entropy: Optional[float] = None  # 量子态熵
    raw_expectation: Optional[float] = None  # 量子测量期望值 [-1, 1] (Pauli-Z)
    
    def to_json(self) -> str:
        """序列化为 JSON"""
        return json.dumps({
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "prediction": self.prediction,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "tick_data": {
                "timestamp": self.tick_data.timestamp,
                "symbol": self.tick_data.symbol,
                "bid": self.tick_data.bid,
                "ask": self.tick_data.ask,
                "volume": self.tick_data.volume,
                "wick_ratio": self.tick_data.wick_ratio,
                "vol_density": self.tick_data.vol_density,
                "vol_shock": self.tick_data.vol_shock,
                "ema_fast": self.tick_data.ema_fast,
                "ema_slow": self.tick_data.ema_slow,
                "rsi": self.tick_data.rsi,
                "dom_pressure": self.tick_data.dom_pressure,
                "spread": self.tick_data.spread,
                "tick_rate": self.tick_data.tick_rate,
                "bid_ask_imbalance": self.tick_data.bid_ask_imbalance,
            },
            "q_state_entropy": self.q_state_entropy,
            "raw_expectation": self.raw_expectation,
        }, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "AlphaSignal":
        """从 JSON 反序列化"""
        data = json.loads(json_str)
        tick_data = TickData(
            timestamp=data["tick_data"]["timestamp"],
            symbol=data["tick_data"]["symbol"],
            bid=data["tick_data"]["bid"],
            ask=data["tick_data"]["ask"],
            volume=data["tick_data"]["volume"],
            wick_ratio=data["tick_data"]["wick_ratio"],
            vol_density=data["tick_data"]["vol_density"],
            vol_shock=data["tick_data"]["vol_shock"],
            ema_fast=data["tick_data"].get("ema_fast", 0.0),
            ema_slow=data["tick_data"].get("ema_slow", 0.0),
            rsi=data["tick_data"].get("rsi", 50.0),
            dom_pressure=data["tick_data"].get("dom_pressure", 0.0),
            spread=data["tick_data"].get("spread"),
            tick_rate=data["tick_data"].get("tick_rate"),
            bid_ask_imbalance=data["tick_data"].get("bid_ask_imbalance"),
        )
        return cls(
            timestamp=data["timestamp"],
            symbol=data["symbol"],
            prediction=data["prediction"],
            direction=OrderSide(data["direction"]),
            confidence=data["confidence"],
            tick_data=tick_data,
            q_state_entropy=data.get("q_state_entropy"),
            raw_expectation=data.get("raw_expectation"),
        )


@dataclass
class RiskDecision:
    """
    风控引擎的最终决策
    """
    timestamp: int
    symbol: str
    action: str                 # "BET" | "PASS"
    
    # Meta-Labeling 输出
    meta_prob: float            # 元模型概率
    
    # 仓位计算
    position_size: float        # 最终仓位（手数）
    kelly_fraction: float       # 凯利比例
    vol_scalar: float           # 波动率缩放系数
    
    # 风控检查结果
    lvar_cost: float            # 隐含滑点成本
    alpha_signal: Optional[AlphaSignal] = None
    
    # 如果 action == "BET"，生成订单指令
    order: Optional[OrderCommand] = None
    close_order: Optional[OrderCommand] = None  # 反转平仓指令


# ================== 心跳协议 ==================

@dataclass
class Heartbeat:
    """心跳消息"""
    source: str             # "ALPHA" | "RISK" | "MT5"
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    status: str = "OK"
    
    def to_csv(self) -> str:
        return f"{MessageType.HEARTBEAT.value},{self.source},{self.timestamp},{self.status}"
    
    @classmethod
    def from_csv(cls, csv_line: str) -> "Heartbeat":
        parts = csv_line.strip().split(",")
        if len(parts) < 4 or parts[0] != MessageType.HEARTBEAT.value:
            raise ValueError(f"Invalid HEARTBEAT format: {csv_line}")
        return cls(
            source=parts[1],
            timestamp=int(parts[2]),
            status=parts[3],
        )


# ================== 常量 ==================

# ZMQ 配置
ZMQ_HWM = 1000              # 高水位线
ZMQ_LINGER = 0              # 关闭时不等待

# 超时配置
HEARTBEAT_INTERVAL_MS = 1000    # 心跳间隔
HEARTBEAT_TIMEOUT_MS = 5000     # 心跳超时（死人开关）
LATENCY_THRESHOLD_MS = 100      # 延迟阈值（看门狗）

# 风控参数
META_THRESHOLD = 0.15            # Meta-Labeling 阈值 (量子置信度概率 [0,1])
TARGET_VOLATILITY = 0.04        # 目标波动率（2%）
MAX_POSITION_SIZE = 1.0         # 最大仓位（手）
MIN_POSITION_SIZE = 0.05        # 最小仓位（手）
ATR_SL_MULTIPLIER = 4.0         # ATR 止损倍数 (SL = Entry ± N × ATR)
ATR_TP_MULTIPLIER = 6.0         # ATR 止盈倍数 (TP = Entry ± N × ATR)

# 离场策略配置 (参考 docs/交易风控离场策略研究.md)
CONFIDENCE_DECAY_RATIO = 0.8    # 信号衰减阈值 (置信度下降 20%)
OOD_THRESHOLD_SIGMA = 3.0       # 微观结构熔断阈值 (3σ)
ATR_K_HIGH_CONFIDENCE = 2.5     # 高置信度 ATR 乘数
ATR_K_LOW_CONFIDENCE = 1.0      # 低置信度 ATR 乘数
CONFIDENCE_HIGH_THRESHOLD = 0.7 # 高置信度阈值
MAX_HOLDING_BARS = 60           # 最大持仓 K 线数 (时间障碍)

# 魔术数
MAGIC_NUMBER = 999001           # Q-Link 系统魔术数
