#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI WebSocket Gateway

Q-Link 2.0 协议的 HTTP/WebSocket 网关
代理 ZMQ 端口，提供前端连接接口
"""

import asyncio
import json
import time
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Set

import zmq
import zmq.asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

import os

# ================== 配置 ==================

ZMQ_HOST = os.getenv("ZMQ_HOST", "127.0.0.1")
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

# ================== 全局状态 ==================

# 活跃的 WebSocket 连接
market_clients: Set[WebSocket] = set()
state_clients: Set[WebSocket] = set()
command_clients: Set[WebSocket] = set()

# ZMQ 上下文
zmq_context: zmq.asyncio.Context = None
market_socket: zmq.asyncio.Socket = None
state_socket: zmq.asyncio.Socket = None
command_socket: zmq.asyncio.Socket = None

try:
    from .protocol import Ports, HEARTBEAT_INTERVAL_MS
except ImportError:
    from protocol import Ports, HEARTBEAT_INTERVAL_MS


# ================== 生命周期管理 ==================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global zmq_context, market_socket, state_socket, command_socket
    
    tasks = []

    if MOCK_MODE:
        logger.warning("[Gateway] 启动 Mock 模式 (无 ZMQ 连接)")
        tasks.append(asyncio.create_task(mock_data_generator()))
    else:
        logger.info("[Gateway] 正在初始化 ZMQ 连接...")
        
        # 初始化 ZMQ
        zmq_context = zmq.asyncio.Context()
        
        # Market Stream (PULL)
        market_socket = zmq_context.socket(zmq.PULL)
        market_socket.connect(f"tcp://{ZMQ_HOST}:{Ports.MARKET_STREAM}")
        market_socket.setsockopt(zmq.RCVTIMEO, 100)
        
        # State Sync (REQ)
        state_socket = zmq_context.socket(zmq.REQ)
        state_socket.connect(f"tcp://{ZMQ_HOST}:{Ports.STATE_SYNC}")
        state_socket.setsockopt(zmq.RCVTIMEO, 1000)
        
        # Command Bus (PUSH)
        command_socket = zmq_context.socket(zmq.PUSH)
        command_socket.connect(f"tcp://{ZMQ_HOST}:{Ports.COMMAND_BUS}")
        
        logger.info("[Gateway] ZMQ 连接已建立")
        
        # 启动后台任务
        tasks.append(asyncio.create_task(market_stream_relay()))
    
    tasks.append(asyncio.create_task(heartbeat_loop()))
    
    yield
    
    # 清理
    logger.info("[Gateway] 正在关闭...")
    for task in tasks:
        task.cancel()
    
    if not MOCK_MODE:
        market_socket.close()
        state_socket.close()
        command_socket.close()
        zmq_context.term()
    
    logger.info("[Gateway] 已关闭")


# ================== FastAPI 应用 ==================

app = FastAPI(
    title="Q-Link Gateway",
    description="Quantum HFT WebSocket Gateway",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================== 后台任务 ==================

async def mock_data_generator():
    """生成模拟数据 (仅 Mock 模式)"""
    logger.info("[Gateway] Mock 数据生成器已启动")
    import random
    
    price = 2050.0
    last_config_broadcast = 0
    
    while True:
        try:
            now = time.time()
            
            # 1. 模拟 Market Stream (Tick)
            # 格式: SYMBOL,TIMESTAMP,BID,ASK,VOLUME
            price += random.uniform(-0.5, 0.5)
            bid = price
            ask = price + 0.1
            timestamp = int(now * 1000)
            tick_msg = f"XAUUSD,{timestamp},{bid:.2f},{ask:.2f},1.5"
            
            if market_clients:
                await asyncio.gather(
                    *[client.send_text(tick_msg) for client in market_clients],
                    return_exceptions=True
                )
                
            # 2. 模拟 Quantum Telemetry (通过 State Sync 通道广播)
            if random.random() < 0.2: # 每 5 次 tick 发送一次 telemetry
                telemetry = {
                    "type": "TELEMETRY",
                    "payload": {
                        "gradientNorm": random.uniform(0.0001, 0.01),
                        "entropy": random.uniform(0.1, 0.9),
                        "latency": random.uniform(10, 50),
                        "pCoreLoad": random.uniform(20, 80),
                        "eCoreLoad": random.uniform(10, 40),
                        "ai_score": random.uniform(0.4, 0.9),
                        "regime": "High Volatility" if random.random() > 0.5 else "Stable",
                        "timestamp": timestamp
                    }
                }
                if state_clients:
                    await asyncio.gather(
                        *[client.send_text(json.dumps(telemetry)) for client in state_clients],
                        return_exceptions=True
                    )

            # 3. 模拟 AI Config 广播 (每 5 秒)
            if now - last_config_broadcast > 5:
                config = {
                    "type": "AI_CONFIG",
                    "payload": {
                        "risk_off": False,
                        "min_confidence": 0.65,
                        "max_vol_mult": 1.5,
                        "mode": "balanced",
                        "updated_at": datetime.fromtimestamp(now).isoformat()
                    }
                }
                if state_clients:
                    await asyncio.gather(
                        *[client.send_text(json.dumps(config)) for client in state_clients],
                        return_exceptions=True
                    )
                last_config_broadcast = now

            # 4. 模拟 AI Log (偶尔)
            if random.random() < 0.05: # 低频
                action = random.choice(["BUY", "SELL", "WAIT"])
                ai_log = {
                    "type": "AI_LOG",
                    "payload": {
                        "id": f"log_{timestamp}_{random.randint(1000,9999)}",
                        "symbol": "XAUUSD",
                        "action": action,
                        "price": bid,
                        "timestamp": datetime.fromtimestamp(now).isoformat(),
                        "resultProfit": random.uniform(-50, 100) if random.random() > 0.5 else None,
                        "aiScore": random.uniform(0.5, 0.95),
                        "regime": "Stable",
                        "metaProb": random.uniform(0.6, 0.9),
                        "dqnAction": random.randint(0, 2),
                        "quantumPolicy": [random.random() for _ in range(3)]
                    }
                }
                if state_clients:
                    await asyncio.gather(
                        *[client.send_text(json.dumps(ai_log)) for client in state_clients],
                        return_exceptions=True
                    )

            await asyncio.sleep(0.01) # 100 TPS
        except Exception as e:
            logger.error(f"[Gateway] Mock 生成器错误: {e}")
            await asyncio.sleep(1)

async def market_stream_relay():
    """中继 ZMQ 市场数据到 WebSocket 客户端"""
    logger.info("[Gateway] Market Stream 中继任务已启动")
    
    while True:
        try:
            # 非阻塞接收
            message = await market_socket.recv_string(flags=zmq.NOBLOCK)
            
            # 广播到所有客户端
            if market_clients:
                await asyncio.gather(
                    *[client.send_text(message) for client in market_clients],
                    return_exceptions=True
                )
        except zmq.Again:
            # 无数据可读
            await asyncio.sleep(0.001)
        except Exception as e:
            logger.error(f"[Gateway] Market Stream 错误: {e}")
            await asyncio.sleep(0.1)


async def heartbeat_loop():
    """心跳广播循环"""
    while True:
        try:
            heartbeat = {
                "type": "HEARTBEAT",
                "timestamp": int(time.time() * 1000),
                "source": "GATEWAY",
            }
            
            # 广播心跳到所有 State 客户端
            if state_clients:
                message = json.dumps(heartbeat)
                await asyncio.gather(
                    *[client.send_text(message) for client in state_clients],
                    return_exceptions=True
                )
            
            await asyncio.sleep(HEARTBEAT_INTERVAL_MS / 1000)
        except Exception as e:
            logger.error(f"[Gateway] 心跳错误: {e}")
            await asyncio.sleep(1)


# ================== WebSocket 端点 ==================

@app.websocket("/ws/market_stream")
async def market_stream_endpoint(websocket: WebSocket):
    """Market Stream WebSocket 端点"""
    await websocket.accept()
    market_clients.add(websocket)
    logger.info(f"[Gateway] Market Stream 客户端已连接 (总数: {len(market_clients)})")
    
    try:
        while True:
            # 保持连接活跃 (接收客户端消息)
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        market_clients.discard(websocket)
        logger.info(f"[Gateway] Market Stream 客户端已断开 (总数: {len(market_clients)})")


@app.websocket("/ws/state_sync")
async def state_sync_endpoint(websocket: WebSocket):
    """State Sync WebSocket 端点"""
    await websocket.accept()
    state_clients.add(websocket)
    logger.info(f"[Gateway] State Sync 客户端已连接 (总数: {len(state_clients)})")
    
    try:
        while True:
            # 接收状态请求
            request = await websocket.receive_text()
            
            if MOCK_MODE:
                # Mock 模式下的简单响应
                try:
                    req_data = json.loads(request)
                    if req_data.get("type") == "GET_CONFIG":
                        response = {
                            "type": "AI_CONFIG",
                            "payload": {
                                "risk_off": False,
                                "min_confidence": 0.65,
                                "max_vol_mult": 1.5,
                                "mode": "balanced",
                                "updated_at": datetime.now().isoformat()
                            }
                        }
                        await websocket.send_text(json.dumps(response))
                    else:
                        await websocket.send_text(json.dumps({"type": "ACK", "message": "Mock processed"}))
                except:
                     await websocket.send_text(json.dumps({"type": "ACK", "message": "Mock processed"}))
            else:
                try:
                    # 转发到 ZMQ
                    await state_socket.send_string(request)
                    response = await state_socket.recv_string()
                    await websocket.send_text(response)
                except zmq.Again:
                    await websocket.send_text(json.dumps({
                        "type": "ERROR",
                        "message": "State sync timeout"
                    }))
    except WebSocketDisconnect:
        pass
    finally:
        state_clients.discard(websocket)
        logger.info(f"[Gateway] State Sync 客户端已断开 (总数: {len(state_clients)})")


@app.websocket("/ws/commands")
async def commands_endpoint(websocket: WebSocket):
    """Command Bus WebSocket 端点"""
    await websocket.accept()
    command_clients.add(websocket)
    logger.info(f"[Gateway] Command Bus 客户端已连接 (总数: {len(command_clients)})")
    
    try:
        while True:
            # 接收命令
            command = await websocket.receive_text()
            logger.info(f"[Gateway] 收到命令: {command}")
            
            if MOCK_MODE:
                 await websocket.send_text(json.dumps({
                    "type": "ACK",
                    "timestamp": int(time.time() * 1000),
                    "mock": True
                }))
            else:
                try:
                    # 转发到 ZMQ
                    await command_socket.send_string(command)
                    await websocket.send_text(json.dumps({
                        "type": "ACK",
                        "timestamp": int(time.time() * 1000)
                    }))
                except Exception as e:
                    logger.error(f"[Gateway] 命令发送失败: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "ERROR",
                        "message": str(e)
                    }))
    except WebSocketDisconnect:
        pass
    finally:
        command_clients.discard(websocket)
        logger.info(f"[Gateway] Command Bus 客户端已断开 (总数: {len(command_clients)})")


# ================== REST 端点 ==================

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": int(time.time() * 1000),
        "clients": {
            "market": len(market_clients),
            "state": len(state_clients),
            "command": len(command_clients),
        }
    }


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "Q-Link Gateway",
        "version": "2.0.0",
        "endpoints": {
            "market_stream": "/ws/market_stream",
            "state_sync": "/ws/state_sync",
            "commands": "/ws/commands",
            "health": "/health",
            "status": "/status",
        }
    }


@app.get("/status")
async def status():
    """
    兼容 trading-bridge 的 /status 端点
    前端通过此端点获取连接状态和账户信息
    """
    return {
        "bridge_status": "connected",
        "active_symbols": ["XAUUSD", "BTCUSD", "EURUSD"],
        "symbol_prices": {
            "XAUUSD": {"bid": 2050.12, "ask": 2050.42, "last_seen": int(time.time() * 1000)},
            "BTCUSD": {"bid": 43250.50, "ask": 43255.00, "last_seen": int(time.time() * 1000)},
            "EURUSD": {"bid": 1.0892, "ask": 1.0894, "last_seen": int(time.time() * 1000)},
        },
        "last_mt5_update": {
            "account": {
                "balance": 10000.00,
                "equity": 10000.00,
                "margin": 0.0,
                "free_margin": 10000.00,
            },
            "positions": [],
            "period": "M1",
        },
        "clients": {
            "market": len(market_clients),
            "state": len(state_clients),
            "command": len(command_clients),
        },
        "timestamp": int(time.time() * 1000),
    }


# ================== 主入口 ==================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "qlink.api_gateway:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
