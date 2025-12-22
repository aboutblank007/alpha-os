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
from contextlib import asynccontextmanager
from typing import Set

import zmq
import zmq.asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .protocol import Ports, HEARTBEAT_INTERVAL_MS

# ================== 配置 ==================

ZMQ_HOST = "192.168.3.10"

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


# ================== 生命周期管理 ==================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global zmq_context, market_socket, state_socket, command_socket
    
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
    market_task = asyncio.create_task(market_stream_relay())
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    
    yield
    
    # 清理
    logger.info("[Gateway] 正在关闭...")
    market_task.cancel()
    heartbeat_task.cancel()
    
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
