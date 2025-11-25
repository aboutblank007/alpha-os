# AlphaOS 信号系统文档

> 本文档描述了 AlphaOS 的交易信号系统架构、配置和使用方法。

## 系统概述

AlphaOS 信号系统实现了从 MT5 指标到前端通知的完整信号流程，绕过了 TradingView 的 webhook 订阅限制。

### 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         MT5 Terminal                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         PivotTrendSignals.mq5 (MQL5 指标)                │   │
│  │  - 计算 EMA 交叉、ATR 过滤、趋势确认                      │   │
│  │  - 生成 BUY/SELL/RECLAIM 信号                            │   │
│  │  - 写入 JSON 文件到 MQL5/Files/AlphaOS/Signals/          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (Docker 共享卷: signal_data)
┌─────────────────────────────────────────────────────────────────┐
│                      Bridge API (Python)                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              watch_signal_directory()                    │   │
│  │  - 每 500ms 扫描 /app/signals/ 目录                      │   │
│  │  - 读取 JSON 文件内容                                    │   │
│  │  - 插入到 Supabase signals 表                            │   │
│  │  - 删除已处理的文件                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (Supabase Realtime)
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (Next.js)                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              SignalListener.tsx                          │   │
│  │  - 订阅 Supabase signals 表的 INSERT 事件                │   │
│  │  - 显示 Toast 通知                                       │   │
│  │  - 打开 TradePanel 并预填 TP/SL/价格                     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 组件详情

### 1. MQL5 指标 (PivotTrendSignals.mq5)

**位置**: `trading-bridge/mql5/PivotTrendSignals.mq5`

**功能**:
- 基于 Pine Script V3 版本移植
- 计算短期/长期 EMA 交叉
- ATR 过滤器确保信号质量
- 支持多种过滤模式 (Basic/Strict)
- 支持 Reclaim 信号 (价格回归 EMA)
- 自动计算 TP/SL (基于 ATR 倍数)

**信号输出格式** (JSON):
```json
{
  "symbol": "XAUUSD",
  "action": "BUY",
  "price": 2650.50,
  "sl": 2640.00,
  "tp": 2670.00,
  "comment": "买入"
}
```

**文件路径**: `MQL5/Files/AlphaOS/Signals/signal_{SYMBOL}_{TIMESTAMP}.json`

### 2. Python Bridge (main.py)

**位置**: `trading-bridge/src/main.py`

**信号监听函数**:
```python
async def watch_signal_directory():
    """
    每 500ms 扫描信号目录，处理新的 JSON 文件
    """
    SIGNAL_DIR = os.environ.get("SIGNAL_DIR", "/app/signals")
    # ...
```

**环境变量**:
| 变量名 | 说明 | 示例 |
|--------|------|------|
| `SIGNAL_DIR` | 信号文件目录 | `/app/signals` |
| `SUPABASE_URL` | Supabase 项目 URL | `https://xxx.supabase.co` |
| `SUPABASE_KEY` | Supabase Anon Key | `sb_publishable_xxx` |

### 3. 前端组件 (SignalListener.tsx)

**位置**: `src/components/SignalListener.tsx`

**功能**:
- 使用 Supabase Realtime 订阅 `signals` 表
- 收到新信号时显示 Toast 通知
- 点击通知打开 TradePanel 并预填参数

### 4. 数据库表 (signals)

**位置**: `src/db/signals_table.sql`

**表结构**:
```sql
CREATE TABLE signals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  symbol TEXT NOT NULL,
  action TEXT NOT NULL,  -- BUY, SELL, RECLAIM_BUY, RECLAIM_SELL
  price DECIMAL(20, 5),
  sl DECIMAL(20, 5),
  tp DECIMAL(20, 5),
  status TEXT DEFAULT 'new',  -- new, viewed, executed, expired
  source TEXT DEFAULT 'mt5_indicator',
  raw_data JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 启用 Realtime
ALTER PUBLICATION supabase_realtime ADD TABLE signals;
```

## Docker 配置

### docker-compose.yml

```yaml
services:
  mt5:
    image: gmag11/metatrader5_vnc:latest
    volumes:
      # 指标文件
      - ../mql5:/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Indicators/AlphaOS
      # 共享信号目录
      - signal_data:/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files/AlphaOS/Signals

  bridge-api:
    environment:
      - SUPABASE_URL=${NEXT_PUBLIC_SUPABASE_URL}
      - SUPABASE_KEY=${NEXT_PUBLIC_SUPABASE_ANON_KEY}
      - SIGNAL_DIR=/app/signals
    volumes:
      # 挂载同一个共享卷
      - signal_data:/app/signals

volumes:
  signal_data:
```

### 权限配置

**重要**: 共享卷需要正确的权限，以便 MT5 进程 (以 `abc` 用户运行) 可以写入：

```bash
# 在服务器上执行
docker exec mt5-vnc chmod 777 "/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files/AlphaOS/Signals/"
```

## 部署步骤

### 1. 同步代码到服务器

```bash
./deploy_service.sh bridge-api
```

### 2. 在 MT5 中添加指标

1. 访问 MT5 VNC: `http://YOUR_SERVER:3000`
2. 打开导航器 (Ctrl+N)
3. 找到 `Indicators → AlphaOS → PivotTrendSignals`
4. 拖放到图表上
5. 配置参数 (可选)

### 3. 验证信号流程

**检查 Bridge 日志**:
```bash
ssh alphaos "docker logs -f bridge-api 2>&1 | grep -E 'Signal|signal'"
```

**预期输出**:
```
Starting signal watcher on /app/signals...
🔔 New Signal Received: {'symbol': 'XAUUSD', 'action': 'BUY', ...}
✅ Signal saved to DB
```

## 故障排查

### 问题 1: 信号文件没有生成

**检查**:
```bash
# 查看 MT5 日志
docker exec mt5-vnc tail -100 "/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Logs/$(date +%Y%m%d).log"
```

**可能原因**:
- 指标未正确添加到图表
- 市场条件不满足信号生成条件
- 文件权限问题

### 问题 2: Bridge 没有读取到信号

**检查**:
```bash
# 检查共享卷内容
docker exec bridge-api ls -la /app/signals/
docker exec mt5-vnc ls -la "/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files/AlphaOS/Signals/"
```

**可能原因**:
- Docker 卷没有正确共享
- 权限问题 (需要 chmod 777)

### 问题 3: 信号没有保存到数据库

**检查**:
```bash
# 检查 Supabase 环境变量
docker exec bridge-api env | grep -i supabase
```

**可能原因**:
- `SUPABASE_URL` 或 `SUPABASE_KEY` 未设置
- 数据库表不存在
- 网络连接问题

### 问题 4: 前端没有收到通知

**检查**:
- 浏览器控制台是否有 WebSocket 错误
- Supabase Realtime 是否启用
- `signals` 表是否添加到 `supabase_realtime` publication

## 信号类型说明

| 信号类型 | 说明 |
|----------|------|
| `BUY` | 标准买入信号 (EMA 金叉 + 过滤器确认) |
| `SELL` | 标准卖出信号 (EMA 死叉 + 过滤器确认) |
| `RECLAIM_BUY` | 价格回归 EMA 后的买入信号 |
| `RECLAIM_SELL` | 价格回归 EMA 后的卖出信号 |

## 指标参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `InpPivotPeriod` | 2 | Pivot 点周期 |
| `InpATRPeriod` | 14 | ATR 计算周期 |
| `InpEMALength1` | 6 | 短期 EMA 周期 |
| `InpEMALength2` | 24 | 长期 EMA 周期 |
| `InpFilterMode` | FILTER_BASIC | 过滤模式 |
| `InpTPMultiplier` | 1.5 | TP ATR 倍数 |
| `InpSLMultiplier` | 1.0 | SL ATR 倍数 |
| `InpUseHTFFilter` | true | 使用高时间框架过滤 |
| `InpHTFPeriod` | H1 | 高时间框架周期 |

---

**最后更新**: 2024-11-25
**版本**: 2.10

