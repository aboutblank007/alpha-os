## 🔧 服务器连接与容器管理

### 服务器连接方式

**连接命令：**
```bash
ssh alphaos
```

### Docker 容器管理

**项目路径：**
```bash
cd ~/trading-bridge/docker
```

**容器列表：**
- `mt5-vnc`: MT5 交易平台 (VNC: http://49.235.153.73:3000)
- `bridge-api`: API 服务 (http://api.lootool.cn:8000)

**前端服务：**
- `alpha-os-web`: 前端页面 (http://49.235.153.73:3001)
  - 部署路径: `~/alpha-os`
  - 独立容器部署，通过 `deploy_frontend.sh` 管理

**常用命令：**

```bash
# 1. 交易桥接服务 (MT5 + API)
cd ~/trading-bridge/docker
docker-compose ps
docker-compose logs -f
docker-compose restart

# 2. 前端服务
cd ~/alpha-os
docker-compose ps
docker-compose logs -f
docker-compose restart
```

### 文件同步与部署

**一键部署前端 (AlphaOS Web):**
```bash
./deploy_frontend.sh
```

**同步 MQL5 EA:**
```bash
scp trading-bridge/mql5/BridgeEA.mq5 alphaos:~/trading-bridge/mql5/
# 然后在 VNC 中重新编译 EA
```

**同步 Python Server:**
```bash
scp trading-bridge/src/main.py alphaos:~/trading-bridge/src/
# 然后重启 API 容器
ssh alphaos "cd ~/trading-bridge/docker && docker-compose restart bridge-api"
```

---

# 📋 待办事项

## 🐛 已解决的问题

### 2025-11-23: Ubuntu Docker 桥接网络通信问题
**问题描述：**
- MT5 EA 无法连接到 API 容器 (Error 5203, 1001)
- Docker 容器间 HTTP 请求卡死 (curl 挂起)
- 本地 Python 脚本无法获取 EA 状态

**根本原因：**
1. **MTU 问题**：云服务器网卡 MTU (1450) 小于 Docker 默认 MTU (1500)，导致 HTTP 大包丢弃。
2. **DNS 解析问题**：Wine 环境下 Docker 内部 DNS 解析不稳定。
3. **API 代码陈旧**：服务器端 Python 代码未及时同步，返回旧版 JSON 格式。

**解决方案：**
- ✅ **网络优化**：在 `docker-compose.yml` 中设置 `mtu: 1400`。
- ✅ **DNS 穿透**：配置公网域名 `api.lootool.cn` 指向服务器 IP，绕过 Docker 内部 DNS。
- ✅ **代码同步**：更新服务器端 `main.py` 和 `BridgeEA.mq5`。

---

## 🆕 新需求

### 优先级 1: 在仪表盘添加主流货币对K线图和技术指标 (✅ 已完成)
- [x] **集成 TradingView 轻量级图表库**: 支持缩放、平移和自定义数据。
- [x] **自定义数据源 (OANDA API)**: 实现了 `OandaAdapter`，通过 `getPricing` 和 `getCandles` 获取实时和历史数据。
- [x] **技术指标计算**: 集成了 `technicalindicators` 库，前端动态计算 SMA、EMA、RSI 和 MACD。
- [x] **多品种切换**: 支持 USDJPY, EURUSD, XAUUSD, BTCUSD。
- [x] **图表交互**:
    - [x] 动态切换时间周期 (1M, 5M, 1H, 4H, 1D)。
    - [x] 指标开关控制 (Checkbox)。
    - [x] 十字光标和实时价格标签。
- [x] **数据源迁移**: 
    - [x] 从 OANDA 迁移到 MT5 Bridge。
    - [x] 支持从 MT5 获取历史 K 线 (通过 `GET /history`)。
    - [x] 支持从 MT5 获取实时报价。

### 优先级 2: 跨平台交易桥接系统 (✅ 已完成)

**需求描述：**
搭建一个跨平台的交易桥接系统，通过 Docker 容器在 Ubuntu 服务器上运行 MT5 交易平台和桥接 API 服务，实现 Mac 本地系统与 MT5 之间的交易指令传递和状态同步。

**技术实现状态：**

#### 已完成工作：

1. **系统架构设计**
   - [x] 定义 Docker 容器结构 (Wine + MT5 + Python/FastAPI)
   - [x] 架构升级：从 ZeroMQ 迁移到 **HTTP Polling** (简化 Wine 兼容性)
   - [x] 网络架构：公网域名透传 + Docker MTU 优化

2. **服务端实现 (Ubuntu/Docker)**
   - [x] 创建 Wine + MT5 的 Dockerfile
   - [x] 开发 MT5 Expert Advisor (EA) (基于 WebRequest)
   - [x] 开发桥接 API 服务器 (Python/FastAPI)
   - [x] 配置 Docker Compose (双容器: mt5, bridge-api)
   - [x] **历史数据支持**: 升级 Server 支持 K 线数据查询与回传。

3. **客户端实现 (Mac/AlphaOS)**
   - [x] 开发 AlphaOS 中的 API 客户端
   - [x] 实现交易指令接口 (HTTP POST)
   - [x] 实现状态同步 (HTTP GET)
   - [x] **MT5 Client**: 封装 `getHistory` 和 `getStatus` 方法。

4. **验证与部署**
   - [x] 解决 Docker 构建问题 (pip 源, 架构兼容)
   - [x] 解决 VNC 显示问题
   - [x] **端到端测试通过** (Mac -> API -> MT5 EA 双向通信成功)

**最新进展 (2025-11-23):**
- ✅ 解决了复杂的 Docker 网络 MTU 问题。
- ✅ 通过配置公网域名解析解决了 Wine 环境下的 DNS 问题。
- ✅ 验证了指令下发和状态上报的全流程。
- ✅ **数据源切换**: 完成从 OANDA 到 MT5 的切换。

**技术架构确认:**
```
MT5 EA (Ubuntu/Wine) ←HTTP Polling→ FastAPI (Ubuntu/Docker) ←HTTP REST→ AlphaOS (Mac)
      |                                   |
api.lootool.cn:8000                 49.235.153.73:8000
```

### 优先级 3: 仪表盘 UI 优化 (✅ 已完成)

**已完成工作 (Phase 4: UI/UX Polish):**
- [x] **全局样式升级**: 采用 Deep Space 玻璃拟态主题，优化调色板和渐变。
- [x] **仪表盘重构**:
    - [x] 优化统计卡片 (Stat Cards) 视觉效果。
    - [x] 重构图表头部，移除冗余控件。
    - [x] 改进侧边栏和布局结构。
- [x] **市场监视面板 (Market Watch)**:
    - [x] 新增独立侧边栏面板。
    - [x] 显示 MT5 实时连接状态。
    - [x] 集成实时报价和快速交易按钮。
    - [x] 解决布局重叠问题。

---

### 优先级 4: 稳定性与自动化 (Phase 2 & 3) ✅ 已完成

**目标描述：**
执行优化计划的第二阶段（稳定性）和第三阶段（自动化）。

**主要任务：**

#### 阶段 2: 稳定性 (Stability)
- [x] **环境变量强校验**: 引入 `zod`，启动时验证 Supabase/OANDA 配置。
- [x] **Bridge 健康监控**: 集中化状态轮询 (`useBridgeStatus`)，监控连接延迟。
- [x] **代码重构**: 更新图表组件使用统一的状态 Hook。

#### 阶段 3: 自动化 (Automation)
- [x] **自动交易归档**: Bridge 服务集成 Supabase，实现下单即写入数据库 (无需 CSV)。
- [x] **基础设施更新**: Docker Compose 注入数据库凭证。
- [ ] **AI 复盘页面**: 搭建 `/review` 页面框架 (下一步计划)。

---

## 🔮 下一步计划 (Next Steps)

### 1. AI 复盘接口 (AI Review Interface)
- [ ] 实现 AI 分析逻辑，对交易记录进行智能评分和建议。
- [ ] 开发 `/review` 页面前端。

### 2. 移动端适配 (Mobile Optimization)
- [ ] 优化仪表盘在移动设备上的显示。
- [ ] 调整图表和表格的响应式布局。

### 3. OANDA 连接修复
- [ ] 解决本地开发环境连接 OANDA API 的 SSL/网络问题。

---

## 📝 待完成的其他功能

**最后更新：** 2025-11-23 21:00
