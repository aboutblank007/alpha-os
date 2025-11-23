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
... (保持不变)

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

3. **客户端实现 (Mac/AlphaOS)**
   - [x] 开发 AlphaOS 中的 API 客户端
   - [x] 实现交易指令接口 (HTTP POST)
   - [x] 实现状态同步 (HTTP GET)

4. **验证与部署**
   - [x] 解决 Docker 构建问题 (pip 源, 架构兼容)
   - [x] 解决 VNC 显示问题
   - [x] **端到端测试通过** (Mac -> API -> MT5 EA 双向通信成功)

**最新进展 (2025-11-23):**
- ✅ 解决了复杂的 Docker 网络 MTU 问题。
- ✅ 通过配置公网域名解析解决了 Wine 环境下的 DNS 问题。
- ✅ 验证了指令下发和状态上报的全流程。

**技术架构确认:**
```
MT5 EA (Ubuntu/Wine) ←HTTP Polling→ FastAPI (Ubuntu/Docker) ←HTTP REST→ AlphaOS (Mac)
      |                                   |
api.lootool.cn:8000                 49.235.153.73:8000
```

### 优先级 3: 仪表盘 UI 优化 (✅ 已完成)
... (保持不变)

---

## 📝 待完成的其他功能
... (保持不变)

**最后更新：** 2025-11-23 07:30
