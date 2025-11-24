# 代码清理与文档更新摘要

> 日期: 2025-11-24
> 版本: v2.0.0

---

## 📋 清理内容

### 1. 删除的功能模块

#### ❌ OANDA API 集成
**原因**: 已完全切换到 MT5 Bridge 获取实时价格和历史数据

**删除的文件和代码**:
- ✅ 删除 `src/lib/oanda.ts` 文件（不存在，无需删除）
- ✅ 清理 `src/env.ts` 中的 OANDA 环境变量定义
  - 移除: `OANDA_API_KEY`, `OANDA_ACCOUNT_ID`, `OANDA_ENVIRONMENT`
  - 保留: `TRADING_BRIDGE_API_URL` (MT5 Bridge)
- ✅ 重构 `src/app/api/prices/route.ts`
  - 移除 OANDA API 调用
  - 重命名函数: `convertFromOandaInstrument` → `normalizeSymbol`
  - 重命名函数: `convertToOandaInstrument` → `addUnderscoreToSymbol`
  - 更新注释: "映射 OANDA 周期到 MT5 周期" → "映射图表周期到 MT5 时间框架"
- ✅ 更新 `src/app/debug/page.tsx`
  - 移除 OANDA 状态显示
  - 添加 MT5 Bridge 状态显示

#### ❌ TradingView Chrome 扩展
**原因**: 已弃用，不再使用 TradingView 集成

**删除的内容**:
- ✅ 删除整个 `alpha-link-extension/` 目录
  - `background.js`
  - `content.js`
  - `manifest.json`
  - `popup.html`
  - `popup.js`
  - `README.md`

#### ❌ 废弃的 API 端点
- ✅ 删除 `src/app/api/test-env/` 目录（空目录）
- ✅ 删除 `src/app/api/sync/` 目录（空目录）

---

## 📝 更新的文件

### 核心文件

1. **`README.md`** ✅
   - 完全重写，反映当前架构
   - 详细的系统架构图
   - MT5 Bridge 集成说明
   - Docker 部署指南
   - 更新的功能列表
   - 故障排除指南

2. **`src/env.ts`** ✅
   - 移除所有 OANDA 相关配置
   - 精简为仅包含 Supabase 和 MT5 Bridge
   - 更清晰的环境变量验证

3. **`src/app/api/prices/route.ts`** ✅
   - 完全基于 MT5 Bridge
   - 移除所有 OANDA API 调用
   - 保留 Mock 降级机制
   - 重命名工具函数避免混淆

4. **`src/app/debug/page.tsx`** ✅
   - 更新环境变量检查逻辑
   - 显示 MT5 Bridge 状态而非 OANDA
   - 保持 Supabase 连接检查

---

## ✅ 验证结果

### 代码扫描
```bash
# OANDA 引用检查
grep -r "OANDA\|oanda" src/
# 结果: 0 个匹配 ✅

# TradingView 扩展引用检查
grep -r "alpha-link-extension" .
# 结果: 0 个匹配 ✅
```

### Linter 检查
```bash
# 检查所有修改的文件
# 结果: 无 linter 错误 ✅
```

---

## 📊 当前技术栈

### 数据源
- ✅ **MT5 Bridge**: 实时价格、历史K线、账户信息
- ✅ **Supabase**: 交易记录、笔记、用户设置
- ❌ ~~OANDA API~~ (已移除)
- ❌ ~~TradingView 扩展~~ (已移除)

### 前端框架
- Next.js 16.0.3 (App Router + Turbopack)
- React 19.2.0
- TypeScript 5.x
- Tailwind CSS 4.x

### 后端服务
- Python FastAPI (MT5 Bridge)
- Next.js API Routes
- Supabase PostgreSQL

### 容器化
- Docker + Docker Compose
- Wine (运行 MT5)
- noVNC (Web 远程桌面)

---

## 🔄 数据流（当前架构）

```
前端 UI
  ↓
Next.js API Routes (/api/prices, /api/bridge/*)
  ↓
Python FastAPI Bridge (http://api.lootool.cn:8000)
  ↓
ZeroMQ (REQ/REP)
  ↓
MT5 Expert Advisor (BridgeEA.mq5)
  ↓
经纪商服务器

同步 ↓
Supabase Database
  ↓ 实时订阅
前端 UI 更新
```

---

## 🎯 主要改进

### 1. 架构简化
- 移除了不再使用的 OANDA 集成层
- 统一数据源为 MT5 Bridge
- 减少了外部依赖

### 2. 代码清晰度
- 删除了混淆的函数名（`convertFromOandaInstrument`）
- 更新了注释和文档
- 移除了死代码和空目录

### 3. 文档完善
- 全新的 README.md，详细描述当前架构
- 清晰的部署指南
- 完整的故障排除部分

### 4. 维护性提升
- 减少了代码库大小
- 更少的配置项
- 更清晰的依赖关系

---

## 📦 环境变量（当前）

### `.env.local` 配置

```bash
# Supabase 数据库
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key

# MT5 Trading Bridge
TRADING_BRIDGE_API_URL=http://api.lootool.cn:8000
# 本地开发: http://localhost:8000
```

### 已移除的配置
```bash
# ❌ 不再需要
# OANDA_API_KEY=...
# OANDA_ACCOUNT_ID=...
# OANDA_ENVIRONMENT=practice
```

---

## 🚀 后续步骤

### 对于开发者
1. 拉取最新代码
2. 更新 `.env.local` 文件（移除 OANDA 配置）
3. 重新安装依赖: `npm install`
4. 重启开发服务器: `npm run dev`
5. 访问 `/debug` 验证配置

### 对于部署

#### 步骤 1: 备份当前环境
```bash
# 连接到服务器
ssh alphaos

# 备份当前 .env 文件
cd ~/alpha-os
cp .env.local .env.local.backup

# 备份数据库（在 Supabase 控制台导出）
```

#### 步骤 2: 更新前端代码
```bash
# 在本地执行部署脚本
./deploy_service.sh frontend
# 或者
./deploy_service.sh web

# 脚本会自动：
# 1. rsync 同步代码到服务器 ~/alpha-os
# 2. 复制 .env.local 到远程
# 3. 执行 docker-compose up -d --build web
# 4. 重启前端容器
```

#### 步骤 3: 更新 .env 配置
```bash
# SSH 到服务器
ssh alphaos

# 编辑环境变量
cd ~/alpha-os
vim .env.local

# 确保包含以下配置：
# NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
# NEXT_PUBLIC_SUPABASE_ANON_KEY=your_key
# TRADING_BRIDGE_API_URL=http://api.lootool.cn:8000

# 保存后重启前端容器
docker-compose restart web
```

#### 步骤 4: 验证 MT5 Bridge 运行状态
```bash
# 进入 Bridge 目录
cd ~/trading-bridge/docker

# 检查容器状态
docker-compose ps

# 应该看到：
# mt5-vnc      运行中
# bridge-api   运行中

# 查看 Bridge API 日志
docker-compose logs -f bridge-api

# 测试 Bridge API
curl http://api.lootool.cn:8000/status

# 如果 Bridge 未运行，重启：
docker-compose up -d --build
```

#### 步骤 5: 更新数据库（如需要）
```bash
# 登录 Supabase 控制台
# https://app.supabase.com

# 进入 SQL Editor
# 执行 OPTIMIZED_DATABASE_SETUP.sql（如果是全新安装）

# 或者使用 Supabase CLI
supabase db push
```

#### 步骤 6: 验证部署
```bash
# 1. 检查前端容器日志
cd ~/alpha-os
docker-compose logs -f web

# 2. 访问前端页面
# http://49.235.153.73:3001

# 3. 测试关键功能
# - 访问 /debug 页面
# - 检查 Supabase 连接
# - 检查 MT5 Bridge 连接
# - 测试市场行情显示
# - 测试交易执行

# 4. 检查浏览器控制台
# 按 F12 → Console
# 确认无错误信息
```

#### 步骤 7: 监控与回滚
```bash
# 实时监控日志
cd ~/alpha-os
docker-compose logs -f

# 如果出现问题，回滚：
# 1. 恢复旧的 .env
cp .env.local.backup .env.local

# 2. 回滚代码（如果保留了 git）
git log --oneline -5
git reset --hard <commit-hash>

# 3. 重新构建
docker-compose up -d --build web

# 4. 或者恢复数据库快照
# 在 Supabase 控制台 → Database → Backups
```

#### 常见部署问题

**问题 1: 环境变量未生效**
```bash
# 解决方法
cd ~/alpha-os
docker-compose down
docker-compose up -d --build web
# 必须完全停止再重启
```

**问题 2: Bridge 连接失败**
```bash
# 检查 Bridge API 是否运行
curl http://api.lootool.cn:8000/health

# 检查防火墙
sudo ufw status

# 确保 8000 端口开放
sudo ufw allow 8000
```

**问题 3: 数据库连接失败**
```bash
# 检查 Supabase 项目状态
# 登录 https://app.supabase.com
# 确认项目未暂停

# 测试连接
curl -X GET 'https://your-project.supabase.co/rest/v1/trades?select=*&limit=1' \
  -H "apikey: your_anon_key"
```

**问题 4: Docker 磁盘空间不足**
```bash
# 清理未使用的镜像和容器
docker system prune -a

# 查看磁盘使用
df -h
docker system df
```

#### 部署检查清单

- [ ] 代码已同步到服务器
- [ ] .env.local 配置正确
- [ ] 前端容器运行正常
- [ ] Bridge API 运行正常
- [ ] MT5 容器运行正常
- [ ] 数据库连接成功
- [ ] /debug 页面显示全部 ✅
- [ ] 市场行情显示正常
- [ ] 持仓订单显示正常
- [ ] 交易执行功能正常
- [ ] 无浏览器控制台错误

---

## 📌 重要提示

1. **不要尝试配置 OANDA**
   - 系统不再支持 OANDA API
   - 所有价格数据来自 MT5 Bridge

2. **Chrome 扩展已弃用**
   - 不要尝试使用 TradingView 扩展
   - 所有交易通过前端 UI 执行

3. **环境变量已简化**
   - 只需配置 Supabase 和 MT5 Bridge URL
   - 移除旧的环境变量配置

4. **文档已更新**
   - README.md 反映当前架构
   - 数据库文档：`DATABASE_README.md`
   - MT5 同步文档：`DATABASE_MT5_SYNC_UPDATE.md`

---

## ✅ 清理完成确认

- [x] 所有 OANDA 代码已移除
- [x] TradingView 扩展目录已删除
- [x] 环境变量配置已简化
- [x] API 端点已清理
- [x] 文档已更新
- [x] 代码无 linter 错误
- [x] 无遗留引用

**状态**: ✅ 代码清理完成，可以投入使用

---

**生成日期**: 2025-11-24  
**版本**: v2.0.0  
**维护者**: AlphaOS Team

