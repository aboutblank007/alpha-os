# 数据库优化 & MT5 净资产同步更新

## 📋 更新内容

### 1. 数据库优化 ✅
- **新建脚本**: `OPTIMIZED_DATABASE_SETUP.sql`
  - 全新的数据库结构，支持所有前端页面功能
  - 包含：账户表、交易表、笔记表、用户设置表
  - 新增字段：策略名称、标签、情绪评分、风险回报比、MAE/MFE、持仓时间等
  - 优化的索引和视图（每日统计、品种表现、策略分析等）

- **迁移脚本**: `MIGRATION_TO_V2.sql`
  - 从旧数据库安全迁移到新结构
  - 保留所有现有数据

### 2. MT5 净资产实时同步 ✅
**前端不再计算净资产，直接使用 MT5 实时数据**

#### 修改的文件：

**`src/hooks/useBridgeStatus.ts`**
- 新增 `account` 字段到 `BridgeStatus` 接口
- 包含：`balance`（余额）、`equity`（净资产）、`margin`（已用保证金）、`free_margin`（可用保证金）

**`src/components/AppShell.tsx`**
- 移除旧的 `/api/account/balance` API 调用
- 使用 `useBridgeStatus()` 获取 MT5 实时账户信息
- 左上角显示：
  - 主值：MT5 净资产 (equity)
  - 副值：浮动盈亏 (equity - balance)

**`src/app/dashboard/page.tsx`**
- 仪表盘顶部卡片显示 MT5 净资产
- 副标题显示账户余额

## 🚀 部署步骤

### 选项 A：全新安装（无数据）
```sql
-- 在 Supabase SQL Editor 中执行
-- 复制 OPTIMIZED_DATABASE_SETUP.sql 全部内容并执行
```

### 选项 B：迁移现有数据
```sql
-- 在 Supabase SQL Editor 中执行
-- 复制 MIGRATION_TO_V2.sql 全部内容并执行
```

### 更新前端代码
```bash
# 本地测试
npm run dev

# 部署到云端
./deploy_service.sh frontend
```

## ✨ 功能特点

### 实时 MT5 数据
- ✅ 净资产实时更新（每秒）
- ✅ 浮动盈亏即时显示
- ✅ 无需手动刷新
- ✅ 数据来源：MT5 账户信息

### 数据库优化
- ✅ 支持仪表盘所有统计
- ✅ 支持交易日志详细记录
- ✅ 支持数据分析图表
- ✅ 支持用户设置保存
- ✅ 高性能索引
- ✅ 视图简化查询

## 📊 显示位置

### 1. 左上角（AppShell）
```
净资产 (MT5)
$10,245.67 +12.34
```
- 主值：MT5 equity
- 绿色/红色：浮动盈亏

### 2. 仪表盘（Dashboard）
```
净资产
$10,245.67
余额: $10,233.33
```
- 主值：MT5 equity
- 副标题：MT5 balance

## 🔄 数据流

```
MT5 账户
   ↓ (每秒)
BridgeEA.mq5 → Python FastAPI
   ↓ (每秒)
/api/bridge/status
   ↓ (轮询 1秒)
useBridgeStatus Hook
   ↓
AppShell & Dashboard
```

## 🎯 优势

1. **实时性**：数据每秒更新，无延迟
2. **准确性**：直接来自 MT5，无计算误差
3. **性能**：不依赖数据库计算，减轻服务器压力
4. **一致性**：与 MT5 终端显示完全一致

## 📝 注意事项

- MT5 连接断开时显示 `---`
- 确保 `bridge-api` 服务正常运行
- 确保 `.env` 中配置了正确的 `BRIDGE_API_URL`

