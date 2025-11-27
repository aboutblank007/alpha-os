# AlphaOS 数据库配置指南

## 📋 概述

本目录包含完整的数据库设置脚本，适配前端所有功能，包括最新的 **Phase 5 AI 架构**。

## 📁 核心文件

| 文件名 | 用途 | 适用场景 |
|-------|------|---------|
| **`FULL_SCHEMA_V2.sql`** | **完整数据库架构 (单一真理来源)** | 全新安装、重建数据库 |

> ⚠️ 注意：旧的 `OPTIMIZED_DATABASE_SETUP.sql` 和分散的 `src/db/*.sql` 文件已被整合并移除，请统一使用 `FULL_SCHEMA_V2.sql`。

## 🗄 数据表结构

### 1. 核心交易 (Core)
- **`accounts`**: 存储交易账户信息和余额。
- **`trades`**: 完整的交易记录，包含 MAE/MFE、持仓时间等分析字段。
- **`signals`**: 信号表，存储来自 MT5 或 AI 的原始信号。

### 2. 辅助功能 (Features)
- **`journal_notes`**: 交易日记，包含情绪评分和市场观察。
- **`user_preferences`**: 用户偏好设置（UI 主题、风险偏好等）。
- **`automation_rules`**: 自动化交易规则。
    - 新增字段: `ai_mode` (legacy/indicator_ai/pure_ai)
    - 新增字段: `ai_confidence_threshold` (0.0 - 1.0)

### 3. AI 训练 (AI Training)
- **`training_datasets`** (Phase 5 新增):
    - 存储信号触发时的“特征快照” (Feature Snapshot)。
    - 用于 Meta-Labeling 模型的离线训练。
    - 包含 `features` (JSONB), `market_context` (JSONB), `label` (Int)。

## 🚀 使用指南

### 全新安装（推荐）

1. **登录 Supabase 控制台** -> SQL Editor。
2. **复制并执行** `src/db/FULL_SCHEMA_V2.sql` 的全部内容。
3. 这将一次性创建所有表、索引、RLS 策略和触发器。

### 验证安装

执行以下 SQL 检查表是否创建成功：

```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_type = 'BASE TABLE';
```

应该看到：`accounts`, `trades`, `journal_notes`, `user_preferences`, `signals`, `automation_rules`, `training_datasets`。

## 📊 常用查询

### 查看 AI 训练数据积累情况
```sql
SELECT count(*) as total_samples, model_version 
FROM training_datasets 
GROUP BY model_version;
```

### 查看 AI 介入的信号
```sql
SELECT * FROM signals 
WHERE comment LIKE '%AI:%' 
ORDER BY created_at DESC;
```
