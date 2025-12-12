---
description: 基于确定的 Next.js 与 Python 微服务目录结构，生成深度技术文档
---

---
title: AlphaOS 全栈架构扫描与文档生成
description: 自动扫描 Next.js 前端、Python 交易桥、AI 引擎及 Supabase 数据库，检测代码冗余，并生成包含架构图的深度技术文档。
---

# AlphaOS 系统全景扫描与文档生成器

此工作流将穿透远程环境，分析高频交易系统的每一个环节，产出 `docs/ALPHAOS_ARCHITECTURE_V2.md` 并提供清理建议。

## 步骤 1: 基础设施与协议握手 (Infrastructure Handshake)
首先确认系统的“神经中枢”——通信协议。
- **操作**：
  1. 读取并对比 `src/proto/alphaos.proto` 和 `ai-engine/src/proto/alphaos.proto`。
  2. **分析点**：检查两者是否一致？如果不一致，标记为严重隐患。
  3. 提取所有 `service` 和 `message` 定义，建立“指令-动作”映射表。

## 步骤 2: 数据层透视 (Data Layer via MCP)
**必须使用 MCP 工具**，严禁猜测数据库结构。
- **前置检查**：确保本地端口 `54320` 可用（SSH 隧道）。
- **操作**：
  1. 调用 `supabase-remote` (SQL MCP) 工具。
  2. 查询核心表结构：`trades` (交易记录), `ai_decisions` (AI 决策), `market_data` (行情)。
  3. 获取 `auth.users` 的统计信息（非隐私数据），确认用户体系状态。

## 步骤 3: 核心逻辑回溯 (Logic Trace)
深入三大子系统分析业务流。
- **前端 (Web)**：
  - 扫描 `src/app/dashboard` 和 `src/lib/bridge-client.ts`，明确 UI 如何触发 gRPC 指令。
- **交易桥 (Bridge)**：
  - 读取 `trading-bridge/src/main.py` 和 `trading-bridge/mql5/AlphaOS_Executor.mq5`。
  - **关键点**：解释 Python 如何将信号转发给 MT5 终端。
- **AI 引擎 (Brain)**：
  - 分析 `ai-engine/src/models/` 下的 `dqn.py` (策略) 和 `online_lgbm.py` (预测)。
  - 确认训练数据的流向：CSV -> DataLoader -> Model -> Inference。

## 步骤 4: 代码健康与冗余检测 (Health & Cleanup Check)
基于文件列表进行静态分析，寻找可清理对象。
- **识别规则**：
  1. **幽灵脚本**：根目录下的 `*.py`（如 `check_status.py`, `inspect_features.py`）是否在 `package.json` 或 `supervisord.conf` 中被引用？
  2. **重复数据**：`ai_decisions.csv` 是否同时存在于根目录和 `ai-engine/` 目录？
  3. **废弃遗留**：检查 `_archive/` 和 `bridge_logs_full.txt` 的最后修改时间。
- **产出**：生成一份“清理建议清单”，**不执行删除**，只列出供人工确认。

## 步骤 5: 构建深度文档 (Documentation Synthesis)
将所有信息汇总写入 `docs/ALPHAOS_ARCHITECTURE_V2.md`。
- **必含章节**：
  1. **系统拓扑图 (Mermaid)**：绘制 `Next.js <-> gRPC Bridge <-> AI Engine` 及 `Bridge <-> MT5` 的完整链路。
  2. **接口定义**：基于 Proto 文件的详细 API 说明。
  3. **数据流向**：从行情数据进入 CSV 到 AI 发出 Signal 的全过程。
  4. **容器部署**：记录当前 Docker 端口映射（基于 `docker ps`）。
  5. **待优化项 (Action Items)**：基于步骤 4 的冗余检测结果。

## 步骤 6: 审计与痕迹 (Audit Log)
- **操作**：
  1. 写入文档文件。
  2. 强制更新 `docs/DEVELOPMENT_LOG.md`：
     `- [YYYY-MM-DD HH:mm] 自动生成: ALPHAOS_ARCHITECTURE_V2.md - 完成全栈扫描与冗余检测`

## 示例
User: /generate-full-docs
Agent Action: [SSH 读取 Proto] -> [MCP 查询 Schema] -> [扫描 Python 脚本] -> [生成 Markdown]
Agent Reply: "文档生成完毕！\n- 架构文档：docs/ALPHAOS_ARCHITECTURE_V2.md\n- 发现 5 个可清理文件，详情请见文档末尾的'健康检查'章节。"