# Antigravity 全局智能体规则 (Global Agent Rules) v2.1

## 0. 熵减法则 (Anti-Entropy / DRY Protocol) 【最高优先级】

**核心指令：在编写每一行新代码之前，你必须通过工具检索项目现有的“资产”。严禁重复造轮子。**

### A. 资产检索路径 (Asset Map)

在实现功能前，必须先强制检查以下目录：

1. **UI 组件**：必须优先复用 `src/components/ui/` 下的 18+ 个基础组件 (Button, Modal, Input, Toast 等)。
   - *禁止*：手写 `<button className="...">` 或新造模态框。
   - *必须*：`import { Button } from '@/components/ui/Button'`。
2. **工具函数**：检查 `src/lib/`。
   - 格式化/辅助：`utils.ts`
   - 指标计算：`indicators.ts`
   - 接口调用：`bridge-client.ts`, `supabase.ts`
3. **状态管理**：
   - 检查 `src/store/` (useSignalStore, useTradeStore 等)。
   - *禁止*：在页面组件中大量堆砌 `useState` 来管理全局状态。
4. **AI/后端逻辑**：
   - 特征工程：`ai-engine/src/features*.py`
   - 模型逻辑：`ai-engine/src/models/`

### B. 协议强制同步 (Protocol Consistency)

- **单一真理来源**：仅允许修改 `src/proto/alphaos.proto`。
- **同步动作**：修改源文件后，**必须**执行 `./scripts/sync_proto.sh`。
- **禁止事项**：严禁直接编辑 `ai-engine` 或 `trading-bridge` 下的 `.proto` 或 `_pb2.py` 文件，这些必须由脚本生成。

### C. 交易桥架构强制 (Trading Bridge Constitution)

- **核心原则：瘦入口 (Thin Entry)**
  - `main.py` 仅用于启动 Server 和依赖注入，**严禁**包含任何业务逻辑或算法。
- **分层架构 (Layer Enforcement)**：
  - **下单/风控**：必须写入 `src/core/` (`executor.py`, `risk_manager.py`)。
  - **流程调度**：必须写入 `src/managers/`。
  - **外部监听**：必须写入 `src/services/`。
- **MQL5 限制**：
  - 涉及 MetaTrader 交互的修改，优先修改 Python 侧的 gRPC 逻辑。如必须修改 `.mq5`，需指导用户手动在 MetaEditor 编译。

---

## 1. 语言与交互原则 (Core Protocol)

- **强制中文回复**：Agent 的所有思考、解释、代码注释及交互必须全程使用简体中文。
- **工具优先**：禁止臆测。凡涉及文件系统（SSH）、数据库（MCP）或代码分析的操作，必须调用工具获取真实证据。

## 2. 远程基础设施上下文 (Remote Infrastructure)

本项目运行于远程主机 macOS，Agent 需通过 SSH 管理容器。

- **宿主机别名**：macOS
- **容器管理指令**：
  - 查看状态：`ssh macOS 'docker ps'`
  - 核心服务映射：
    - API 核心：`bridge-api` (Port 8000/50050)
    - AI 引擎：`ai-engine` (Port 50051)
    - 前端界面：`alpha-os-web` (Port 3001)
- **项目路径映射 (Crucial)**：
  - 前端根目录：`./src` (Next.js App Router)
  - AI 引擎：`./ai-engine` (Python/PyTorch)
  - 交易桥：`./trading-bridge` (Python/MQL5)

## 3. Supabase 自动化连接与 MCP 规范

数据库位于远程容器 `supabase-db`。

- **自动连接策略**：
  在调用任何 SQL MCP 工具前，必须先执行：
  `lsof -i :54320 >/dev/null 2>&1 || ssh -L 54320:127.0.0.1:5432 macOS -N -f`
- **MCP 工具使用规范**：
  1. 工具名称：优先使用 `supabase-remote-db`。
  2. 数据安全：严禁在未确认 Schema 的情况下执行 DELETE/UPDATE。

## 4. 严格文档与日志同步协议 (Documentation & Logging)

为防止项目漂移，任何变更必须记录。

### A. 全局流水账 (Universal Operation Log)

- **目标文件**：`docs/DEVELOPMENT_LOG.md`
- **触发条件**：任何代码修改、文件创建、Schema 变更。
- **格式**：`- [YYYY-MM-DD HH:mm] [<Type>]: <File> - <Action>`

### B. 架构文档同步

- **触发条件**：当变更涉及 数据库 Schema、Proto 接口、容器拓扑 或 AI 模型逻辑 时。
- **执行动作**：必须同步更新 `docs/ALPHAOS_ARCHITECTURE_V2.md` 中的 Mermaid 图表或文字说明，保持文档与代码现状一致。

## 5. 自动故障诊断机制 (Auto-Debug Protocol)

- **触发条件**：SSH 失败、Docker 报错、API 超时。
- **自动响应动作**：
  1. **禁止直接报错结束**。
  2. 立即查询日志：`ssh macOS 'docker logs --tail 50 [容器名称]'`
  3. 分析原因并给出修复建议。

## 6. 代码架构宪法 (Code Architecture Constitution)

针对 AlphaOS 项目结构的具体编程约束：

### A. 前端 (Next.js/React)

- **Server Components 优先**：默认在 `src/app` 下的 Page 使用 Server Components 获取数据，Client Components 仅用于交互组件。
- **路径别名**：必须使用 `@/` (如 `@/components/ui/Button`)，**严禁**使用 `../../components` 这种相对路径。
- **Tailwind CSS**：严禁写传统的 `.css` 文件（`globals.css` 除外），必须使用 Tailwind 类名。

### B. AI Engine (Python)

- **模块化**：不要把所有逻辑塞进 `main.py` 或 `train.py`。
- **路径**：新的模型架构必须放在 `ai-engine/src/models/`，新的数据处理器放在 `ai-engine/src/`。

## 7. 工业级代码标准 (Industrial Grade Standards)

**核心指令：默认假设所有外部依赖（API、数据库、网络）都会失败。代码必须具备“反脆弱性”。**

### A. 严禁裸奔 (No Silent Failures)

- **禁止 `print()`**：Python 代码严禁使用 `print()` 调试。
  - *必须*：使用 `loguru` 或标准 `logging` 库。
  - *格式*：`logger.info("Strategy started", symbol="BTCUSD")`（必须包含上下文变量）。
- **防御性编程**：
  - 凡是涉及网络请求、文件读写、数据库操作，**必须**使用 `try/except` 包裹。
  - **禁止**使用空的 `except: pass`。必须捕获具体异常并记录 Error Log。
  - **重试机制**：对不稳定的外部调用（如 MT5 桥接、OpenAI API），必须引入重试逻辑（推荐 Python `tenacity` 库）。

### B. 类型与配置安全 (Type & Config Safety)

- **拒绝硬编码 (No Magic Numbers)**：
  - 禁止在代码中写死 `0.01` (Lot Size) 或 `"API_KEY"`。
  - *必须*：所有配置项移入 `.env` 或 `config.py`，并通过 Pydantic (`BaseSettings`) 加载。
- **强制类型提示 (Strict Typing)**：
  - Python 函数必须包含 Type Hints (如 `def calc(price: float) -> float:`)。
  - 复杂数据结构必须使用 `Pydantic` 模型定义，而不是随意的 Dict。

### C. 生产级交付 (Definition of Done)

代码在输出前必须通过以下自我质问：

1. **重启测试**：如果服务崩溃重启，这个状态会丢失吗？（如果会，需持久化到 Redis/DB）。
2. **错误测试**：如果 API 超时，整个程序会崩溃退出吗？（如果是，请改为降级处理）。
3. **日志测试**：出问题时，我能只看日志就定位原因吗？