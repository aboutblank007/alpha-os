# AI 变更日志 (AI_CHANGELOG.md)

本文档记录了 AI 对项目进行的每一次修改、优化和架构调整。

## 变更记录格式
- **日期**: YYYY-MM-DD
- **模块**: [Quantum Engine / Frontend / Docs / Risk]
- **变更描述**: 简述修改内容
- **影响范围**: 受影响的文件及功能

---

## 2025-12-24 (Quantum Risk & Model Standardization)
- **模块**: [Risk / Project Config / Models]
- **变更描述**: 
    - **风险引擎重构**: 严格对齐 `docs/XAUUSD_Quantum_Strategic_Research.md`，将 `RiskEngine` 的主决策逻辑切换为 **量子元标记 (Quantum Meta-Labeling)**，原 XGBoost 模型转为影子模式运行。
    - **模型管理标准化**: 在 `.rules` 中新增“模型管理与命名规范”，强制执行 `{Symbol}_{Architecture}_{Version}` 命名规则，并明确生命周期管理准则。
    - **模型训练与优化**: 使用最新 54MB `QuantumNet` 数据集对 XGBoost 元标记模型进行了全量重训练，校验集 AUC 提升至 **0.90185**。
    - **系统精简与清理**: 删除了 `xau_light`, `xau` 遗留目录及旧备份，并修改 `launch.sh` 剥离自动回退逻辑，确保生产环境加载的唯一确定性。
- **影响范围**: 
    - `quantum-engine/qlink/risk_engine.py`
    - `quantum-engine/qlink/launch.sh`
    - `.rules`
    - `quantum-engine/models/xau_v2_alpha101/`
    - `quantum-engine/scripts/train_xgb_meta_v2.py` (New)

---

## 2025-12-24 (Frontend Implementation)
- **模块**: [Frontend / Docs]
- **变更描述**: 
    - 根据 `docs/交易系统前端功能设计.MD` 完成核心可视化组件的生产级实现。
    - 新增 **深度分析面板 (Deep Analysis)**: 实现基于 Canvas 的 `OrderBookHeatmap`，并集成至 Dashboard。
    - 新增 **量子遥测 (Quantum Telemetry)**: 实现 `GradientNormChart` 用于监测贫瘠高原 (Barren Plateau) 及置信度分布。
    - 新增 **智能交易日记 (Forensic Journal)**: 实现交易回放 (Forensic Replay) 功能，支持查看历史时刻的市场深度快照与 AI 决策归因。
    - 更新设计文档，添加系统实现状态备忘录。
- **影响范围**: 
    - `src/components/charts/OrderBookHeatmap.tsx` (New)
    - `src/components/charts/GradientNormChart.tsx` (New)
    - `src/app/(main)/dashboard/page.tsx`
    - `src/app/(main)/ai/page.tsx`
    - `src/app/(main)/journal/page.tsx`
    - `docs/交易系统前端功能设计.MD`

## 2025-12-24 (初始化)
- **模块**: [Project Config / Rules]
- **变更描述**: 
    - 更新 `.rules` 规则文件，细化了 M2 Pro 架构下的项目结构映射、技术约束（float64, adjoint 微分）及各模块编码准则。
    - 创建了 `quantum-engine/logs` 目录并配置了 `alpha_engine.py`, `risk_engine.py`, `api_gateway.py` 的自动化文件日志记录。
    - 创建了本 `AI_CHANGELOG.md` 文档，并将其纳入 `.rules` 强制更新流程。
- **影响范围**: 
    - `.rules`
    - `quantum-engine/logs/`
    - `quantum-engine/qlink/alpha_engine.py`
    - `quantum-engine/qlink/risk_engine.py`
    - `quantum-engine/qlink/api_gateway.py`
    - `AI_CHANGELOG.md`

---

## [2025-12-24 12:45] 文档规范化与规则文件对齐
- Change: 
    - 对 `docs/XAUUSD_Quantum_Strategic_Research.md` 进行了全面的格式规范化（标题优化、LaTeX 公式标准、表格整理），并基于用户反馈完整还原了所有引用标记及页脚补充信息。
    - 同步更新了 `.rules` 和 `.cursor/rules/rules.mdc`，确立了以 **Scheme C (M2 Pro CPU)** 为核心的架构规范、模型命名规范及变更追踪协议。
- Files: 
    - `docs/XAUUSD_Quantum_Strategic_Research.md`
    - `.rules`
    - `.cursor/rules/rules.mdc`
    - `AI_CHANGELOG.md`
- Note: 确保了文档在 GitHub 渲染下的专业性与技术准确性，同时强化了后续开发的架构一致性约束。
