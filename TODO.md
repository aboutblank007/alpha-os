# 📋 待办事项

## 🐛 已解决的问题

### 2025-11-21: OANDA API 网络连接问题

**问题描述：**
- 错误：`fetch failed` (HTTP 500)
- 无法连接到 OANDA API 服务器

**根本原因：**
- 网络连接问题（地理限制，中国大陆无法直接访问 OANDA API）

**解决方案：**
1. ✅ **推荐**：使用模拟数据模式
   - 注释掉 `.env.local` 中的 OANDA 配置
   - 系统自动切换到模拟价格
   - 所有功能完全可用

2. ⚙️ **可选**：使用 VPN 访问真实价格
   - 连接美国/新加坡节点
   - 保持 OANDA 配置

**已实现的改进：**
- ✅ 增强错误诊断（显示详细网络错误）
- ✅ 添加超时控制（10秒）
- ✅ 智能降级机制（自动切换模拟数据）
- ✅ 创建诊断工具（`/debug` 页面）

### 2025-11-21: Supabase 配置问题

**问题描述：**
- 错误：`Forbidden use of secret API key in browser`

**根本原因：**
- 使用了 service_role key 而不是 anon key

**解决方案：**
- ✅ 使用 Supabase 控制台中的 anon (public) key
- ✅ 更新 `.env.local` 中的 `NEXT_PUBLIC_SUPABASE_ANON_KEY`

---

## 🆕 新需求

### 优先级 1: 在仪表盘添加主流货币对K线图和技术指标

**需求描述：**
在现有的 Dashboard 页面中添加：
1. 主流货币对的 K线图（蜡烛图）
2. 用户使用的技术指标

**技术实现计划：**

#### 已有资源：
- ✅ TradingView Lightweight Charts 库 (v4.1.0)
- ✅ `/api/prices` POST 端点（可获取历史K线）
- ✅ `TradingViewChart` 组件（已实现但未使用）
- ✅ OANDA API 集成（支持获取蜡烛图数据）

#### 需要实现：

1. **在 Dashboard 添加图表组件**
   - [ ] 确定图表展示位置（布局设计）
   - [ ] 集成 `TradingViewChart` 组件到 Dashboard
   - [ ] 实现多品种切换功能

2. **添加技术指标**
   - [ ] 确认用户使用的具体指标（待用户提供）
     - EMA（指数移动平均线）？
     - MACD（异同移动平均线）？
     - RSI（相对强弱指标）？
     - 布林带（Bollinger Bands）？
     - 其他？
   - [ ] 在图表组件中集成技术指标
   - [ ] 实现指标参数配置

3. **货币对选择**
   - [ ] 确认需要展示的货币对（待用户提供）
     - EURUSD（欧元/美元）？
     - USDJPY（美元/日元）？
     - GBPUSD（英镑/美元）？
     - XAUUSD（黄金）？
     - 其他？
   - [ ] 实现货币对切换 UI
   - [ ] 可能支持多图表同时展示

4. **数据优化**
   - [ ] 实现K线数据缓存
   - [ ] 优化数据加载性能
   - [ ] 实时价格更新集成

#### 待确认信息：

**请用户提供：**
1. 您通常使用哪些技术指标？
2. 需要在图表上展示哪些货币对？
3. 图表应该放在 Dashboard 的什么位置？
4. 希望使用什么时间周期（1分钟、5分钟、15分钟、1小时等）？

#### 预计工作量：
- 基础图表集成：2-3小时
- 技术指标实现：每个指标 1-2小时
- UI/UX 优化：2小时
- 测试和调试：1-2小时

**总计：约 6-10 小时**

#### 参考资料：
- TradingView Lightweight Charts 文档：https://tradingview.github.io/lightweight-charts/
- 已有实现：`src/components/charts/TradingViewChart.tsx`
- API 端点：`src/app/api/prices/route.ts`

---

## 📝 待完成的其他功能

### 已计划但未实现：

- [ ] WebSocket 实时价格推送（替代当前的轮询）
- [ ] 价格预警功能
- [ ] MAE/MFE 分析
- [ ] 回撤分析
- [ ] 交易策略回测
- [ ] 移动端适配
- [ ] 多账户支持
- [ ] PDF 报告导出

---

## 🔧 技术债务

- [ ] 优化价格 API 的错误处理
- [ ] 添加更多的单元测试
- [ ] 改进文档结构（已整理，持续优化）
- [ ] 性能监控和优化

---

## 📚 文档更新

### 最近更新：
- ✅ 整理并合并多个分散的文档
- ✅ 创建主 README.md（包含所有核心信息）
- ✅ 创建 DEVELOPMENT.md（技术参考）
- ✅ 创建 QUICK_FIX.md（快速故障排除）
- ✅ 创建 DOCS_INDEX.md（文档导航）
- ✅ 创建 `/debug` 可视化诊断页面

### 当前文档结构：
- `README.md` - 主文档（用户指南、快速开始、故障排除）
- `DEVELOPMENT.md` - 开发文档（API、数据库、组件）
- `QUICK_FIX.md` - 快速修复指南
- `DOCS_INDEX.md` - 文档索引
- `PROJECT_CHANGELOG.md` - 完整变更日志
- `TODO.md` - 本文件（待办事项和需求跟踪）

---

**最后更新：** 2025-11-21  
**下次会议：** 待定（讨论K线图和技术指标的具体需求）

