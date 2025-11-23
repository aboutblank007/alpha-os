# AlphaOS 项目优化指令

本文档根据 AlphaOS 项目现状，基于标准化优化模板生成，旨在指导项目的系统性重构与优化。

## 一、项目审查与分析阶段

### 1.1 现状评估
- **现有页面结构**：
  - **核心页**：Dashboard (`src/app/dashboard`), Journal (`src/app/journal`), Analytics (`src/app/analytics`)
  - **功能页**：Review (`src/app/review`), Settings (`src/app/settings`)
  - **开发/测试页**：UI Kit (`src/app/ui`), Debug (`src/app/debug`)
- **组件架构**：
  - **基础UI组件** (`src/components/ui`)：已包含 Button, Input, Modal, Select, Toast 等基础组件。
  - **业务组件** (`src/components/*`)：包含 MarketWatch, RecentTrades, EquityCurve, TradeControl 等交易相关复杂组件。
- **布局分析**：
  - 主要依赖 `AppShell` (`src/components/AppShell.tsx`) 作为全局布局容器。
  - 需检查 `AppShell` 在移动端的表现（Sidebar 响应式行为）。

### 1.2 用户场景分析
- **核心用户群体**：专业交易员、量化分析师。
- **主要设备**：
  - **桌面端 (Desktop)**：高频交易、图表分析、多屏监控（主要场景）。
  - **移动端 (Mobile)**：账户监控、紧急平仓、查看日志（辅助场景）。
- **页面优先级**：
  1. **Dashboard** (实时行情与订单管理) - P0
  2. **Journal** (复盘与日志) - P1
  3. **Analytics** (绩效分析) - P1

---

## 二、响应式布局全面优化

### 2.1 断点系统重构
- **Tailwind v4 适配**：利用 `src/app/globals.css` 中的 `@theme` 指令直接定义断点（如需自定义）。保持默认 Tailwind 断点系统：
  - `sm`: 640px
  - `md`: 768px
  - `lg`: 1024px
  - `xl`: 1280px
  - `2xl`: 1536px
- **超宽屏优化**：针对多屏交易环境，在 `2xl` 以上增加 `3xl` (1920px+) 断点支持，利用 `max-w-[screen-3xl]` 约束内容。

### 2.2 网格与布局系统
- **Dashboard 网格化**：使用 `CSS Grid` 重构 Dashboard 布局，支持拖拽排序 (`@dnd-kit` 已安装)。
- **间距标准化**：严格执行 `p-2`, `p-4`, `gap-4` 等标准间距，移除硬编码的像素值。

### 2.3 组件级响应式
- **AppShell (导航栏)**：
  - **桌面**：侧边栏固定展开。
  - **移动/平板**：侧边栏收起为汉堡菜单/抽屉模式，顶部增加移动端 Header。
- **MarketWatch (行情)**：
  - **桌面**：详细列表模式，显示买/卖价、点差、涨跌幅。
  - **移动**：紧凑卡片模式，仅显示最新价和涨跌颜色。
- **Charts (图表)**：
  - **桌面**：全功能 TradingView 图表。
  - **移动**：简化版图表或隐藏非必要指标，优先保证加载速度。

---

## 三、设计系统标准化

### 3.1 颜色系统规范 (基于 Tailwind v4)
- **当前状态**：`globals.css` 已定义 `--color-background`, `--color-accent-primary` 等变量。
- **优化动作**：
  - 确保所有自定义颜色支持 **Tailwind 透明度修改符** (如 `bg-accent-primary/50`)。
  - 完善语义化命名：
    - `bg-surface-glass` -> 玻璃态背景
    - `text-gradient` -> 渐变文字
  - 检查深色模式对比度，确保关键数据（红/绿涨跌）在深色背景下清晰可见。

### 3.2 字体排版系统
- **字体统一**：继续使用 `Inter` (已配置)，确保数字字体（价格跳动）使用等宽字体 (`font-mono`) 以避免跳动抖动。
- **字号规范**：
  - `text-xs`/`text-sm`：用于次要数据、标签。
  - `text-base`：正文、普通数值。
  - `text-lg`/`text-xl`：关键价格大字显示。

### 3.3 间距与尺寸标准
- **容器圆角**：统一使用 `rounded-xl` 或 `rounded-2xl` (符合现代 SaaS 风格)，与 `globals.css` 中的玻璃态风格匹配。
- **边框风格**：统一使用 `border-white/10` 或 `border-glass-border` 实现细腻边框。

---

## 四、组件优化与重构

### 4.1 组件拆分原则
- **Refactor Target**: `TradeControl.tsx` 和 `MarketWatch.tsx` 可能包含过多逻辑。
- **Action**:
  - 将 `MarketWatch` 拆分为 `SymbolRow`, `SymbolCard`, `MarketHeader`。
  - 将 `TradeControl` 的状态逻辑抽离为 `useTradeLogic` hook。

### 4.2 通用组件库建设 (`src/components/ui`)
- **完善现有组件**：
  - **Button**: 增加 `loading` 状态支持。
  - **Modal**: 确保支持点击遮罩关闭、ESC 关闭。
  - **Input**: 增加 `prefix` / `suffix` 支持（用于显示 $ 符号或单位）。
- **新增组件**：
  - **DataGrid**: 封装统一的表格组件，支持排序、筛选（用于日志和订单列表）。
  - **StatusBadge**: 统一的盈亏/状态标签（Open, Pending, Closed, Profit, Loss）。

### 4.3 表单组件优化
- **统一验证**：结合 `zod` (已安装) 和 React Hook Form (建议引入) 进行表单验证。
- **样式统一**：Input 和 Select 组件应共享相同的 `h-10`, `px-3`, `rounded-lg`, `bg-surface-glass` 样式。

---

## 五、交互体验提升

### 5.1 动画与过渡
- **现有配置**：`tailwind.config.ts` 中已定义 `fade-in`, `slide-up`。
- **优化**：
  - 订单提交成功时添加微交互动画。
  - 价格变动时背景色闪烁（绿/红）提示。
  - 页面切换使用 `template.tsx` 实现平滑过渡。

### 5.2 加载状态设计
- **全局 Loading**：使用 `Skeleton` 组件替换简单的 `Spinner`，特别是在 Dashboard 数据加载时。
- **局部反馈**：按钮点击后立即进入 `disabled` + `loading` 状态，防止重复提交。

### 5.3 错误处理与反馈
- **Toast 系统**：统一使用 `sonner` 或现有 `Toast` 组件显示 API 错误、交易成功/失败信息。
- **网络状态**：在 Header 显示 WebSocket 连接状态（已有的 `useBridgeStatus`），断开时显示醒目警告。

---

## 六、性能优化

### 6.1 渲染性能
- **行情数据优化**：
  - `MarketWatch` 中的价格更新频率极高。使用 `React.memo` 包装行组件。
  - 避免在父组件直接传递大对象，仅传递 Symbol ID，子组件通过 Context 或 Store 订阅具体数据。
- **虚拟滚动**：
  - 交易历史 (`RecentTrades`) 可能有上千条记录，**必须**引入 `react-window` 或 `react-virtuoso`。

### 6.2 资源优化
- **动态导入**：
  - `TradingViewChart` 库较大，应使用 `next/dynamic` 进行懒加载 (`ssr: false`)。
  - 模态框内容组件懒加载。

---

## 七、可访问性 (A11y) 完善

### 7.1 键盘导航
- **交易快捷键**：为专业用户实现键盘快捷键（如 `F9` 下单, `Esc` 取消选择）。
- **Tab 顺序**：确保表单 Input -> Quantity -> Buy/Sell 按钮的 Tab 顺序符合逻辑。

### 7.2 语义化 HTML
- 侧边栏使用 `<nav>` 或 `<aside>`。
- 统计卡片数据使用 `<dl>`, `<dt>`, `<dd>` 结构。

---

## 八、移动端特别优化

### 8.1 触摸优化
- **下单按钮**：移动端下单按钮高度至少 `48px`，且置于屏幕底部易操作区域。
- **图表交互**：禁用移动端图表内的默认页面滚动（防止手势冲突）。

### 8.2 视口设置
- 确保 `viewport` meta 标签包含 `maximum-scale=1, user-scalable=no` (针对交易应用，防止误触缩放)。

---

## 九、主题与暗黑模式

### 9.1 主题系统
- **默认深色**：鉴于 `globals.css` 的设定，AlphaOS 默认为深色/午夜蓝主题。
- **一致性检查**：确保所有新组件不使用硬编码的 `#000` 或 `#fff`，而是使用 `bg-background`, `text-foreground`。

---

## 十、代码规范与维护性

### 10.1 Tailwind 最佳实践
- 排序：使用 `prettier-plugin-tailwindcss` 自动排序类名。
- 复用：将常用按钮样式提取为 `.btn-primary` (在 `@layer components` 中定义) 或封装为 React 组件。

### 10.2 组件目录结构
- 保持 `src/components/[module]/[Component].tsx` 结构。
- 每个复杂组件文件夹下应包含 `index.ts` 导出。

---

## 十一、交付清单 (Checklist)

- [ ] **Layout**: AppShell 响应式适配（移动端抽屉菜单）。
- [ ] **UI Kit**: 完善 `src/components/ui`，补充 DataGrid, StatusBadge。
- [ ] **Dashboard**: 实现网格布局拖拽，应用虚拟滚动。
- [ ] **Perf**: 行情组件 `React.memo` 优化，图表组件懒加载。
- [ ] **Theme**: 检查所有硬编码颜色，替换为 CSS 变量。
- [ ] **Testing**: 关键交易流程（下单、平仓）端到端测试。
