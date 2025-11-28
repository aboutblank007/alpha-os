# AlphaOS 数据采集指南 (Backtest Data Collection)

本文档详细说明了如何通过 MT5 回测功能收集高质量的特征数据，用于 LightGBM 模型训练。

## 1. 系统架构

数据采集流程如下：
1.  **MT5 策略测试器 (Strategy Tester)**：运行专用数据采集 EA (`PivotTrend_DataCollector.mq5`)。
2.  **Signal-First 机制**：
    *   **信号生成**：一旦检测到买卖信号，立即写入 `signals_[Symbol].json` (包含 30+ 个技术特征)。
    *   **结果追踪**：内置虚拟交易引擎跟踪订单，当触发 TP/SL 时，写入 `outcomes_[Symbol].json`。
3.  **数据合成**：后期通过 Python 脚本将 Signal 和 Outcome 按 `signal_id` 进行 Join，生成带标签的训练集。

---

## 2. 环境准备

### 2.1 软件要求
*   **MetaTrader 5 (MT5)**：建议版本 5.00 build 4000+。
*   **Docker (可选)**：如果运行在云端容器中。
*   **Python 3.10+**：用于数据清洗和模型训练。

### 2.2 关键文件
*   `trading-bridge/mql5/PivotTrend_DataCollector.mq5` (核心采集 EA)
*   `ai-engine/export_training_data.py` (数据导出与合并脚本)

---

## 3. 操作步骤 (核心流程)

### 第一步：MT5 设置 (关键！)

**注意：采集器必须作为 EA (Expert Advisor) 运行，而非 Indicator。**

1.  **编译 EA**：
    *   在 MetaEditor 中编译 `PivotTrend_DataCollector.mq5`。

2.  **配置策略测试器** (`Ctrl+R`)：
    *   **Mode**: 选择 **`Expert Advisor`** (不要选 Indicator)。
    *   **Expert**: 选择 `AlphaOS\PivotTrend_DataCollector.ex5`。
    *   **Optimization (优化)**: **必须选择 `Disabled`**。
        *   *原因：开启优化会导致文件写入临时 Agent 目录，难以定位且会被自动删除。*
    *   **Symbol**: 选择目标品种 (如 `XAUUSD`)。
    *   **Timeframe**: 推荐 `M5`。
    *   **Visual mode**: 可选。开启可直观验证，关闭则速度更快。

### 第二步：运行回测

点击 **Start** 开始回测。
*   观察 Journal 日志，应出现 `Signal: BUY...` 或 `Outcome: ...` 字样。
*   如果日志无报错，说明数据正在写入。

### 第三步：数据导出

#### 场景 A: 本地运行 MT5
文件通常位于：
*   Windows: `%APPDATA%\MetaQuotes\Terminal\...\MQL5\Files\AlphaOS\Signals\`
*   Mac (CrossOver): `Drive_C/Program Files/MetaTrader 5/MQL5/Files/AlphaOS/Signals/`

#### 场景 B: Docker 容器运行 (云端)
如果使用了 Agent 模式或路径偏移，可以使用以下命令“捞取”数据并导出：

```bash
# 1. 打包容器内数据
ssh alphaos "docker exec mt5-vnc tar -czf /config/.wine/drive_c/backtest_data.tar.gz -C '/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files/AlphaOS/Signals' ."

# 2. 复制到宿主机
ssh alphaos "docker cp mt5-vnc:/config/.wine/drive_c/backtest_data.tar.gz ~/backtest_data.tar.gz"

# 3. 下载到本地
scp alphaos:~/backtest_data.tar.gz ./
```

### 第四步：生成训练集 (CSV)

使用项目提供的脚本将分散的 JSON 文件合并为 CSV：

```bash
python3 ai-engine/export_training_data.py
```

输出文件 `training_data.csv` 即可直接用于 LightGBM 训练。

---

## 4. 常见问题排查 (Troubleshooting Log)

### Q: 回测时图表不显示指标，也没有文件生成？
**原因**: 错误地将采集器作为 **Indicator** 运行，或者代码逻辑依赖 `OnCalculate` 但在 EA 模式下未触发。
**解决**: 
1. 将 `.mq5` 代码中的 `#property indicator_chart_window` 移除。
2. 将 `OnCalculate` 改为 `OnTick`。
3. 在 Strategy Tester 中明确选择 **Expert Advisor** 模式。

### Q: 运行了很久，文件夹还是空的？
**原因**: 
1. **Optimization 开启了**：MT5 会使用临时 Agent 目录 (如 `Tester/Agent-127.0.0.1-3000/...`)，测试结束后可能自动清理。
   *   **Fix**: 设置 Optimization 为 Disabled。
2. **逻辑缺陷**：旧版代码只在 `OnTrade` 或平仓时写文件。如果没有平仓（或回测提前结束），数据就丢了。
   *   **Fix**: 采用 **Signal-First** 策略。信号产生即写入 `signals_xxx.json`；平仓时追加 `outcomes_xxx.json`。

### Q: 报错 `array out of range`？
**原因**: EA 模式下 `CopyBuffer` 需要显式管理数据请求，且初始加载时历史数据可能不足。
**解决**: 增加 `CopyBuffer` 的请求长度（如 300 根），并添加数组边界检查 (`if(idx >= size) return;`)。
