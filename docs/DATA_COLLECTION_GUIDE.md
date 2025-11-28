# AlphaOS 数据采集指南 (Backtest Data Collection)

本文档详细说明了如何通过 MT5 回测功能收集高质量的特征数据，用于 LightGBM 模型训练。

## 1. 系统架构

数据采集流程如下：
1.  **MT5 策略测试器**：运行专用数据采集指标 (`PivotTrend_DataCollector.mq5`)。
2.  **文件系统**：指标将每根 K 线的详细特征（Technical + Price Action）输出为 JSON 文件到指定目录。
3.  **Python Bridge**：后台运行的 Python 服务监听该目录，读取 JSON 文件。
4.  **Supabase**：Python Bridge 解析数据并存入 `training_signals` 表。

---

## 2. 环境准备

### 2.1 软件要求
*   **MetaTrader 5 (MT5)**：已安装并登录。
*   **Python 3.10+**：建议使用 Conda 环境。
*   **Supabase 项目**：已创建并包含 `training_signals` 表（参考 `src/db/data_collection.sql`）。

### 2.2 关键文件检查
确保以下文件已在项目中：
*   `trading-bridge/mql5/PivotTrend_DataCollector.mq5` (专用采集指标)
*   `trading-bridge/src/main.py` (桥接服务)
*   `ai-engine/src/alphaos_pb2.py` (gRPC 生成代码)

---

## 3. 操作步骤

### 第一步：MT5 设置

1.  **编译指标**：
    *   打开 MT5 MetaEditor。
    *   打开 `trading-bridge/mql5/PivotTrend_DataCollector.mq5`。
    *   点击 **Compile** (F7)。确保无报错。

2.  **配置策略测试器 (Strategy Tester)**：
    *   打开 MT5 终端，按 `Ctrl+R` 打开策略测试器。
    *   **Overview/Single** 模式。
    *   **Indicator**: 选择 `AlphaOS\PivotTrend_DataCollector.ex5`。
    *   **Symbol**: 选择你要采集的品种 (如 `XAUUSD`, `EURUSD`)。
    *   **Timeframe**: 选择目标时间周期 (如 `M5` 或 `M15`)。
    *   **Date**: 选择回测时间段 (建议最近 6-12 个月)。
    *   **Visual mode**: **必须关闭** (以加快速度)，或者开启以验证信号生成。

### 第二步：启动 Python Bridge

打开终端 (Terminal)，进入项目根目录。

1.  **配置环境变量**：
    请替换 `<YOUR_SUPABASE_URL>` 和 `<YOUR_SUPABASE_KEY>` 为实际值。
    
    **注意**：`SIGNAL_DIR` 必须指向 MT5 测试器生成文件的实际路径。在回测模式下，路径通常位于 MT5 数据文件夹的 `Tester` 目录下。
    
    *提示：您可以在 MT5 中点击 "File" -> "Open Data Folder" 来找到根路径。*

    ```bash
    # Supabase 配置
    export SUPABASE_URL="<YOUR_SUPABASE_URL>"
    export SUPABASE_KEY="<YOUR_SUPABASE_KEY>"

    # 信号文件目录配置
    # 对于 macOS (CrossOver/Parallels):
    # export SIGNAL_DIR="$HOME/Library/Application Support/com.neu.crossover/Bottles/MetaTrader 5/drive_c/Program Files/MetaTrader 5/MQL5/Files/AlphaOS/Signals"
    
    # 对于 Windows / 标准路径:
    # export SIGNAL_DIR="/path/to/MetaTrader5/MQL5/Files/AlphaOS/Signals"
    
    # 重要：如果是回测模式，MT5 可能会写入 Tester 目录，请检查 Data Folder/Tester/Agent-X/MQL5/Files/AlphaOS/Signals
    # 为了简单起见，您可以先用“实盘”模式（直接加载指标到图表）测试路径是否正确。
    export SIGNAL_DIR="/Users/xiao/Library/Application Support/com.neu.crossover/Bottles/MetaTrader 5/drive_c/Program Files/MetaTrader 5/MQL5/Files/AlphaOS/Signals"
    ```

2.  **设置 Python 路径**：
    因为代码需要跨目录引用 gRPC 模块，必须设置 `PYTHONPATH`。

    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/ai-engine/src:$(pwd)/trading-bridge/src
    ```

3.  **启动服务**：
    使用 `uvicorn` 启动 FastAPI 服务。

    ```bash
    # 确保已安装依赖
    # pip install fastapi uvicorn supabase grpcio grpcio-tools pandas

    uvicorn trading-bridge.src.main:app --reload --host 0.0.0.0 --port 8000
    ```

### 第三步：开始采集

1.  Python 服务启动后，会显示 `Starting signal watcher on ...`。
2.  在 MT5 策略测试器中点击 **Start**。
3.  观察 Python 终端输出：
    *   成功：`🔔 New Signal Received: ...`
    *   入库：`💾 Extended training features saved to DB`

---

## 4. 验证与后续

### 检查数据
登录 Supabase Dashboard，查询 `training_signals` 表：
```sql
SELECT count(*) FROM training_signals;
SELECT * FROM training_signals ORDER BY timestamp DESC LIMIT 10;
```
确认 `ema_spread`, `trend_direction`, `volatility_ok` 等扩展字段均有数值。

### 训练模型
数据采集完成后，即可使用 `ai-engine/src/train.py` (需根据新表结构微调) 读取数据库中的数据进行 LightGBM 模型训练。

---

## 5. 常见问题 (Troubleshooting)

*   **Q: 终端没有任何反应？**
    *   A: 检查 `SIGNAL_DIR` 路径是否正确。MT5 的沙盒机制非常严格。如果是回测模式，文件可能在 `Tester/Agent` 目录下。
    *   A: 确保指标已正确编译并加载。

*   **Q: 报错 `ModuleNotFoundError: No module named 'alphaos_pb2'`？**
    *   A: `PYTHONPATH` 未正确设置。请在项目根目录重新执行 `export PYTHONPATH=$PYTHONPATH:$(pwd)/ai-engine/src:$(pwd)/trading-bridge/src`。

*   **Q: 数据库连接失败？**
    *   A: 检查环境变量 `SUPABASE_URL` 和 `SUPABASE_KEY` 是否正确导出。

