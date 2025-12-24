# Quantum Engine: XAUUSD 高频量子交易引擎 (v2.1)

本目录包含基于量子变分电路 (VQC) 的高频交易核心引擎。系统专为 **Apple M2 Pro (Apple Silicon)** 架构优化，通过 Sidecar 模式提供毫秒级的预测与风控。

> [!CAUTION]
> **[Ref: docs/MT5 交易系统生产级落地方案.MD]**
> 系统底层采用 `float64` 双精度计算，严禁在后端代码中使用 `float32` 导致数值溢出或精度损失。

---

## 🏗️ 架构说明

系统遵循 **Sidecar 设计模式**，各个组件通过 ZeroMQ (ZMQ) 管道实现解耦通信：

- **Pod B (Alpha Engine)**: 负责量子推理。加载 `Alpha101` 模型，计算 `TS_Rank` 特征并输出 `AlphaSignal`。
- **Pod C (Risk Engine)**: 负责三道防线风控。执行 `Meta-Labeling` 概率过滤、波动率目标仓位管理及 L-VaR 审查。
- **Execution Domain (MT5)**: 负责行情流推送与订单执行。

---

## 🛠️ 环境配置 (M2 Pro 专用)

### 1. 初始化环境
```bash
cd quantum-engine
./scripts/setup_venv.sh
source .venv/bin/activate
```

### 2. 硬件加速约束
根据 **[Ref: docs/M2 Pro 量子回归训练指南.md]**，系统强制指定以下配置：
- **Backend**: `lightning.qubit` (C++ CPU 仿真，禁用 MPS/GPU)
- **Precision**: `float64`
- **Parallelism**: `OMP_NUM_THREADS=8` (适配 M2 Pro 的 P-Core)

---

## 📊 数据采集流程

数据链路基于 **[Ref: docs/QuantumNet 数据采集.MD]** 实现：

1. **部署 EA**: 在 MT5 端挂载 `mql5/QuantumNet_Data_Miner_v2.mq5`。
2. **精度校验**: 确保导出的 CSV 特征文件符合 `%.7f` 高精度协议。
3. **特征集**: 核心特征包含 `wick_ratio`, `vol_density`, `dom_pressure_proxy`, `bid_ask_imbalance` 等 21 个物理因子。

---

## 🧬 模型重训指南 (Alpha101 2.0)

当模型预测力下降（Alpha 衰减）时，需执行重训：

### 1. 运行重训脚本
```bash
PYTHONPATH=. OMP_NUM_THREADS=8 .venv/bin/python3 src/train_quantum_regressor.py
```
- **核心逻辑**: 自动执行 `TS_Rank` (时间序列排名)，将特征映射到 `[0, pi]` 量子旋转角。
- **产物**: 保存在 `models/xau_v2_alpha101/`，包含权重 (`.pt`) 与预处理器 (`.pkl`)。

### 2. 对称性验证
```bash
.venv/bin/python3 verify_symmetry.py
```
- 确保零输入输出为零，消除方向性偏移 (Directional Bias)。

---

## 🚀 实盘部署与监控

### 1. 启动全链路
```bash
cd qlink
./launch.sh <MT5_IP> <SYMBOL>
```

### 2. 日志标准
所有后端日志均汇总于 `quantum-engine/logs/`，采用标准化格式：
- `alpha_engine.log`: 推理状态。
- `risk_engine.log`: 订单与风控详情。

**标准格式示例**:
`[TIMESTAMP] [LEVEL] [COMPONENT] Action | Remarks {Metadata JSON}`

---

## 🛑 故障排除

| 错误现象 | 可能原因 | 解决办法 |
| :--- | :--- | :--- |
| `NoneType` 减法异常 | MT5 持仓止损同步延迟 | 检查 `risk_engine.py` 的容错逻辑 (Fixed in L1209) |
| `IndexError` (维度错误) | TS_Rank 缓冲区形状失真 | 确保使用 `np.vstack` 对齐矩阵 |
| 订单发送缓冲区满 | ZMQ SNDHWM 过小或 EA 断连 | 检查网络连接，确认 `setsockopt(zmq.SNDHWM, 1000)` |

---

**[最新战略研究参考: docs/XAUUSD_Quantum_Strategic_Research.md]**
