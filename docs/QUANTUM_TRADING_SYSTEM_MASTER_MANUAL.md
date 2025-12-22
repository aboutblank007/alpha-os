# AlphaOS 量子交易系统主手册 (Master Manual)

**适用架构**: Apple Silicon (M2 Pro/Max)  
**系统核心**: QuantumNet-Lite + MetaTrader 5  
**最后更新**: 2025-12

---

## 1. 架构总论：生产环境选型决策

本部分基于对计算吞吐量与数值精度的深度评估，确立了系统的硬件与软件基石。

### 1.1 核心结论：CPU 优于 GPU

针对 XAUUSD（黄金/美元）微观结构数据的特性，经过对混合模式（方案 A）、MPS 加速模式（方案 B）和纯 CPU 优化模式（方案 C）的对比，我们确立 **方案 C（纯 CPU 优化仿真）** 为唯一符合企业级生产标准的选择。

*   **MPS (GPU) 的缺陷**：Metal Performance Shaders 缺乏原生的双精度（Float64）支持。强制使用 Float32 会导致量子态模拟产生累积误差，引发“贫瘠高原”现象，导致模型无法收敛。
*   **M2 Pro CPU 的优势**：
    *   **指令集并行**：利用 Firestorm 核心的 NEON 指令集加速复数运算。
    *   **伴随微分 (Adjoint Differentiation)**：支持高效的梯度计算，$O(1)$ 复杂度。
    *   **Float64 原生支持**：完美捕捉微小的价差信号。

### 1.2 推荐架构栈

*   **硬件**：Apple M2 Pro (绑定 P-Cores 线程)
*   **后端**：PennyLane `lightning.qubit` (C++ Backend)
*   **接口**：PyTorch (Float64 Mode)
*   **数据流**：ZeroMQ (MT5 -> Python) -> Pickled Scaler -> QNode

---

## 2. 数据工程：微观结构特征采集

量子模型的有效性完全取决于输入数据的物理意义。我们使用定制的 MT5 EA 提取特征。

### 2.1 关键特征物理意义

| 特征名称 | 物理含义 | 量子态映射建议 |
| :--- | :--- | :--- |
| **ema_spread** | 趋势动量 | $\theta \in [-\pi, \pi]$ (相位翻转) |
| **wick_ratio** | 市场熵/拒绝力度 | $\theta \in [0, \pi]$ (振幅) |
| **volume_shock** | 相变/情绪突变 | 需对数缩放以防态空间压缩 |
| **dom_pressure** | 订单流不平衡代理 | 辅助量子比特编码 |

### 2.2 数据采集 EA (MQL5)

使用以下策略在 MetaTrader 5 策略测试器中生成训练数据。

#### 核心逻辑 (OnTick)

```cpp
// 仅在新 K 线生成时采集，保证数据收敛
if(!IsNewBar()) return;

// 计算影线比率 (Wick Ratio)
double candle_range = rates[0].high - rates[0].low;
double wick_upper   = rates[0].high - MathMax(rates[0].open, rates[0].close);
double wick_lower   = MathMin(rates[0].open, rates[0].close) - rates[0].low;
double wick_ratio   = (candle_range > _Point) ? (wick_upper + wick_lower) / candle_range : 0.0;

// 计算成交量冲击 (Volume Shock)
double vol_sum = 0;
for(int i=1; i<=InpVolMaPeriod; i++) vol_sum += (double)rates[i].tick_volume;
double vol_ma = vol_sum / (double)InpVolMaPeriod;
double vol_shock = (vol_ma > 0) ? (double)rates[0].tick_volume / vol_ma : 1.0;

// 写入 CSV
FileWrite(file_handle, ... wick_ratio, vol_shock ...);
```

*完整 EA 代码请参见 `src/Connectors/MT5/Experts/QuantumNet_Data_Miner.mq5`*

### 2.3 数据清洗与全息映射 (Holographic Mapping)

在Python端进行预处理，将原始 CSV 转换为量子就绪数据：

1.  **对数变换**：对 `volume_shock` 等长尾分布特征应用 $\ln(1+x)$。
2.  **角度归一化**：使用 `MinMaxScaler` 将特征严格映射到 $[-\pi, \pi]$ 或 $[0, \pi]$，避免相位混叠。
3.  **目标缩放**：回归目标 `target` 缩放到 $[-0.8, 0.8]$ 以匹配 Pauli-Z 算符的期望值范围。

---

## 3. 模型训练与优化 (M2 Pro 专用)

### 3.1 环境配置

建立支持 MPS（用于经典层）和 NEON 加速（用于量子层）的混合环境。

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pennylane
```

### 3.2 训练脚本 (train_quantum_m2.py)

该脚本针对 M2 Pro 统一内存架构进行了优化。

```python
import pennylane as qml
import torch
import torch.nn as nn

# 1. 强类型与后端配置
torch.set_default_dtype(torch.float64) # 关键：防止梯度噪声
dev = qml.device("lightning.qubit", wires=12) # 关键：CPU 高性能后端

# 2. 量子节点 (QNode)
@qml.qnode(dev, interface="torch", diff_method="adjoint") # 关键：伴随微分
def quantum_circuit(inputs, weights):
    # 角度嵌入
    qml.AngleEmbedding(inputs, wires=range(12), rotation='Y')
    # 强纠缠层
    qml.StronglyEntanglingLayers(weights, wires=range(12))
    return qml.expval(qml.PauliZ(0))

# 3. 混合模型
class QuantumNetNative(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes={"weights": (3, 12, 3)})
        
    def forward(self, x):
        return self.q_layer(x)

# 4. 训练循环 (CPU 线程绑定)
# 建议在运行前设置环境变量: export OMP_NUM_THREADS=8
```

### 3.3 性能优化指南

1.  **Batch Size**: 尽管 M2 内存够大，但对于 CPU 模拟，建议 Batch Size 控制在 `32-128` 之间，以平衡单步延迟和梯度稳定性。
2.  **早停策略**: 金融数据噪声极大。监控验证集 Loss，一旦连续 5 个 Epoch 不下降立即停止，防止过拟合市场微观噪声。
3.  **多进程调参**: 利用 Python `multiprocessing` 在 M2 Pro 的 10-12 个核心上并行运行多个训练任务，以空间换时间。

---

## 4. 附录：文件索引

*   `docs/量子计算生产环境方案.md`: 详细的硬件选型报告。
*   `docs/QuantumNet 数据采集.MD`: 详细的数据字典与 EA 逻辑。
*   `docs/M2 Pro 量子回归训练指南.md`: 独立的训练操作手册。
