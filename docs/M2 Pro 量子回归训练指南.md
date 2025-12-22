# M2 Pro 量子回归训练指南

## 1. 简介

本指南旨在详细说明如何在 Apple M2 Pro 芯片上利用 Metal Performance Shaders (MPS) 加速器，对 `QuantumNet-Lite` 量子回归模型进行高效训练。通过利用 macOS 的统一内存架构和 GPU 加速，我们可以显著缩短训练周期，实现分钟级的模型迭代。

## 2. 环境配置 (M2 Pro/Max 专用)

为了在 Mac 上启用 GPU 加速，必须安装支持 MPS 的 PyTorch 版本。

### 2.1 依赖安装

确保你的 Python 环境（推荐 Python 3.9 或 3.10）已安装以下库：

```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn polars tqdm
```

### 2.2 验证 MPS 加速

运行以下 Python 代码验证 Metal 加速是否开启：

```python
import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print("✅ MPS Acceleration Enabled!")
else:
    print("❌ MPS not available.")
```

## 3. 数据准备

训练数据来源于 MetaTrader 5 的策略测试器导出。

- **源文件**: `QuantumNet_Training_Data.csv` (由 EA 生成)
- **特征集**: 包含 RSI, ATR, Wick Ratio, Volume Density, Volume Shock 等量子特征。
- **预处理**: 使用 Python 进行标准化 (Z-Score) 和 序列化 (Rolling Window)。

## 4. 训练脚本实现

以下是完整的训练脚本 `train_quantum_m2.py`，专为 M2 Pro 优化。它实现了数据加载、模型初始化（映射到 MPS）、以及回归损失函数（MSE）的训练循环。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import sys
import os

# 导入你的模型定义
sys.path.append("ai-engine/src")
from models.quantum_net import QuantumNetLite

# --- 配置 ---
BATCH_SIZE = 512       # M2 Pro 拥有主要内存，大 Batch 更高效
LEARNING_RATE = 1e-4
EPOCHS = 50
SEQ_LEN = 64           # 时间序列长度
FEATURES_DIM = 12      # 根据你的 CSV 特征数量调整
DEVICE = torch.device("mps")  # 强制使用 MPS

class FinancialDataset(Dataset):
    def __init__(self, csv_file, seq_len=64):
        self.seq_len = seq_len
        
        # 1. 加载数据
        df = pd.read_csv(csv_file)
        
        # 2. 特征选择 (排除时间戳和 Symbol)
        # 假设最后两列是 Target (Policy, Value)，其余是特征
        # 请根据你的 CSV 结构调整此处
        feature_cols = [c for c in df.columns if c not in ['timestamp', 'symbol', 'target_next_close_change']]
        target_col = 'target_next_close_change' # 回归目标
        
        self.data = df[feature_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)
        
        # 3. 标准化
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)
        
        self.data_tensor = torch.tensor(self.data, device=torch.device("cpu")) # 放在 CPU 以节省显存，训练时搬运
        self.target_tensor = torch.tensor(self.targets, device=torch.device("cpu"))

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # 提取序列窗口
        x = self.data_tensor[idx : idx + self.seq_len]
        # 提取最后一个时间步的目标 (或者下一帧的目标)
        y_val = self.target_tensor[idx + self.seq_len] 
        # y_policy = ... (如果需要分类)
        
        return x, y_val

def train():
    print(f"🚀 Starting Training on {DEVICE}")
    
    # 1. 准备数据
    dataset = FinancialDataset("QuantumNet_Training_Data.csv", seq_len=SEQ_LEN)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. 初始化模型
    model = QuantumNetLite(input_dim=dataset.data.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_value = nn.MSELoss() # 回归损失
    
    # 3. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE).unsqueeze(1) # (Batch, 1)
            
            optimizer.zero_grad()
            
            # Forward
            policy, value = model(batch_x)
            
            # Loss Calculation (这里只关注回归 Value)
            loss = criterion_value(value, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE).unsqueeze(1)
                
                _, value = model(batch_x)
                loss = criterion_value(value, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"📉 Epoch {epoch+1} Val Loss: {val_loss:.6f}")
        
    # 4. 保存模型
    torch.save(model.state_dict(), "ai-engine/models/quantum_net_m2.pth")
    print("💾 Model Saved!")

if __name__ == "__main__":
    train()
```

## 5. M2 Pro 性能优化建议

1.  **Batch Size**: M2 Pro 的内存带宽极高。建议尝试 `512`, `1024` 甚至 `2048` 的 Batch Size，这通常比小 Batch 训练更快。
2.  **数据类型**: MPS 目前对 `float32` 支持最好。尽量避免使用 `float64`，会导致回退到 CPU 运算，严重拖慢速度。
3.  **内存管理**: 如果遇到内存不足（OOM），请在 `Dataset` 中保留数据在 CPU (`data_tensor` 不要直接 `.to("mps")`)，只在 `__getitem__` 或循环中移动 Batch 到 MPS。这利用了 M2 的统一内存架构优势。

## 6. 下一步：模型部署

训练完成后的模型 (`quantum_net_m2.pth`) 可以通过 `ai-engine/src/client.py` 加载，用于实时推理。
请确保推理时的 `input_dim` 和 `seq_len` 与训练时完全一致。