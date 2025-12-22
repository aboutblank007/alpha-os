## Quantum Engine（真量子电路回归引擎）

目标：基于 `QuantumNet_Training_Data.aligned.csv` 训练**真量子电路（PQC/VQC）回归模型**，输出
`y_hat = 预测的 target_next_close_change`，用于后续独立的交易规则（不依赖旧 ai-engine 框架）。

### 目录结构
- `quantum-engine/requirements.txt`：独立依赖（PennyLane 等）
- `quantum-engine/src/`：训练、推理、预处理代码
- `quantum-engine/models/`：模型与预处理产物（参数、scaler、pca 等）
- `quantum-engine/reports/`：训练评估报告（样本外指标、漂移、稳定性等）

### 快速开始（安装依赖后）
训练：

```bash
python3 quantum-engine/src/train_quantum_regressor.py \
  --data /Users/hanjianglin/github/alpha-os/QuantumNet_Training_Data.aligned.csv \
  --outdir /Users/hanjianglin/github/alpha-os/quantum-engine/models
```

推理（输出一个 y_hat）：

```bash
python3 quantum-engine/src/infer_quantum_regressor.py \
  --modeldir /Users/hanjianglin/github/alpha-os/quantum-engine/models \
  --row-json '{"ema_spread":0.1,"rsi":55,"atr":0.8,"adx":20,"wick_ratio":0.3,"volume_density":1.1,"volume_shock":1.2,"dom_pressure_proxy":10}'
```


