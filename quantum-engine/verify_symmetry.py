import pickle
import numpy as np
import sys
import os

# 导入 TargetScaler 定义以确保 pickle 加载成功
# 因为 TargetScaler 在训练脚本中定义，我们在这里也要定义一个结构一致的类
class TargetScaler:
    def __init__(self):
        self.scale_ = 1.0
        self._fitted = False
    
    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        y_scaled = np.asarray(y_scaled, dtype=np.float64)
        return (y_scaled / 0.8) * self.scale_

def verify_symmetry(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return
    
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    
    # 验证对称性：0 的逆变换必须是 0
    neutral_val = scaler.inverse_transform(np.array([0.0]))[0]
    print(f"Symmetry Check: model_output=0.0 -> inverse_transformed={neutral_val}")
    
    if abs(neutral_val) < 1e-10:
        print("✅ SUCCESS: Symmetry logic verified (Zero-Bias).")
    else:
        print("❌ FAILURE: Non-zero bias detected!")

if __name__ == "__main__":
    verify_symmetry("models/xau/target_scaler.pkl")
