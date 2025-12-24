#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证量子模型加载与推理环境
"""

import os
import sys
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("verify_load")

# 添加 qlink 目录到路径
SCRIPT_DIR = Path(__file__).parent.resolve()
QUANTUM_ENGINE_DIR = SCRIPT_DIR.parent
QLINK_DIR = QUANTUM_ENGINE_DIR / "qlink"
sys.path.append(str(QLINK_DIR))

# 导入 Alpha Engine 中的组件
try:
    from alpha_engine import QuantumPredictor, TargetScaler
    import __main__
    __main__.TargetScaler = TargetScaler
    logger.info("✅ 成功从 alpha_engine 导入 QuantumPredictor 并注入 TargetScaler 到 __main__")
except ImportError as e:
    logger.error("❌ 无法导入 QuantumPredictor: %s", e)
    sys.exit(1)

def main():
    # 1. 检查模型路径
    model_dir = QUANTUM_ENGINE_DIR / "models" / "xau_light"
    if not model_dir.exists():
        logger.error("❌ 模型目录不存在: %s", model_dir)
        return
    
    logger.info("开始加载模型: %s", model_dir)
    
    # 2. 尝试加载模型
    try:
        predictor = QuantumPredictor(model_dir)
        logger.info("✅ 模型加载成功！")
        
        # 3. 打印模型关键参数
        logger.info("模型配置: qubits=%d, layers=%d, backend=%s", 
                    predictor.n_qubits, predictor.n_layers, predictor.backend)
        logger.info("特征列数: %d", len(predictor.feature_cols))
        
        if predictor.feature_transformer:
            logger.info("✅ 特征变换器已就绪")
        if predictor.target_scaler:
            logger.info("✅ 目标变量缩放器已就绪")
            
        # 4. 模拟一次推理 (冒烟测试)
        import torch
        import numpy as np
        
        # 构造假数据
        dummy_input = torch.zeros((1, predictor.n_qubits), dtype=torch.float64)
        with torch.no_grad():
            output = predictor.model(dummy_input)
            logger.info("✅ 冒烟测试成功: 原始预测值 = %s", output.item())
            
    except Exception as e:
        logger.exception("❌ 模型加载或测试失败: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
