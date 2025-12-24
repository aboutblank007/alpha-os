#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证风控引擎加载
"""

import os
import sys
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("verify_risk")

# 添加 qlink 目录到路径
SCRIPT_DIR = Path(__file__).parent.resolve()
QUANTUM_ENGINE_DIR = SCRIPT_DIR.parent
QLINK_DIR = QUANTUM_ENGINE_DIR / "qlink"
sys.path.append(str(QLINK_DIR))

# 导入 Risk Engine 中的组件
try:
    # 注意：risk_engine 依赖 protocol
    from risk_engine import ATRCalculator
    import xgboost as xgb
    logger.info("✅ 成功导入风险控制相关组件")
except ImportError as e:
    logger.error("❌ 导入失败: %s", e)
    sys.exit(1)

def main():
    # 1. 检查模型路径 (XAU)
    model_dir = QUANTUM_ENGINE_DIR / "models" / "xau"
    if not model_dir.exists():
        logger.error("❌ 模型目录不存在: %s", model_dir)
        return
    
    # 2. 检查 XGBoost 模型文件
    xgb_path = model_dir / "meta_labeling_xgb.json"
    if not xgb_path.exists():
        logger.error("❌ XGBoost 模型不存在: %s", xgb_path)
    else:
        try:
            model = xgb.Booster()
            model.load_model(str(xgb_path))
            logger.info("✅ XGBoost Meta-Labeling 模型加载成功: %s", xgb_path)
        except Exception as e:
            logger.error("❌ XGBoost 模型加载失败: %s", e)

    # 3. 检查 Meta Transformer
    meta_trans_path = model_dir / "meta_transformer.pkl"
    if meta_trans_path.exists():
        import pickle
        try:
            with meta_trans_path.open("rb") as f:
                trans = pickle.load(f)
            logger.info("✅ Meta Transformer 加载成功")
        except Exception as e:
            logger.error("❌ Meta Transformer 加载失败: %s", e)
    
    # 4. 检查 ATR 计算器
    atr = ATRCalculator(period=14)
    logger.info("✅ ATR 计算器初始化成功")

if __name__ == "__main__":
    main()
