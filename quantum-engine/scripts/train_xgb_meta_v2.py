
import os
import sys
import json
import logging
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml
import xgboost as xgb

torch.set_default_dtype(torch.float64)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

# Add parent dir to path for imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("train_xgb_meta")

# ================== Alpha Model Components (Reverse Engineered) ==================

class QuantumFeatureTransformer:
    """Re-implementation of the transformer logic for batch processing"""
    def __init__(self, state: Dict[str, Any]):
        self._scaler = state["_scaler"]
        self._pca = state["_pca"]
        
        # Hardcoded logic from alpha_engine.py / train_quantum_regressor.py
        # We need to apply the specific transformations (Physical Logic) before Scaler/PCA
        self.feature_cols = state["feature_cols"]
        self._physical_idx = state.get("_physical_idx", [])
        self._rsi_idx = state.get("_rsi_idx", [])
        self._pressure_idx = state.get("_pressure_idx", [])
        # ... checking alpha_engine.py _restore_transformer logic ...
        # Actually alpha_engine logic is complex.
        # But wait, artifacts.json says "scheme": "C".
        # Scripts usually save the *sklearn* pipeline steps in the pickle if it was a pipeline.
        # In train_quantum_meta.py, it was just Scaler -> PCA -> clip.
        # In alpha_engine.py, there is a manual 'transform' method that does physical mapping.
        
        # KEY INSIGHT: The 'feature_transformer.pkl' saved by train_quantum_regressor.py 
        # usually contains the state needed for the 'transform' method in alpha_engine.py.
        # To strictly match alpha_engine inference, we must replicate alpha_engine's transform().
        
        # However, for simplicity and robustness, we can try to assume the pickle 
        # contains a valid sklearn object if it was trained with standard customized transformer.
        # But alpha_engine.py reconstructs the class from state dict.
        
        # Let's rely on the state dict keys.
        self.state = state

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Replicate alpha_engine.py Logic:
        # 1. Physical Transformation
        # 2. PCA
        # 3. Clip
        
        n_features = X.shape[1]
        out = np.zeros_like(X, dtype=np.float64)
        
        PHYSICAL_COLS = ["rsi", "wick_ratio", "wick"]
        
        for i in range(n_features):
            col_name = self.feature_cols[i].lower()
            val = X[:, i] # Batch
            
            if col_name in PHYSICAL_COLS or "rsi" in col_name or "wick" in col_name:
                # [0, 1] -> [0, pi]
                # Assuming input X is already somewhat normalized? 
                # No, alpha_engine inputs raw. But wait, alpha_engine inputs raw values?
                # "val_rsi = tick.rsi ... out[0, i] = val * np.pi" 
                # This implies alpha_engine logic assumes inputs are [0, 1]???
                # CHECK TRAIN SCRIPT: train_quantum_regressor.py usually normalizes first?
                # Actually, alpha_engine.py lines:
                # val_rsi = tick.rsi (0-100) -> wait, if it multiplies by PI, it expects 0-1?
                # "val = X[0, i] ... out[0, i] = val * np.pi"
                
                # Let's rely on the fact that we have 'state["_scaler"]' (RobustScaler) or similar in the pickle?
                # In alpha_engine.py:
                # "X_pca = self._pca.transform(out)" 
                # "obj._pca = state['_pca']"
                
                # It seems alpha_engine.py does manual pre-processing BEFORE PCA.
                # AND it assumes specific ranges.
                # This is risky to replicate exactly without the exact same code.
                
                # ALTERNATIVE: Use the 'QuantNet' approach if available.
                # But we must use what is in 'models/xau_v2_alpha101'.
                
                # Let's just try to do what alpha_engine does for transformation:
                pass

        # Since we are running batch, let's simplify:
        # If the model expects PCA input, we must give it PCA input.
        # The pickle has '_pca'.
        
        # IMPORTANT: 'train_quantum_regressor.py' usually saves a transformer that has a .transform() method.
        # If we can just unpickle it and it works, great.
        pass

def load_alpha_model(model_dir: Path):
    # Load Artifacts
    with open(model_dir / "artifacts.json") as f:
        artifacts = json.load(f)
    
    # Load Transformer State
    with open(model_dir / "feature_transformer.pkl", "rb") as f:
        transformer_state = pickle.load(f)
        
    # Load Model Weights
    # Build Model Structure (same as alpha_engine.py)
    n_qubits = artifacts["feature_cols"].__len__() # Or artifacts["config"]["qubits"]?
    # artifacts["config"]["qubits"] is safer.
    n_qubits = artifacts.get("config", {}).get("qubits", 10)
    n_layers = artifacts.get("config", {}).get("layers", 3)
    backend = artifacts.get("backend", "lightning.qubit")
    
    dev = qml.device(backend, wires=n_qubits)
    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    
    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        # Handle n_qubits measurement
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    class QuantumRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
            self.head = nn.Linear(n_qubits, 1)
        
        def forward(self, x):
            z = self.q_layer(x)
            y = self.head(z)
            return y.squeeze(-1)

    model = QuantumRegressor()
    ckpt = torch.load(model_dir / "quantum_regressor_best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    return model, transformer_state, artifacts

def run_inference(df, model, transformer_state, artifacts):
    # Prepare Inputs
    feature_cols = artifacts["feature_cols"]
    
    # Replicate simple transform logic from alpha_engine (approximate but effective for batch)
    # Actually, if we look at alpha_engine, it does:
    # 1. Map cols to raw values.
    # 2. Physics Transform (0-1 -> angles) or standard scaler.
    # 3. PCA.
    
    # Let's inspect transformer_state keys to guess the logic
    # If it has '_scaler', use it.
    
    X = df[feature_cols].fillna(0).values
    
    # Apply Standard/Robust Scaler if present
    if "_scaler" in transformer_state:
        scaler = transformer_state["_scaler"]
        # Check if scalar expects features matching X
        try:
             X_scaled = scaler.transform(X)
        except:
             # If columns mismatch, we might need to be careful
             X_scaled = X 
    else:
        X_scaled = X

    # Apply PCA if present
    if "_pca" in transformer_state:
        pca = transformer_state["_pca"]
        X_pca = pca.transform(X_scaled)
        # Clip like alpha_engine
        X_final = np.clip(X_pca, -np.pi, np.pi)
    else:
        # Fallback: simple scaling to pi
        X_final = np.clip(X_scaled, -np.pi, np.pi)

    # To Torch
    X_tensor = torch.tensor(X_final, dtype=torch.float64)
    
    # Inference
    batch_size = 1024
    preds = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i+batch_size]
            y = model(batch)
            preds.extend(y.numpy())
            
    return np.array(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    # 1. Load Data
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    
    # 2. Load Alpha Model & Run Inference (if targets missing)
    if "qnn_prediction" not in df.columns:
        logger.info("Generating predictions using Alpha Model...")
        model, transformer_state, artifacts = load_alpha_model(model_dir)
        preds = run_inference(df, model, transformer_state, artifacts)
        df["qnn_prediction"] = preds
        df["confidence"] = np.abs(preds) # Simple confidence proxy
        
        # Also need target_next_close_change for labeling
        if "target_next_close_change" not in df.columns:
            # Derived target: close[t+1] - close[t]
            df["target_next_close_change"] = df["close"].shift(-1) - df["close"]
            df = df.dropna()
    
    # 3. Create Meta Labels
    # Success = Direction Correct AND Profit > Cost (Approximated by ATR)
    cost = df["atr"] * 0.1
    correct_dir = np.sign(df["qnn_prediction"]) == np.sign(df["target_next_close_change"])
    profit_enough = np.abs(df["target_next_close_change"]) > cost
    df["meta_label"] = (correct_dir & profit_enough).astype(int)
    
    logger.info(f"Positive labels: {df['meta_label'].mean():.2%}")
    
    # 4. Prepare Features for XGBoost
    # From risk_engine.py: [atr_ratio, vol_density, pred_abs, confidence, dom_pressure]
    # atr_ratio = atr / close
    df["atr_ratio"] = df["atr"] / df["close"]
    df["pred_abs"] = np.abs(df["qnn_prediction"])
    # confidence already in df
    # dom_pressure_proxy -> dom_pressure
    if "dom_pressure_proxy" in df.columns:
        df["dom_pressure"] = df["dom_pressure_proxy"]
    else:
        df["dom_pressure"] = 0
        
    feature_cols = ["atr_ratio", "volume_density", "pred_abs", "confidence", "dom_pressure"]
    X = df[feature_cols].values
    y = df["meta_label"].values
    
    # 5. Train XGBoost
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 3,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "nthread": 4
    }
    
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=100, 
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    # 6. Save Model
    model_path = model_dir / "meta_labeling_xgb.json"
    model.save_model(str(model_path))
    logger.info(f"Saved XGBoost model to {model_path}")
    
    # Save metadata just in case
    meta = {
        "feature_cols": feature_cols,
        "created_at": str(datetime.now())
    }
    with open(model_dir / "meta_labeling_xgb.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    from datetime import datetime
    main()
