import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import logging
import os
import json
import argparse
import shutil
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FilterTrainer")

MODEL_DIR = 'models' if os.path.exists('models') else 'ai-engine/models'
VERSION_FILE = os.path.join(MODEL_DIR, 'version.json')

def get_next_version(reset=False):
    """
    Get the next version number.
    If reset is True, starts from 1 and clears existing version info.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if reset:
        logger.warning("⚠️  RESET REQUESTED: Starting from v1")
        return 1
        
    if os.path.exists(VERSION_FILE):
        try:
            with open(VERSION_FILE, 'r') as f:
                data = json.load(f)
                return data.get('current_version', 0) + 1
        except Exception as e:
            logger.error(f"Failed to read version file: {e}")
            return 1
    
    return 1

def update_version_file(version, description=""):
    """Update the version.json file with current version info."""
    data = {
        "current_version": version,
        "last_updated": datetime.now().isoformat(),
        "description": description
    }
    try:
        with open(VERSION_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"📝 Updated version tracking: v{version}")
    except Exception as e:
        logger.error(f"Failed to write version file: {e}")

def train_filter(reset=False):
    DATA_FILE = 'training_data_enhanced.csv'
    if not os.path.exists(DATA_FILE):
        logger.error(f"{DATA_FILE} not found. Run enhance_features.py first.")
        return

    # Determine Version
    current_version = get_next_version(reset)
    logger.info(f"🔖 Training Run Version: v{current_version}")
    
    if reset:
        # Optional: Delete old models? 
        # For safety, maybe just overwrite/ignore them. 
        # If user explicitly wants "clean slate", we can delete lgbm_*.txt
        logger.info("Cleaning up old model files...")
        for f in os.listdir(MODEL_DIR):
            if f.startswith("lgbm_") and f.endswith(".txt"):
                os.remove(os.path.join(MODEL_DIR, f))

    df = pd.read_csv(DATA_FILE)
    
    # Validation: Ensure 'symbol' column exists
    if 'symbol' not in df.columns:
        logger.error("Dataset missing 'symbol' column. Cannot perform symbol-specific training.")
        return

    # Valid features logic
    # Ensure is_simulated is treated as feature if present
    if 'is_simulated' in df.columns:
        df['is_simulated'] = df['is_simulated'].astype(int)
        
    # Stats
    logger.info(f"Total samples: {len(df)}")
    if 'is_simulated' in df.columns:
        sim_count = df['is_simulated'].sum()
        real_count = len(df) - sim_count
        logger.info(f"Data Composition: {real_count} Real Trades + {sim_count} Simulated Trades (Counterfactual)")
    
    # Target: MFE (Regression) - using Supabase column name
    target = 'result_mfe'
    
    # Features
    # Exclude non-features, time, and future leakage columns
    # Updated to match Supabase schema (result_* columns)
    exclude = ['result_mfe', 'result_mae', 'result_profit', 'result_win', 'mfe', 'mae', 'outcome', 'pnl',
               'timestamp', 'time', 'close_time', 'exit_price', 'exit_time', 'exit_reason',
               'symbol', 'action', 'comment', 'type', 'signal_id', 'has_outcome', 'id',
               'fwd_high', 'fwd_low', 'trend_dir', 'tr0', 'tr1', 'tr2', 'source',
               'order_id', 'position_id', 'executed', 'execution_price', 'execution_spread',
               'broker_time', 'execution_time', 'holding_period', 'created_at', 'updated_at',
               'sl', 'tp']
    features = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
    
    logger.info(f"Features ({len(features)}): {features}")
    
    # Group by Symbol for Specific Models
    grouped = df.groupby('symbol')
    
    for symbol, group in grouped:
        # Log composition for this symbol
        if 'is_simulated' in group.columns:
            s_sim = group['is_simulated'].sum()
            s_real = len(group) - s_sim
            logger.info(f"⚡ Training Model for {symbol} ({len(group)} samples: {s_real} Real, {s_sim} Sim)...")
        else:
            logger.info(f"⚡ Training Model for {symbol} ({len(group)} samples)...")
            
        X = group[features]
        y = group[target]
        
        # Drop NaNs
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(X) < 30:
            logger.warning(f"⚠️  Not enough data for {symbol} (Found {len(X)}, Need 30+). Skipping.")
            continue
            
        logger.info(f"Data shape for {symbol}: {X.shape}")

        # Split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Check for Pre-trained Model (Warm Start) - Only from SAME version or previous?
        # If reset=True, we explicitly do NOT load previous models.
        # If reset=False, we load the *active* model (which points to previous version)
        
        active_model_filename = f"lgbm_{symbol}.txt"
        active_model_path = os.path.join(MODEL_DIR, active_model_filename)
        
        init_booster = None
        
        if not reset and os.path.exists(active_model_path):
            try:
                # Use current active model as base
                init_booster = lgb.Booster(model_file=active_model_path)
                logger.info(f"🔄 Loaded existing model for {symbol} (Warm Start).")
            except Exception as e:
                logger.warning(f"Failed to load existing model {active_model_path}: {e}")

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1
        }
        
        try:
            bst = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
                init_model=init_booster # Incremental Learning
            )
            
            # Eval
            preds = bst.predict(X_test)
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            logger.info(f"[{symbol}] R2: {r2:.4f}, MAE: {mae:.4f}")

            # Save Versioned Model
            versioned_filename = f"lgbm_{symbol}_v{current_version}.txt"
            versioned_path = os.path.join(MODEL_DIR, versioned_filename)
            bst.save_model(versioned_path)
            logger.info(f"✅ Saved versioned model to {versioned_path}")
            
            # Update Active Model (Copy)
            # We use copy instead of symlink to be safe with Docker volumes/Windows
            shutil.copy2(versioned_path, active_model_path)
            logger.info(f"🔄 Updated active model {active_model_path}")
            
        except Exception as e:
            logger.error(f"❌ Training failed for {symbol}: {e}")
            
    # Finalize version file
    update_version_file(current_version, description="Training Run")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset version to 1 and clear old models")
    args = parser.parse_args()
    
    train_filter(reset=args.reset)


