import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
DATA_PATH = "training_data.csv"
MODEL_PATH = "ai-engine/models/lgbm_scalping_v1.txt"

class ScalpingTrainer:
    def __init__(self):
        self.model = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load pre-processed training data (Signals + Outcomes).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file {file_path} not found.")
            
        df = pd.read_csv(file_path)
        print(f"📊 Loaded {len(df)} records.")
        return df

    def prepare_data(self, df: pd.DataFrame):
        """
        Prepare X and y for training.
        """
        # 1. Filter only completed trades
        if 'has_outcome' in df.columns:
            df = df[df['has_outcome'] == True].copy()
        
        # 2. Define Feature Columns (Exclude metadata like IDs, timestamps, raw prices)
        # Using the columns present in the generated CSV
        feature_cols = [
            'ema_short', 'ema_long', 'atr', 'adx', 'center', 
            'distance_ok', 'slope_ok', 'trend_filter_ok', 'htf_trend_ok', 
            'volatility_ok', 'chop_ok', 'spread_ok', 
            'bars_since_last', 'trend_direction', 'ema_cross_event', 
            'ema_spread', 'atr_percent', 'reclaim_state', 'is_reclaim_signal', 
            'price_vs_center', 'cloud_width'
        ]
        
        # Ensure all columns exist
        available_cols = [c for c in feature_cols if c in df.columns]
        missing_cols = set(feature_cols) - set(available_cols)
        if missing_cols:
            print(f"⚠️ Warning: Missing columns: {missing_cols}")
            
        X = df[available_cols]
        y = df['outcome']
        
        print(f"✅ Prepared {len(X)} samples with {len(available_cols)} features.")
        print(f"   Win Rate in Data: {y.mean():.2%}")
        
        return X, y, available_cols

    def train(self, data_path: str):
        print("🔄 Loading Data...")
        df = self.load_data(data_path)
        
        X, y, feature_cols = self.prepare_data(df)
        
        # Split Data (Shuffle=False for time series, but here signals are discrete events)
        # Since these are discrete signals, random split is acceptable if we assume market regime is mixed,
        # but strictly speaking time-series split is safer. 
        # Let's use a simple shuffle split for now as this is a feature-based classification.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        # Train LightGBM
        print("🚀 Training LightGBM...")
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test)
        
        params = {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(50)
            ]
        )
        
        # Evaluation
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        print("\n📊 Model Evaluation:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save Model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        self.model.save_model(MODEL_PATH)
        print(f"\n💾 Model saved to {MODEL_PATH}")
        
        # Feature Importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 10 Features (Gain):")
        print(importance.head(10))

if __name__ == "__main__":
    trainer = ScalpingTrainer()
    # Ensure we use the correct relative path based on where script is run
    # Assuming run from root: python ai-engine/src/train.py
    # Data is at root: training_data.csv
    data_file = "training_data.csv"
    if os.path.exists(data_file):
        trainer.train(data_file)
    else:
        print(f"❌ File {data_file} not found. Please run export_training_data.py first.")
