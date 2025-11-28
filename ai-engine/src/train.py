import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
from datetime import datetime
from features import FeatureEngineer

# Configuration
MODEL_PATH = "ai-engine/models/lgbm_scalping_v1.txt"
TP_PCT = 0.001   # 0.1%
SL_PCT = 0.0008  # 0.08%
TIME_WINDOW = 5  # Bars (e.g., 5 * 5m = 25m)

class ScalpingTrainer:
    def __init__(self):
        self.fe = FeatureEngineer()
        self.model = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load OHLCV data from CSV/Parquet.
        """
        if not os.path.exists(file_path):
            print(f"⚠️ Data file {file_path} not found. Generating dummy data for test.")
            return self._generate_dummy_data()
            
        df = pd.read_csv(file_path)
        return df

    def _generate_dummy_data(self, n=1000):
        """Generate synthetic data for testing the pipeline"""
        dates = pd.date_range(end=datetime.now(), periods=n, freq='5min')
        data = {
            'time': dates,
            'open': np.random.uniform(100, 200, n),
            'high': np.random.uniform(100, 200, n),
            'low': np.random.uniform(100, 200, n),
            'close': np.random.uniform(100, 200, n),
            'volume': np.random.uniform(1000, 5000, n)
        }
        # Fix H/L logic
        df = pd.DataFrame(data)
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 2, n)
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 2, n)
        return df

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Triple Barrier Method for Scalping.
        Label 1: Hits TP (0.1%) before SL (0.08%) within Time Window (5 bars).
        Label 0: Hits SL or Timeout.
        """
        labels = []
        
        # Convert to numpy for speed
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        n = len(df)
        
        for i in range(n):
            if i + TIME_WINDOW >= n:
                labels.append(0) # Not enough data
                continue
                
            entry_price = close[i]
            tp_price = entry_price * (1 + TP_PCT)
            sl_price = entry_price * (1 - SL_PCT)
            
            outcome = 0 # Default Failure/Timeout
            
            # Look forward
            for j in range(i + 1, i + TIME_WINDOW + 1):
                # Check SL first (Conservative)
                if low[j] <= sl_price:
                    outcome = 0
                    break
                
                # Check TP
                if high[j] >= tp_price:
                    outcome = 1
                    break
            
            labels.append(outcome)
            
        df['target'] = labels
        print(f"✅ Labels created. Positive Rate: {df['target'].mean():.2%}")
        return df

    def train(self, data_path: str):
        print("🔄 Loading Data...")
        df = self.load_data(data_path)
        
        print("🛠️ Engineering Features...")
        # We don't have historical DOM data in standard OHLCV, so we train on technicals only
        # In a real scenario, we'd need a dataset with captured DOM snapshots.
        df = self.fe.calculate_features(df)
        
        print("🏷️ Creating Labels...")
        df = self.create_labels(df)
        
        # Prepare Training Data
        # Drop rows where targets cannot be calculated (end of df)
        df = df.iloc[:-TIME_WINDOW]
        
        # Drop NaNs
        df = df.dropna()
        
        feature_cols = [c for c in df.columns if c not in ['time', 'target', 'open', 'high', 'low', 'close', 'volume']]
        print(f"Training with {len(feature_cols)} features: {feature_cols}")
        
        X = df[feature_cols]
        y = df['target']
        
        # Time-Series Split (Train on first 80%, Test on last 20%)
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # Train LightGBM
        print("🚀 Training LightGBM...")
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(100)
            ]
        )
        
        # Save Model
        self.model.save_model(MODEL_PATH)
        print(f"💾 Model saved to {MODEL_PATH}")
        
        # Feature Importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 10 Features:")
        print(importance.head(10))

if __name__ == "__main__":
    trainer = ScalpingTrainer()
    # Pass a non-existent path to trigger dummy data generation for demonstration
    trainer.train("data/gold_m5.csv")

