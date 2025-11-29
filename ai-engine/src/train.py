import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration
DATA_PATH = "training_data.csv"
MODEL_PATH = "ai-engine/models/lgbm_scalping_v1.txt"

class ScalpingTrainer:
    def __init__(self):
        self.model = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file {file_path} not found.")
            
        df = pd.read_csv(file_path)
        print(f"📊 Loaded {len(df)} records.")
        return df

    def prepare_data(self, df: pd.DataFrame):
        """
        Prepare X and y for training (Regression Task).
        """
        # Filter invalid data
        if 'mfe' not in df.columns:
            raise ValueError("Column 'mfe' not found. Cannot train regression model.")
            
        # Drop extreme outliers (e.g., data errors or news spikes that skew regression)
        # Cap MFE at 99th percentile to stabilize training
        limit = df['mfe'].quantile(0.99)
        df = df[df['mfe'] < limit].copy()
        
        # Feature Engineering: Time
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek

        feature_cols = [
            # --- Core Dynamic (Continuous) ---
            'price_vs_center', 'adx', 'atr_percent', 'cloud_width', 'ema_spread', 'rsi',
            # --- Microstructure ---
            'tick_volume', 'spread', 'candle_size', 'wick_upper', 'wick_lower',
            # --- State & Context ---
            'reclaim_state', 'is_reclaim_signal', 'bars_since_last', 'trend_direction', 'ema_cross_event',
            # --- Time ---
            'hour', 'day_of_week'
        ]
        
        # Ensure columns exist
        available_cols = [c for c in feature_cols if c in df.columns]
        
        X = df[available_cols]
        y = df['mfe'] # Target: Maximum Favorable Excursion
        
        print(f"✅ Prepared {len(X)} samples for Regression.")
        print(f"   Target Mean MFE: {y.mean():.2f}, Max: {y.max():.2f}")
        
        return X, y, available_cols

    def train(self, data_path: str):
        print("🔄 Loading Data for Regression...")
        df = self.load_data(data_path)
        
        X, y, feature_cols = self.prepare_data(df)
        
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        
        # Train LightGBM (Regression)
        print("🚀 Training LightGBM (Regression)...")
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test)
        
        params = {
            'objective': 'regression',
            'metric': ['rmse', 'mae'],
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
            num_boost_round=2000,
            valid_sets=[train_data, valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(100)
            ]
        )
        
        # Evaluation
        y_pred = self.model.predict(X_test)
        
        print("\n📊 Model Evaluation (Regression):")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
        
        # Simulation: What if we only traded top 20% predicted MFE?
        results = pd.DataFrame({'actual_mfe': y_test, 'predicted_mfe': y_pred})
        threshold = results['predicted_mfe'].quantile(0.8)
        top_trades = results[results['predicted_mfe'] > threshold]
        
        print(f"\n💰 Trade Simulation (Top 20% predicted):")
        print(f"Avg Actual MFE (All): {y_test.mean():.2f}")
        print(f"Avg Actual MFE (Top 20%): {top_trades['actual_mfe'].mean():.2f}")
        print(f"Improvement: {(top_trades['actual_mfe'].mean() / y_test.mean() - 1)*100:.1f}%")
        
        # Save Model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        self.model.save_model(MODEL_PATH)
        print(f"\n💾 Model saved to {MODEL_PATH}")
        
        # Feature Importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 15 Features (Gain):")
        print(importance.head(15))

if __name__ == "__main__":
    trainer = ScalpingTrainer()
    data_file = "training_data.csv"
    if os.path.exists(data_file):
        trainer.train(data_file)
    else:
        print(f"❌ File {data_file} not found.")
