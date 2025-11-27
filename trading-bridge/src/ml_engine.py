import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import json
from typing import Dict, Any, Tuple

class FeatureEngineer:
    """
    Responsible for calculating technical indicators and features from raw OHLCV data.
    """
    def __init__(self):
        pass

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators based on OHLCV data.
        Assumes df has columns: 'time', 'open', 'high', 'low', 'close', 'tick_volume'
        """
        if df.empty:
            return df

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'tick_volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 1. RSI (Relative Strength Index) - 14 period
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 2. Moving Averages
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Distance from MA
        df['dist_sma_50'] = (df['close'] - df['sma_50']) / df['sma_50']
        df['dist_sma_200'] = (df['close'] - df['sma_200']) / df['sma_200']

        # 3. Volatility (ATR - Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volatility Ratio (Current ATR / 24h ATR) - assuming H1 candles, 24h = 24 periods
        df['atr_24h'] = true_range.rolling(window=24).mean()
        df['volatility_ratio'] = df['atr'] / df['atr_24h']

        # 4. MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # 5. Time Features
        df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek

        # Fill NaNs
        df = df.fillna(0)
        
        return df

    def get_latest_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Returns the feature vector for the last row as a dictionary.
        """
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        # Select only feature columns (exclude raw OHLCV if needed, but keeping for now)
        # In production, we should strictly define the feature list used during training.
        return latest.to_dict()

class LightGBMPredictor:
    """
    Wraps the LightGBM model for inference.
    """
    def __init__(self, model_path: str = "model_v1.txt"):
        self.model = None
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = lgb.Booster(model_file=self.model_path)
                print(f"✅ Loaded LightGBM model from {self.model_path}")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
        else:
            print(f"⚠️ Model file {self.model_path} not found. Running in dummy mode.")

    def predict(self, features: Dict[str, Any]) -> float:
        """
        Returns the probability of success (Class 1).
        """
        if not self.model:
            # Dummy logic for testing without a trained model
            # Return a random-ish but deterministic value based on RSI if available
            rsi = features.get('rsi', 50)
            if rsi > 70 or rsi < 30:
                return 0.65 # High confidence on extremes
            return 0.45

        # Prepare input vector (must match training feature order)
        # This is a simplification. In real usage, we need a strict schema.
        # For now, we assume the model handles dict input or we convert to list based on known features.
        # LightGBM Booster.predict expects a 2D array.
        
        # TODO: Implement strict feature ordering matching the training phase
        # feature_vector = [features[f] for f in self.feature_names]
        
        # Placeholder return
        return 0.5

# Singleton instance
feature_engineer = FeatureEngineer()
predictor = LightGBMPredictor()
