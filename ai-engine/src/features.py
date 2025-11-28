import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class FeatureEngineer:
    """
    Scalping-Oriented Feature Engineering for AlphaOS.
    Focuses on micro-structure and volatility, relying on existing indicator signals.
    """
    
    def __init__(self):
        self.required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']

    def calculate_features(self, df: pd.DataFrame, dom_bids: List[Any] = None, dom_asks: List[Any] = None) -> pd.DataFrame:
        """
        Main pipeline to calculate all features.
        """
        if df.empty:
            return df

        # 1. Preprocessing
        df = self._preprocess(df)
        
        # 2. Volatility (Retained for dynamic SL/TP context)
        df = self._add_volatility_features(df)

        # 3. Technical Indicators (EMA, ADX, Scalping Features)
        df = self._add_technical_features(df)
        
        # 4. Session/Time Features
        df = self._add_time_features(df)
        
        # 5. Micro-structure / Liquidity (DOM)
        # Note: DOM features are usually point-in-time for the latest snapshot.
        if dom_bids and dom_asks:
            dom_features = self._calculate_dom_features(dom_bids, dom_asks)
            # Assign to the last row
            for k, v in dom_features.items():
                df.at[df.index[-1], k] = v
        
        # Fill NaNs resulting from rolling windows
        df = df.fillna(0)
        
        return df

    def get_latest_features(self, df: pd.DataFrame, dom_bids: List[Any] = None, dom_asks: List[Any] = None) -> Dict[str, Any]:
        """
        Returns the feature vector for the latest candle, ready for inference.
        """
        df_processed = self.calculate_features(df, dom_bids, dom_asks)
        if df_processed.empty:
            return {}
            
        latest = df_processed.iloc[-1]
        return latest.to_dict()

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.required_columns:
            if col not in df.columns:
                pass
        
        # Ensure numeric
        cols = ['open', 'high', 'low', 'close', 'volume']
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
        # Ensure time is datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s' if df['time'].dtype == 'int64' else None)
            
        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # ATR 14 (Still useful for normalizing spread/SL)
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # ATR Ratio (Current vs 24h/Daily avg)
        # Assuming 5min candles, 24h = 288 candles
        df['atr_24h'] = tr.rolling(288).mean()
        df['volatility_ratio'] = df['atr'] / df['atr_24h']
        
        return df

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # EMA
        df['ema_short'] = df['close'].ewm(span=6, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=24, adjust=False).mean()
        
        # ADX Calculation (Simplified)
        high = df['high']
        low = df['low']
        close = df['close']
        
        # TR is already calculated in _add_volatility_features but let's ensure it's available or recalculate
        # _add_volatility_features calculates 'atr' but not raw TR series in df. 
        # Let's recalculate TR for ADX to be safe and self-contained.
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Wilder's Smoothing (alpha = 1/14) -> com = 13
        atr_smooth = tr.ewm(com=13, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm).ewm(com=13, adjust=False).mean() / atr_smooth)
        minus_di = 100 * (pd.Series(minus_dm).ewm(com=13, adjust=False).mean() / atr_smooth)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        df['adx'] = dx.ewm(com=13, adjust=False).mean()
        
        # Scalping Features
        # 1. EMA Spread Ratio
        # (EMA_short - EMA_long) / ATR
        # Ensure ATR is present (calculated in _add_volatility_features)
        if 'atr' not in df.columns:
             df['atr'] = tr.rolling(14).mean()

        df['ema_spread'] = df['ema_short'] - df['ema_long']
        df['ema_spread_ratio'] = df['ema_spread'] / (df['atr'] + 1e-9)
        
        # 2. Trend Direction
        df['trend_direction'] = (df['ema_short'] > df['ema_long']).astype(int)
        
        # 3. Price vs Cloud (Position relative to EMA Long)
        # Normalized by ATR
        df['price_vs_cloud'] = (df['close'] - df['ema_long']) / (df['atr'] + 1e-9)
        
        # 4. ATR Percent
        df['atr_percent'] = (df['atr'] / df['close']) * 100
        
        # 5. Signal Density (Rolling count of potential signals)
        # This is hard without explicit signal logic, but we can proxy with trend flips
        # Trend Flip: Current trend != Prev trend
        trend_flip = (df['trend_direction'] != df['trend_direction'].shift(1)).astype(int)
        df['signal_density'] = trend_flip.rolling(20).sum() # Flips in last 20 bars
        
        # 6. HTF Aligned (Placeholder - requires HTF data)
        # We'll set to 1 (True) for now as we don't have HTF context here
        df['htf_aligned'] = 1
        
        # 7. Critical Filters OK (Composite)
        # Proxy: ADX > 20 AND Spread > 0.1 ATR
        spread_ok = np.abs(df['ema_spread']) > (0.1 * df['atr'])
        adx_ok = df['adx'] > 20
        df['critical_filters_ok'] = (spread_ok & adx_ok).astype(int)
        
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'time' not in df.columns:
            return df
            
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        df['day_of_week'] = df['time'].dt.dayofweek
        
        # Session Volatility Logic
        # Asian: 00-08 UTC, London: 08-16 UTC, NY: 13-21 UTC (approx)
        df['is_asian'] = df['hour'].between(0, 8).astype(int)
        df['is_london'] = df['hour'].between(8, 16).astype(int)
        df['is_ny'] = df['hour'].between(13, 21).astype(int)
        
        return df

    def _calculate_dom_features(self, bids: List[Any], asks: List[Any]) -> Dict[str, float]:
        """
        Calculate micro-structure features from Order Book (DOM).
        Expects lists of objects/dicts with 'price' and 'volume'.
        """
        if not bids or not asks:
            return {
                'ofi': 0,
                'spread_ticks': 0,
                'bid_ask_imbalance': 0,
                'depth_ratio': 1.0
            }
            
        best_bid = bids[0].price
        best_ask = asks[0].price
        
        spread = best_ask - best_bid
        
        # Volume Imbalance (Level 1)
        bid_vol_l1 = bids[0].volume
        ask_vol_l1 = asks[0].volume
        imbalance_l1 = (bid_vol_l1 - ask_vol_l1) / (bid_vol_l1 + ask_vol_l1 + 1e-9)
        
        # Depth Ratio (Total visible volume)
        total_bid_vol = sum([b.volume for b in bids])
        total_ask_vol = sum([a.volume for a in asks])
        depth_ratio = total_bid_vol / (total_ask_vol + 1e-9)
        
        return {
            'ofi': imbalance_l1, 
            'spread': spread,
            'bid_ask_imbalance': (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-9),
            'depth_ratio': depth_ratio
        }
