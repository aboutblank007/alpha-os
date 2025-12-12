import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass

    def _infer_pip_scale(self, price: float) -> float:
        """
        推断不同品种的“每点价格”刻度，默认与 1e-4 量级对齐。
        这样可在多资产场景下把价格差异统一到相似的尺度。
        """
        if price is None:
            return 1e-4
        p = abs(price)
        if p >= 20000:   # BTC / 高价指数
            return 1.0
        if p >= 2000:    # XAU / 高价商品
            return 0.1
        if p >= 200:     # JPY 货币对、部分指数
            return 0.01
        if p >= 2:       # 常规外汇
            return 0.0001
        return 0.0001

    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Handle initial NaN
        rsi = rsi.fillna(50)
        return rsi

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index)"""
        # True Range
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift(1))
        df['tr2'] = abs(df['low'] - df['close'].shift(1))
        tr = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        
        # Directional Movement
        up_move = df['high'] - df['high'].shift(1)
        down_move = df['low'].shift(1) - df['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smooth
        # Wilder's Smoothing is roughly alpha=1/period
        tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / tr_smooth)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / tr_smooth)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return adx

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add enhanced derived features (volatility, volume density, etc.)
        Used for Filter Model v2.
        """
        df = df.copy()

        if 'pip_scale' not in df.columns:
            df['pip_scale'] = df['close'].apply(self._infer_pip_scale)
        
        # 1. Volatility Dynamics
        if 'spread' in df.columns and 'atr' in df.columns:
            atr_pips = df['atr'] / (df['pip_scale'] + 1e-8)
            spread_pips = df['spread'] / (df['pip_scale'] + 1e-8)
            df['spread_to_atr'] = spread_pips / (atr_pips + 1e-5)
        else:
            df['spread_to_atr'] = 0.0
        df['spread_to_atr'] = df['spread_to_atr'].clip(lower=0, upper=100)
            
        # 2. Volume Dynamics
        if 'tick_volume' in df.columns and 'candle_size' in df.columns:
            candle_pips = df['candle_size'] / (df['pip_scale'] + 1e-8)
            df['volume_density'] = df['tick_volume'] / (candle_pips + 1e-3)
        else:
            df['volume_density'] = 0.0
        df['volume_density'] = df['volume_density'].clip(lower=-1e5, upper=1e5)
            
        # 3. Trend Strength
        if 'price_vs_center' in df.columns and 'center' in df.columns and 'atr' in df.columns:
            atr_pips = df['atr'] / (df['pip_scale'] + 1e-8)
            df['cloud_dist_atr'] = df['price_vs_center'].abs() / (atr_pips + 1e-5)
        else:
            df['cloud_dist_atr'] = 0.0
        df['cloud_dist_atr'] = df['cloud_dist_atr'].clip(lower=0, upper=1e3)
            
        # 4. Wick Ratios
        if 'candle_size' in df.columns and 'wick_upper' in df.columns and 'wick_lower' in df.columns:
            candle_body = (df['candle_size'] - df['wick_upper'] - df['wick_lower']).abs()
            df['wick_ratio'] = (df['wick_upper'] + df['wick_lower']) / (candle_body + 1e-5)
        else:
            df['wick_ratio'] = 0.0
        df['wick_ratio'] = df['wick_ratio'].clip(lower=0, upper=50)
            
        return df

    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 1-5m microstructure features:
        - Volume Shock
        - Order Imbalance Proxy (using tick volume)
        - Volatility Shock
        """
        df = df.copy()
        
        # Ensure base features exist
        if 'atr' not in df.columns:
            df['atr'] = self.calculate_atr(df)
        if 'candle_size' not in df.columns:
            df['candle_size'] = (df['high'] - df['low'])

        # 1. Volume Shock: Vol > 2 * EMA(Vol, 20)
        if 'tick_volume' in df.columns:
            vol_ema = df['tick_volume'].ewm(span=20).mean()
            df['volume_shock'] = (df['tick_volume'] / (vol_ema + 1e-5))
        else:
            df['volume_shock'] = 0.0
        df['volume_shock'] = df['volume_shock'].clip(lower=0, upper=50)
        
        # 2. Volatility Shock: Range > 2 * ATR
        df['volatility_shock'] = df['candle_size'] / (df['atr'] + 1e-5)
        df['volatility_shock'] = df['volatility_shock'].clip(lower=0, upper=50)
        
        # 3. Order Imbalance Proxy (Close Location Value)
        # CLV = ((C - L) - (H - C)) / (H - L) -> ranges -1 to 1
        # High CLV + High Vol -> Buying Pressure
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['candle_size'] + 1e-5)
        if 'tick_volume' in df.columns:
            df['order_imbalance_proxy'] = clv * df['tick_volume']
        else:
            df['order_imbalance_proxy'] = 0.0
        df['order_imbalance_proxy'] = df['order_imbalance_proxy'].clip(lower=-1e7, upper=1e7)
        
        return df

    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical features required by the Pure AI model.
        Matches PivotTrendSignals.mq5 logic.
        """
        df = df.copy()
        
        # Basic Indicators
        df['ema_short'] = self.calculate_ema(df['close'], 12) # Fast EMA
        df['ema_long'] = self.calculate_ema(df['close'], 26)  # Slow EMA
        df['atr'] = self.calculate_atr(df, 14)
        df['adx'] = self.calculate_adx(df, 14)
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        
        # Derived Features (AlphaOS specific)
        df['center'] = (df['ema_short'] + df['ema_long']) / 2
        df['pip_scale'] = df['close'].apply(self._infer_pip_scale)
        df['price_vs_center'] = (df['close'] - df['center']) / (df['pip_scale'] + 1e-8)
        df['cloud_width'] = (df['ema_short'] - df['ema_long']).abs() / (df['pip_scale'] + 1e-8)
        df['ema_spread'] = df['cloud_width'] # Alias
        df['atr_percent'] = df['atr'] / df['close'] * 100
        
        # Candle Features
        df['candle_size'] = (df['high'] - df['low'])
        df['wick_upper'] = (df['high'] - df[['open', 'close']].max(axis=1))
        df['wick_lower'] = (df[['open', 'close']].min(axis=1) - df['low'])
        
        # Log Returns (Pure Price Action)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility_5'] = df['log_return'].rolling(window=5).std()
        
        return df.dropna()
