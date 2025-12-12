import polars as pl
import numpy as np

class FeatureEngineerPolars:
    def __init__(self):
        pass

    def _infer_pip_scale_expr(self, col_name='close') -> pl.Expr:
        """
        Polars Expression to infer pip scale based on price.
        """
        return (
            pl.when(pl.col(col_name) >= 20000).then(1.0)
            .when(pl.col(col_name) >= 2000).then(0.1)
            .when(pl.col(col_name) >= 200).then(0.01)
            .when(pl.col(col_name) >= 2).then(0.0001)
            .otherwise(0.0001)
        )

    def calculate_ema(self, df: pl.DataFrame, col_name: str, period: int, alias: str = None) -> pl.DataFrame:
        """
        Calculate EMA using polars ewm_mean.
        """
        if alias is None:
            alias = f"ema_{period}"
        
        # Defensive drop to prevent DuplicateError in some Polars versions
        if alias in df.columns:
            df = df.drop(alias)
            
        # Polars ewm_mean uses 'span' parameter similar to pandas
        return df.with_columns(
            pl.col(col_name).ewm_mean(span=period, adjust=False).alias(alias)
        )

    def calculate_atr(self, df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """
        Calculate ATR.
        TR = Max(H-L, |H-Cp|, |L-Cp|)
        """
        # Calculate True Range
        # We need previous close
        df = df.with_columns(
            pl.col('close').shift(1).alias('prev_close')
        )
        
        tr_expr = pl.max_horizontal(
            (pl.col('high') - pl.col('low')).abs(),
            (pl.col('high') - pl.col('prev_close')).abs(),
            (pl.col('low') - pl.col('prev_close')).abs()
        ).alias('tr')

        df = df.with_columns(tr_expr)
        
        # ATR = EMA(TR, alpha=1/period) match Pandas alpha logic (ewm uses alpha via span or com or alpha directly)
        # Pandas: ewm(alpha=1/period, adjust=False)
        # Polars ewm_mean supports 'alpha' directly
        if 'atr' in df.columns:
            df = df.drop('atr')
            
        return df.with_columns(
            pl.col('tr').ewm_mean(alpha=1.0/period, adjust=False).alias('atr')
        ).drop(['prev_close', 'tr'])

    def calculate_rsi(self, df: pl.DataFrame, col_name: str = 'close', period: int = 14) -> pl.DataFrame:
        """
        Calculate RSI.
        """
        delta = pl.col(col_name).diff()
        
        gain = pl.when(delta > 0).then(delta).otherwise(0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0)
        
        # Rolling mean (simple moving average for initial, usually RMA for RSI)
        # Standard RSI uses Wilder's Smoothing which is equivalent to EMA(alpha=1/N)
        # Pandas features.py used rolling(window=period).mean() which is SMA.
        # To maintain 1:1 parity with existing Pandas logic:
        avg_gain = gain.rolling_mean(window_size=period)
        avg_loss = loss.rolling_mean(window_size=period)

        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        if 'rsi' in df.columns:
            df = df.drop('rsi')
            
        return df.with_columns([
            rsi.fill_null(50).alias('rsi')
        ])

    def calculate_adx(self, df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """
        Calculate ADX.
        Matches Pandas implementation:
        PD: ewm(alpha=1/period) which is Wilder's smoothing.
        """
        df = df.with_columns(
            pl.col('close').shift(1).alias('prev_close'),
            pl.col('high').shift(1).alias('prev_high'),
            pl.col('low').shift(1).alias('prev_low')
        )

        # TR
        tr = pl.max_horizontal(
            (pl.col('high') - pl.col('low')).abs(),
            (pl.col('high') - pl.col('prev_close')).abs(),
            (pl.col('low') - pl.col('prev_close')).abs()
        )

        # DM
        up_move = pl.col('high') - pl.col('prev_high')
        down_move = pl.col('prev_low') - pl.col('low')

        plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0)
        minus_dm = pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0)

        # Smoothing (Wilder's alpha = 1/period)
        # In Pandas code: tr.ewm(alpha=1/period, adjust=False).mean()
        tr_smooth = tr.ewm_mean(alpha=1.0/period, adjust=False)
        plus_di = 100 * (plus_dm.ewm_mean(alpha=1.0/period, adjust=False) / tr_smooth)
        minus_di = 100 * (minus_dm.ewm_mean(alpha=1.0/period, adjust=False) / tr_smooth)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx = dx.ewm_mean(alpha=1.0/period, adjust=False)

        if 'adx' in df.columns:
            df = df.drop('adx')

        return df.with_columns(adx.alias('adx')).drop(['prev_close', 'prev_high', 'prev_low'])

    def add_technical_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add all technical features (Parity with features.py).
        """
        # Basic Indicators
        df = self.calculate_ema(df, 'close', 12, 'ema_short')
        df = self.calculate_ema(df, 'close', 26, 'ema_long')
        df = self.calculate_atr(df, 14)
        df = self.calculate_adx(df, 14)
        df = self.calculate_rsi(df, 'close', 14)

        # Derived Features
        # Step-by-step to be safe (no penalty in Polars lazy graph)
        
        # 1. Center
        df = df.with_columns(
            ((pl.col('ema_short') + pl.col('ema_long')) / 2).alias('center')
        )

        # 2. Pip Scale
        df = df.with_columns(
            self._infer_pip_scale_expr('close').alias('pip_scale')
        )

        # 3. Price vs Center
        df = df.with_columns(
            ((pl.col('close') - pl.col('center')) / (pl.col('pip_scale') + 1e-8)).alias('price_vs_center')
        )
            
        # 4. Cloud Width
        df = df.with_columns(
            ((pl.col('ema_short') - pl.col('ema_long')).abs() / (pl.col('pip_scale') + 1e-8)).alias('cloud_width')
        )

        # 5. ATR Percent
        df = df.with_columns(
            (pl.col('atr') / pl.col('close') * 100).alias('atr_percent')
        )
            
        # 6. Candle Size
        df = df.with_columns(
            (pl.col('high') - pl.col('low')).alias('candle_size')
        )
            
        # 7. Wicks
        df = df.with_columns([
            (pl.col('high') - pl.max_horizontal('open', 'close')).alias('wick_upper'),
            (pl.min_horizontal('open', 'close') - pl.col('low')).alias('wick_lower')
        ])
            
        # 8. Log Returns
        df = df.with_columns(
            (pl.col('close') / pl.col('close').shift(1)).log().alias('log_return')
        )

        # Aliases
        df = df.with_columns(
            pl.col('cloud_width').alias('ema_spread')
        )

        # Volatility 5 (Rolling Std)
        df = df.with_columns(
            pl.col('log_return').rolling_std(window_size=5).alias('volatility_5')
        )

        return df.drop_nulls()

    def add_derived_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add enhanced derived features (v2 Model).
        """
        # Ensure pip_scale exists
        if 'pip_scale' not in df.columns:
            df = df.with_columns(self._infer_pip_scale_expr('close').alias('pip_scale'))

        # Prepare Exprs
        atr_pips = pl.col('atr') / (pl.col('pip_scale') + 1e-8)
        spread_pips = pl.col('spread') / (pl.col('pip_scale') + 1e-8)
        
        # Candle Pips - candle_size should be there, but check
        if 'candle_size' not in df.columns:
             df = df.with_columns((pl.col('high') - pl.col('low')).alias('candle_size'))
             
        candle_pips = pl.col('candle_size') / (pl.col('pip_scale') + 1e-8)
        
        # 1. Spread to ATR
        spread_to_atr = (spread_pips / (atr_pips + 1e-5)).clip(0, 100)
        
        # 2. Volume Density
        # Note: tick_volume might not implement division nicely if Int, cast to Float
        volume_density = (pl.col('tick_volume').cast(pl.Float64) / (candle_pips + 1e-3)).clip(-1e5, 1e5)
        
        # 3. Cloud Dist ATR
        cloud_dist_atr = (pl.col('price_vs_center').abs() / (atr_pips + 1e-5)).clip(0, 1e3)
        
        # 4. Wick Ratio
        # ensure wick_upper/lower exist
        if 'wick_upper' not in df.columns:
             df = df.with_columns([
                (pl.col('high') - pl.max_horizontal('open', 'close')).alias('wick_upper'),
                (pl.min_horizontal('open', 'close') - pl.col('low')).alias('wick_lower')
             ])
        
        body = (pl.col('candle_size') - pl.col('wick_upper') - pl.col('wick_lower')).abs()
        wick_ratio = ((pl.col('wick_upper') + pl.col('wick_lower')) / (body + 1e-5)).clip(0, 50)

        df = df.with_columns([
            spread_to_atr.alias('spread_to_atr'),
            volume_density.alias('volume_density'),
            cloud_dist_atr.alias('cloud_dist_atr'),
            wick_ratio.alias('wick_ratio')
        ])
        
        return df

    def add_microstructure_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add microstructure features.
        """
        # Ensure base features
        if 'atr' not in df.columns:
            df = self.calculate_atr(df)
        if 'candle_size' not in df.columns:
            df = df.with_columns((pl.col('high') - pl.col('low')).alias('candle_size'))
            
        # 1. Volume Shock
        # Use explicit alias for intermediate to avoid collision risk?
        vol_ema_alias = 'vol_ema_temp'
        if vol_ema_alias in df.columns: df = df.drop(vol_ema_alias)
        
        df = df.with_columns(pl.col('tick_volume').ewm_mean(span=20).alias(vol_ema_alias))
        
        volume_shock = (pl.col('tick_volume') / (pl.col(vol_ema_alias) + 1e-5)).clip(0, 50)
        
        # 2. Volatility Shock
        volatility_shock = (pl.col('candle_size') / (pl.col('atr') + 1e-5)).clip(0, 50)
        
        # 3. Order Imbalance Proxy (CLV * Vol)
        clv = ((pl.col('close') - pl.col('low')) - (pl.col('high') - pl.col('close'))) / (pl.col('candle_size') + 1e-5)
        order_imbalance = (clv * pl.col('tick_volume')).clip(-1e7, 1e7)
        
        df = df.with_columns([
            volume_shock.alias('volume_shock'),
            volatility_shock.alias('volatility_shock'),
            order_imbalance.alias('order_imbalance_proxy')
        ])
        
        # Cleanup temp
        if vol_ema_alias in df.columns: df = df.drop(vol_ema_alias)
        
        return df

    def process_all(self, candles_dict_list: list) -> pl.DataFrame:
        """
        Main entry point: Convert list of dicts -> Polars DF -> Add all features.
        """
        if not candles_dict_list:
            return pl.DataFrame()
            
        # Create DataFrame
        df = pl.DataFrame(candles_dict_list)
        
        # Ensure columns are float for calculations
        cols_to_float = ['open', 'high', 'low', 'close', 'tick_volume', 'real_volume', 'aggressor_buy_vol', 'aggressor_sell_vol']
        # Filter only existing columns
        existing_cols = [c for c in cols_to_float if c in df.columns]
        
        # Cast
        if existing_cols:
             df = df.with_columns([pl.col(c).cast(pl.Float64) for c in existing_cols])

        # Add Features in Pipeline
        df = self.add_technical_features(df)
        
        # We need 'spread' for derived features. If not in input, assume 0 or 1 pip
        if 'spread' not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias('spread'))
            
        df = self.add_derived_features(df)
        df = self.add_microstructure_features(df)
        
        return df
