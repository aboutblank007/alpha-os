import pandas as pd
import numpy as np

def inspect():
    df = pd.read_csv('training_data.csv')
    print(f"Total rows: {len(df)}")
    
    if 'has_outcome' in df.columns:
        df = df[df['has_outcome'] == True]
        print(f"Rows with outcome: {len(df)}")
        
    y = df['outcome']
    print(f"Target Mean: {y.mean():.4f}")
    
    feature_cols = [
        'atr', 'adx', 
        'distance_ok', 'slope_ok', 'trend_filter_ok', 'htf_trend_ok', 
        'volatility_ok', 'chop_ok', 'spread_ok', 
        'bars_since_last', 'trend_direction', 'ema_cross_event', 
        'ema_spread', 'atr_percent', 'reclaim_state', 'is_reclaim_signal', 
        'price_vs_center', 'cloud_width'
    ]
    
    print("\nFeature Statistics:")
    print(df[feature_cols].describe().T[['mean', 'std', 'min', '50%', 'max']])
    
    print("\nCorrelations with Target:")
    corrs = df[feature_cols].corrwith(df['outcome']).sort_values(ascending=False)
    print(corrs)
    
    print("\nBoolean Feature Counts:")
    for col in ['distance_ok', 'slope_ok', 'trend_filter_ok', 'htf_trend_ok', 'volatility_ok', 'chop_ok', 'spread_ok']:
        if col in df.columns:
            print(f"{col}: {df[col].value_counts().to_dict()}")

if __name__ == "__main__":
    inspect()
