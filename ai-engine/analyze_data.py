import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

# Set style
plt.style.use('ggplot')

OUTPUT_DIR = 'analysis_results'
DATA_FILE = 'training_data.csv'

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Convert timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='s')
    
    return df

def calculate_metrics(df):
    print("Calculating metrics...")
    
    # Calculate PnL
    # For BUY/RECLAIM_BUY: exit_price - price
    # For SELL/RECLAIM_SELL: price - exit_price
    
    df['pnl_points'] = 0.0
    
    mask_buy = df['action'].isin(['BUY', 'RECLAIM_BUY'])
    mask_sell = df['action'].isin(['SELL', 'RECLAIM_SELL'])
    
    df.loc[mask_buy, 'pnl_points'] = df.loc[mask_buy, 'exit_price'] - df.loc[mask_buy, 'price']
    df.loc[mask_sell, 'pnl_points'] = df.loc[mask_sell, 'price'] - df.loc[mask_sell, 'exit_price']
    
    # Assuming XAUUSD, 1 point = $1 profit per lot (simplified, usually 1 pip = $1 or $10 depending on contract)
    # Let's just use points for now as raw PnL
    
    df['cumulative_pnl'] = df['pnl_points'].cumsum()
    
    # Win/Loss
    df['win'] = df['outcome'] == 1.0
    
    total_trades = len(df)
    wins = df['win'].sum()
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    avg_win = df[df['win']]['pnl_points'].mean()
    avg_loss = df[~df['win']]['pnl_points'].mean()
    
    profit_factor = abs(df[df['win']]['pnl_points'].sum() / df[~df['win']]['pnl_points'].sum()) if df[~df['win']]['pnl_points'].sum() != 0 else float('inf')
    
    # Drawdown
    df['peak'] = df['cumulative_pnl'].cummax()
    df['drawdown'] = df['cumulative_pnl'] - df['peak']
    max_drawdown = df['drawdown'].min()
    
    metrics = {
        'Total Trades': total_trades,
        'Win Rate': f"{win_rate:.2f}%",
        'Total PnL (Points)': f"{df['pnl_points'].sum():.2f}",
        'Average Win': f"{avg_win:.2f}",
        'Average Loss': f"{avg_loss:.2f}",
        'Profit Factor': f"{profit_factor:.2f}",
        'Max Drawdown': f"{max_drawdown:.2f}"
    }
    
    return df, metrics

def plot_equity_curve(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['cumulative_pnl'], label='Equity Curve', color='blue')
    plt.title('Equity Curve (Cumulative PnL)')
    plt.xlabel('Date')
    plt.ylabel('PnL (Points)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'equity_curve.png'))
    plt.close()

def plot_monthly_pnl(df):
    df['month_year'] = df['timestamp'].dt.to_period('M')
    monthly_pnl = df.groupby('month_year')['pnl_points'].sum()
    
    plt.figure(figsize=(12, 6))
    monthly_pnl.plot(kind='bar', color=np.where(monthly_pnl >= 0, 'green', 'red'))
    plt.title('Monthly PnL')
    plt.xlabel('Month')
    plt.ylabel('PnL (Points)')
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'monthly_pnl.png'))
    plt.close()

def plot_trade_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df['pnl_points'], bins=50, color='purple', alpha=0.7)
    plt.title('Trade PnL Distribution')
    plt.xlabel('PnL (Points)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'pnl_distribution.png'))
    plt.close()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df = load_data(DATA_FILE)
    df, metrics = calculate_metrics(df)
    
    print("Analysis Results:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
        
    with open(os.path.join(OUTPUT_DIR, 'metrics.txt'), 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
            
    plot_equity_curve(df)
    plot_monthly_pnl(df)
    plot_trade_distribution(df)
    print(f"Charts saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
