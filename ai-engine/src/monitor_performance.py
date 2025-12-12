
import os
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client
from tabulate import tabulate
import logging

# Configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

def get_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ Supabase credentials missing (SUPABASE_URL/KEY).")
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_data(supabase, days=7):
    """Fetch recent trades and scans."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    
    # 1. Real Trades
    res_trades = supabase.table("training_signals") \
        .select("*") \
        .gte("timestamp", cutoff) \
        .not_.is_("result_profit", "null") \
        .execute()
    trades = pd.DataFrame(res_trades.data) if res_trades.data else pd.DataFrame()
    
    # 2. Market Scans (Decisions)
    # We want ALL recent decisions to check distribution (BUY/SELL/WAIT)
    # Note: training_signals contains BUY/SELL, market_scans contains WAIT/SCAN
    # We need to fetch 'training_signals' again but without result_profit filter? 
    # Or just fetch recent rows from both tables.
    
    # Fetch recent BUY/SELL actions
    # Note: training_signals might not have 'ai_score' as column, it's inside 'ai_features' JSONB
    # We can fetch 'ai_features' and extract it in pandas, or just ignore score for now if not critical
    res_actions = supabase.table("training_signals") \
        .select("timestamp, symbol, action, ai_features") \
        .gte("timestamp", cutoff) \
        .execute()
    df_actions = pd.DataFrame(res_actions.data) if res_actions.data else pd.DataFrame()
    
    # Extract score from ai_features if available
    if not df_actions.empty and 'ai_features' in df_actions.columns:
        # ai_features is likely a dict or None
        def extract_score(x):
            if isinstance(x, dict): return x.get('ai_score', 0)
            return 0
        df_actions['ai_score'] = df_actions['ai_features'].apply(extract_score)
        # Drop ai_features to match columns with scans
        df_actions = df_actions.drop(columns=['ai_features'])

    # Fetch recent WAIT/SCAN actions from market_scans (which HAS ai_score column)
    res_scans = supabase.table("market_scans") \
        .select("timestamp, symbol, action, ai_score") \
        .gte("timestamp", cutoff) \
        .execute()
    df_scans = pd.DataFrame(res_scans.data) if res_scans.data else pd.DataFrame()
    
    # Combine for decision distribution
    decisions = pd.concat([df_actions, df_scans], ignore_index=True)
    
    return trades, decisions

def calculate_metrics(trades, decisions):
    print("\n" + "="*60)
    print(f"🤖 AI PERFORMANCE REPORT (Last 7 Days) - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    # --- 1. Real Trade Performance ---
    if not trades.empty:
        # Win Rate
        trades['is_win'] = trades['result_profit'] > 0
        total_trades = len(trades)
        wins = trades['is_win'].sum()
        win_rate = (wins / total_trades) * 100
        total_pnl = trades['result_profit'].sum()
        
        print(f"\n📊 REAL TRADING RESULTS:")
        print(f"   • Total Trades:   {total_trades}")
        print(f"   • Win Rate:       {win_rate:.1f}%  ({wins}/{total_trades})")
        print(f"   • Total PnL:      ${total_pnl:.2f}")
        
        # Breakdown by Symbol
        print(f"\n   Stats by Symbol:")
        sym_stats = trades.groupby('symbol').agg(
            Count=('result_profit', 'count'),
            WinRate=('is_win', lambda x: (x.sum()/len(x))*100),
            PnL=('result_profit', 'sum')
        ).reset_index()
        print(tabulate(sym_stats, headers="keys", tablefmt="simple", floatfmt=".1f"))
    else:
        print("\n📊 REAL TRADING RESULTS: No closed trades found in last 7 days.")

    # --- 2. Decision Distribution ---
    if not decisions.empty:
        print(f"\n🧠 AI DECISION DISTRIBUTION:")
        total_decisions = len(decisions)
        
        # Count by Action
        action_counts = decisions['action'].value_counts()
        
        buys = action_counts.get('BUY', 0)
        sells = action_counts.get('SELL', 0)
        waits = action_counts.get('WAIT', 0)
        scans = action_counts.get('SCAN', 0)
        
        trade_count = buys + sells
        activity_rate = (trade_count / total_decisions) * 100
        
        print(f"   • Total Opportunities Analyzed: {total_decisions}")
        print(f"   • Trade Taken (BUY/SELL):       {trade_count} ({activity_rate:.1f}%)")
        print(f"   • Trade Avoided (WAIT/SCAN):    {waits + scans} ({100 - activity_rate:.1f}%)")
        
        if trade_count > 0:
            print(f"   • Buy/Sell Ratio:               {buys}:{sells}")
            
    else:
        print("\n🧠 AI DECISION DISTRIBUTION: No data recorded yet.")
        
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    supabase = get_supabase()
    if supabase:
        try:
            trades, decisions = fetch_data(supabase)
            calculate_metrics(trades, decisions)
        except Exception as e:
            print(f"Error fetching data: {e}")
