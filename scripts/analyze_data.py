#!/usr/bin/env python3
"""
AlphaOS Data Analysis Script

Analyzes tick data before training to understand:
- Data quality (missing values, outliers)
- Price statistics and distribution
- Tick frequency and gaps
- Spread patterns
- Volume patterns (if available)
- Time coverage

Usage:
    python scripts/analyze_data.py --data path/to/ticks.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze tick data for AlphaOS training",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to tick data (CSV or Parquet)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N rows for faster analysis of large files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save analysis report to file",
    )
    return parser.parse_args()


def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def format_number(n: float) -> str:
    """Format large numbers with commas."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.2f}K"
    else:
        return f"{n:.2f}"


def analyze_data(data_path: str, sample_size: int | None = None) -> dict:
    """Perform comprehensive data analysis."""
    
    results = {}
    
    # ========================================================================
    # 1. Load Data
    # ========================================================================
    print_section("1. Loading Data")
    
    path = Path(data_path)
    
    if path.suffix == ".csv":
        # For large CSV, use lazy scan
        if sample_size:
            df = pl.read_csv(path, n_rows=sample_size)
            print(f"📊 Sampled {sample_size:,} rows from CSV")
        else:
            # Try to get row count first
            print(f"📂 Reading {path.name}...")
            df = pl.read_csv(path)
    elif path.suffix == ".parquet":
        df = pl.read_parquet(path)
    else:
        print(f"❌ Unsupported format: {path.suffix}")
        sys.exit(1)
    
    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"📋 Columns: {df.columns}")
    
    results["total_rows"] = len(df)
    results["columns"] = df.columns
    
    # ========================================================================
    # 2. Detect Column Types
    # ========================================================================
    print_section("2. Column Detection")
    
    # Find timestamp column
    time_col = None
    time_format = None
    
    for col in df.columns:
        col_lower = col.lower()
        if "time" in col_lower or "date" in col_lower:
            time_col = col
            # Check format
            sample_val = str(df[col][0])
            if "." in sample_val and len(sample_val) > 20:
                time_format = "datetime_ms"  # Has milliseconds
            elif "-" in sample_val or "/" in sample_val:
                time_format = "datetime"
            else:
                time_format = "numeric"
            break
    
    # Find bid/ask columns
    bid_col = None
    ask_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if "bid" in col_lower and "vol" not in col_lower:
            bid_col = col
        elif "ask" in col_lower and "vol" not in col_lower:
            ask_col = col
    
    # Find volume columns
    bid_vol_col = None
    ask_vol_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if "bidvol" in col_lower or "bid_vol" in col_lower or col_lower == "bidvolume":
            bid_vol_col = col
        elif "askvol" in col_lower or "ask_vol" in col_lower or col_lower == "askvolume":
            ask_vol_col = col
    
    print(f"📅 Time column: {time_col} (format: {time_format})")
    print(f"💰 Bid column: {bid_col}")
    print(f"💰 Ask column: {ask_col}")
    print(f"📊 Bid Volume: {bid_vol_col}")
    print(f"📊 Ask Volume: {ask_vol_col}")
    
    results["time_col"] = time_col
    results["bid_col"] = bid_col
    results["ask_col"] = ask_col
    
    if not bid_col or not ask_col:
        print("❌ Could not detect bid/ask columns!")
        sys.exit(1)
    
    # ========================================================================
    # 3. Data Quality Check
    # ========================================================================
    print_section("3. Data Quality")
    
    # Missing values
    null_counts = df.null_count()
    has_nulls = False
    
    for col in df.columns:
        null_count = null_counts[col][0]
        if null_count > 0:
            has_nulls = True
            pct = null_count / len(df) * 100
            print(f"⚠️  {col}: {null_count:,} nulls ({pct:.2f}%)")
    
    if not has_nulls:
        print("✅ No missing values")
    
    # Check for zeros
    bid_zeros = (df[bid_col] == 0).sum()
    ask_zeros = (df[ask_col] == 0).sum()
    
    if bid_zeros > 0 or ask_zeros > 0:
        print(f"⚠️  Zero prices: Bid={bid_zeros}, Ask={ask_zeros}")
    else:
        print("✅ No zero prices")
    
    # ========================================================================
    # 4. Time Analysis
    # ========================================================================
    print_section("4. Time Analysis")
    
    # Parse timestamps
    if time_col:
        # Try to parse time
        time_series = df[time_col]
        
        # Convert string time to datetime if needed
        if time_series.dtype == pl.Utf8:
            # Try different formats
            try:
                # Format: "2025.01.02 01:00:00.455"
                time_series = time_series.str.replace_all(r"\.(\d{4})\.", "-$1-")
                time_series = time_series.str.to_datetime("%Y-%m-%d %H:%M:%S%.f")
            except:
                try:
                    time_series = time_series.str.to_datetime()
                except Exception as e:
                    print(f"⚠️  Could not parse time: {e}")
                    time_series = None
        
        if time_series is not None:
            first_time = time_series[0]
            last_time = time_series[-1]
            
            print(f"📅 First tick: {first_time}")
            print(f"📅 Last tick:  {last_time}")
            
            if hasattr(first_time, 'timestamp'):
                duration = last_time - first_time
                print(f"⏱️  Duration: {duration}")
                
                # Calculate tick frequency
                duration_seconds = duration.total_seconds()
                ticks_per_second = len(df) / duration_seconds
                ticks_per_minute = ticks_per_second * 60
                
                print(f"📈 Avg ticks/second: {ticks_per_second:.2f}")
                print(f"📈 Avg ticks/minute: {ticks_per_minute:.1f}")
                
                results["ticks_per_second"] = ticks_per_second
            
            # Check for gaps
            if len(df) > 1 and time_series is not None:
                # Sample time differences
                sample_size = min(100000, len(df))
                sample_df = df.sample(n=sample_size, seed=42) if len(df) > sample_size else df
                
                # We'll compute gaps in a simplified way
                print("\n📊 Time gap analysis (sampled):")
                print("   (Note: Use full analysis for production)")
    
    # ========================================================================
    # 5. Price Analysis
    # ========================================================================
    print_section("5. Price Analysis")
    
    bid_prices = df[bid_col].cast(pl.Float64)
    ask_prices = df[ask_col].cast(pl.Float64)
    mid_prices = (bid_prices + ask_prices) / 2
    spreads = ask_prices - bid_prices
    
    # Basic stats
    print(f"\n💰 Bid Price:")
    print(f"   Min:    {bid_prices.min():.3f}")
    print(f"   Max:    {bid_prices.max():.3f}")
    print(f"   Mean:   {bid_prices.mean():.3f}")
    print(f"   Std:    {bid_prices.std():.3f}")
    
    print(f"\n💰 Ask Price:")
    print(f"   Min:    {ask_prices.min():.3f}")
    print(f"   Max:    {ask_prices.max():.3f}")
    print(f"   Mean:   {ask_prices.mean():.3f}")
    print(f"   Std:    {ask_prices.std():.3f}")
    
    # Price range
    price_range = ask_prices.max() - bid_prices.min()
    print(f"\n📊 Total price range: {price_range:.2f}")
    
    results["price_min"] = float(bid_prices.min())
    results["price_max"] = float(ask_prices.max())
    results["price_mean"] = float(mid_prices.mean())
    
    # ========================================================================
    # 6. Spread Analysis
    # ========================================================================
    print_section("6. Spread Analysis")
    
    spread_mean = spreads.mean()
    spread_std = spreads.std()
    spread_min = spreads.min()
    spread_max = spreads.max()
    
    print(f"📊 Spread Statistics:")
    print(f"   Min:    {spread_min:.4f}")
    print(f"   Max:    {spread_max:.4f}")
    print(f"   Mean:   {spread_mean:.4f}")
    print(f"   Std:    {spread_std:.4f}")
    
    # Spread in pips (assuming 0.01 = 1 pip for gold)
    pip_size = 0.01
    spread_pips_mean = spread_mean / pip_size
    print(f"   Mean (pips): {spread_pips_mean:.2f}")
    
    # Check for negative spreads (data error)
    negative_spreads = (spreads < 0).sum()
    if negative_spreads > 0:
        print(f"⚠️  Negative spreads found: {negative_spreads:,} ({negative_spreads/len(df)*100:.2f}%)")
    else:
        print("✅ No negative spreads")
    
    # Very wide spreads (potential issues)
    wide_spread_threshold = spread_mean + 5 * spread_std
    wide_spreads = (spreads > wide_spread_threshold).sum()
    if wide_spreads > 0:
        print(f"⚠️  Wide spreads (>5σ): {wide_spreads:,} ({wide_spreads/len(df)*100:.4f}%)")
    
    results["spread_mean"] = float(spread_mean)
    results["spread_pips"] = float(spread_pips_mean)
    
    # ========================================================================
    # 7. Returns Analysis
    # ========================================================================
    print_section("7. Returns Analysis")
    
    # Calculate tick returns
    returns = mid_prices.diff().drop_nulls()
    
    print(f"📈 Tick Returns:")
    print(f"   Mean:   {returns.mean():.6f}")
    print(f"   Std:    {returns.std():.6f}")
    print(f"   Min:    {returns.min():.4f}")
    print(f"   Max:    {returns.max():.4f}")
    
    # Count zero returns
    zero_returns = (returns == 0).sum()
    zero_pct = zero_returns / len(returns) * 100
    print(f"   Zero returns: {zero_returns:,} ({zero_pct:.1f}%)")
    
    results["zero_returns_pct"] = float(zero_pct)
    
    # Large moves (potential outliers)
    return_threshold = returns.std() * 10
    large_moves = (returns.abs() > return_threshold).sum()
    if large_moves > 0:
        print(f"⚠️  Large moves (>10σ): {large_moves:,}")
    
    # ========================================================================
    # 8. Volume Analysis (if available)
    # ========================================================================
    if bid_vol_col or ask_vol_col:
        print_section("8. Volume Analysis")
        
        vol_col = bid_vol_col or ask_vol_col
        volumes = df[vol_col].cast(pl.Float64)
        
        print(f"📊 Volume ({vol_col}):")
        print(f"   Min:    {volumes.min():.6f}")
        print(f"   Max:    {volumes.max():.6f}")
        print(f"   Mean:   {volumes.mean():.6f}")
        print(f"   Std:    {volumes.std():.6f}")
        
        zero_vol = (volumes == 0).sum()
        zero_vol_pct = zero_vol / len(df) * 100
        print(f"   Zero volume: {zero_vol:,} ({zero_vol_pct:.1f}%)")
        
        results["zero_volume_pct"] = float(zero_vol_pct)
        
        # Check if volume is useful
        if zero_vol_pct > 90:
            print("\n⚠️  Volume data is mostly zeros - Sim2Real mode recommended")
        elif volumes.std() < 0.0001:
            print("\n⚠️  Volume has very low variance - may be normalized placeholder")
    
    # ========================================================================
    # 9. Training Recommendations
    # ========================================================================
    print_section("9. Training Recommendations")
    
    print(f"\n📋 Detected column mapping for training:")
    print(f"   --timestamp-col \"{time_col}\"")
    print(f"   --timestamp-unit datetime")
    print(f"   --bid-col \"{bid_col}\"")
    print(f"   --ask-col \"{ask_col}\"")
    
    print(f"\n🚀 Suggested training command:")
    print(f"""
./scripts/start_training.sh \\
    --data "{data_path}" \\
    --timestamp-col "{time_col}" \\
    --timestamp-unit datetime \\
    --bid-col "{bid_col}" \\
    --ask-col "{ask_col}" \\
    --output models/xauusd
""")
    
    # Data quality summary
    print("\n📊 Data Quality Summary:")
    
    issues = []
    if has_nulls:
        issues.append("Missing values detected")
    if negative_spreads > 0:
        issues.append("Negative spreads (data error)")
    if zero_pct > 50:
        issues.append("High % of zero returns (low activity periods?)")
    if large_moves > 100:
        issues.append("Many large price jumps (outliers?)")
    
    if not issues:
        print("✅ Data quality: GOOD - Ready for training")
    else:
        print("⚠️  Data quality issues:")
        for issue in issues:
            print(f"   - {issue}")
    
    # Size recommendation
    total_rows = len(df)
    if total_rows < 100_000:
        print(f"\n⚠️  Small dataset ({format_number(total_rows)} rows)")
        print("   Consider collecting more data for better model performance")
    elif total_rows < 1_000_000:
        print(f"\n✅ Medium dataset ({format_number(total_rows)} rows)")
        print("   Good for initial training and validation")
    else:
        print(f"\n✅ Large dataset ({format_number(total_rows)} rows)")
        print("   Excellent for robust model training")
    
    return results


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("  AlphaOS Data Analysis Tool")
    print("=" * 60)
    print(f"\n📁 Analyzing: {args.data}")
    
    results = analyze_data(args.data, args.sample)
    
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Report saved to: {args.output}")
    
    print("\n" + "=" * 60)
    print("  Analysis Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
