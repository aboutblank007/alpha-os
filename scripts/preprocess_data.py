#!/usr/bin/env python3
"""
AlphaOS Data Preprocessing Script

Prepares raw tick data for training:
- Removes outliers
- Filters by time range
- Samples data if too large
- Saves in optimized Parquet format

Usage:
    python scripts/preprocess_data.py --input raw_ticks.csv --output processed_ticks.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess tick data for AlphaOS training",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV or Parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output Parquet file",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N rows (None = use all)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10_000_000,
        help="Maximum rows to keep (default: 10M)",
    )
    parser.add_argument(
        "--remove-outliers",
        action="store_true",
        default=True,
        help="Remove price outliers",
    )
    parser.add_argument(
        "--outlier-sigma",
        type=float,
        default=10.0,
        help="Outlier threshold in standard deviations",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="Time (EET)",
        help="Timestamp column name",
    )
    parser.add_argument(
        "--bid-col",
        type=str,
        default="Bid",
        help="Bid column name",
    )
    parser.add_argument(
        "--ask-col",
        type=str,
        default="Ask",
        help="Ask column name",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Filter to last N days from latest timestamp (None = use all data)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  AlphaOS Data Preprocessing")
    print("=" * 60)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # ========================================================================
    # 1. Load Data (Lazy for large files)
    # ========================================================================
    print(f"\n📂 Loading: {input_path}")
    
    # Use lazy evaluation for large files
    if input_path.suffix == ".csv":
        df_lazy = pl.scan_csv(input_path)
    else:
        df_lazy = pl.scan_parquet(input_path)
    
    # Get total row count (approximate for CSV)
    try:
        original_rows = df_lazy.select(pl.len()).collect().item()
    except:
        # Fallback: load sample to estimate
        print("   Estimating row count...")
        sample_df = df_lazy.head(1000).collect()
        original_rows = len(sample_df)
    
    print(f"   Total rows (estimated): {original_rows:,}")
    
    # Sample if requested
    if args.sample:
        print(f"   Sampling {args.sample:,} rows...")
        df_lazy = df_lazy.head(args.sample)
    
    # ========================================================================
    # 2. Parse Timestamps
    # ========================================================================
    print("\n📅 Parsing timestamps...")
    
    time_col = args.time_col
    if time_col in df_lazy.columns:
        # Parse timestamps using lazy evaluation
        try:
            df_lazy = df_lazy.with_columns(
                pl.col(time_col).str.to_datetime("%Y.%m.%d %H:%M:%S%.f").alias("timestamp")
            )
        except:
            try:
                df_lazy = df_lazy.with_columns(
                    pl.col(time_col).str.to_datetime().alias("timestamp")
                )
            except Exception as e:
                print(f"❌ Could not parse timestamps: {e}")
                sys.exit(1)
        
        # Convert to microseconds
        df_lazy = df_lazy.with_columns(
            (pl.col("timestamp").dt.epoch("us")).alias("timestamp_us")
        )
        print("✅ Timestamps parsed")
    
    # ========================================================================
    # 2.5. Filter by Date Range (if --days specified)
    # ========================================================================
    if args.days is not None:
        print(f"\n📅 Filtering to last {args.days} days from latest timestamp...")
        
        # Find latest timestamp (use sample for efficiency)
        print("   Finding latest timestamp...")
        latest_sample = df_lazy.select(pl.col("timestamp_us").max()).collect()
        if latest_sample is not None and len(latest_sample) > 0:
            latest_ts_us = latest_sample.item()
            if latest_ts_us is not None:
                # Calculate cutoff time (latest - N days)
                cutoff_ts_us = latest_ts_us - (args.days * 24 * 60 * 60 * 1_000_000)
                
                # Convert to datetime for display
                from datetime import datetime
                latest_dt = datetime.fromtimestamp(latest_ts_us / 1_000_000)
                cutoff_dt = datetime.fromtimestamp(cutoff_ts_us / 1_000_000)
                
                print(f"   Latest timestamp: {latest_dt}")
                print(f"   Cutoff timestamp: {cutoff_dt}")
                print(f"   Filtering data from {cutoff_dt} to {latest_dt}...")
                
                # Filter data
                df_lazy = df_lazy.filter(pl.col("timestamp_us") >= cutoff_ts_us)
            else:
                print("   ⚠️  Could not determine latest timestamp, skipping date filter")
        else:
            print("   ⚠️  Could not determine latest timestamp, skipping date filter")
    
    # ========================================================================
    # 2.6. Collect Lazy DataFrame
    # ========================================================================
    print("\n🔄 Collecting data (this may take a while for large files)...")
    df = df_lazy.collect()
    original_rows = len(df)
    print(f"✅ Loaded {original_rows:,} rows")
    
    # ========================================================================
    # 3. Rename Price Columns
    # ========================================================================
    print("\n💰 Standardizing column names...")
    
    df = df.with_columns([
        pl.col(args.bid_col).cast(pl.Float64).alias("bid"),
        pl.col(args.ask_col).cast(pl.Float64).alias("ask"),
    ])
    
    # Calculate mid price and spread
    df = df.with_columns([
        ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
        (pl.col("ask") - pl.col("bid")).alias("spread"),
    ])
    
    # ========================================================================
    # 4. Remove Outliers
    # ========================================================================
    if args.remove_outliers:
        print(f"\n🔍 Removing outliers (>{args.outlier_sigma}σ)...")
        
        # Calculate returns
        df = df.with_columns(
            pl.col("mid").diff().alias("returns")
        )
        
        # Calculate return statistics
        returns_mean = df["returns"].drop_nulls().mean()
        returns_std = df["returns"].drop_nulls().std()
        
        threshold = args.outlier_sigma * returns_std
        
        # Filter outliers
        df_clean = df.filter(
            (pl.col("returns").is_null()) |  # Keep first row
            (pl.col("returns").abs() <= threshold)
        )
        
        removed = original_rows - len(df_clean)
        print(f"   Removed {removed:,} outliers ({removed/original_rows*100:.4f}%)")
        
        df = df_clean
    
    # ========================================================================
    # 5. Remove Invalid Data
    # ========================================================================
    print("\n🧹 Removing invalid data...")
    
    before = len(df)
    
    # Remove negative spreads
    df = df.filter(pl.col("spread") > 0)
    
    # Remove zero prices
    df = df.filter((pl.col("bid") > 0) & (pl.col("ask") > 0))
    
    after = len(df)
    if before > after:
        print(f"   Removed {before - after:,} invalid rows")
    else:
        print("   No invalid data found")
    
    # ========================================================================
    # 6. Limit Data Size
    # ========================================================================
    if len(df) > args.max_rows:
        print(f"\n✂️  Limiting to {args.max_rows:,} rows...")
        
        # Take the most recent data
        df = df.sort("timestamp_us").tail(args.max_rows)
        print(f"   Kept most recent {len(df):,} rows")
    
    # ========================================================================
    # 7. Select Final Columns
    # ========================================================================
    print("\n📋 Selecting final columns...")
    
    # Keep essential columns for training
    output_cols = ["timestamp_us", "bid", "ask", "mid", "spread"]
    
    # Add volume if available
    for vol_col in ["BidVolume", "AskVolume", "bidvolume", "askvolume"]:
        if vol_col in df.columns:
            df = df.with_columns(pl.col(vol_col).cast(pl.Float64).alias(vol_col.lower()))
            if vol_col.lower() not in output_cols:
                output_cols.append(vol_col.lower())
    
    df_final = df.select(output_cols)
    
    # ========================================================================
    # 8. Data Statistics
    # ========================================================================
    print("\n📊 Final Data Statistics:")
    print(f"   Rows: {len(df_final):,}")
    print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Price range: {df_final['bid'].min():.2f} - {df_final['ask'].max():.2f}")
    print(f"   Avg spread: {df_final['spread'].mean():.4f}")
    
    # ========================================================================
    # 9. Save Output
    # ========================================================================
    print(f"\n💾 Saving to: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.write_parquet(output_path)
    
    # Show file size
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved {len(df_final):,} rows ({file_size:.1f} MB)")
    
    # ========================================================================
    # 10. Training Command
    # ========================================================================
    print("\n" + "=" * 60)
    print("  Ready for Training!")
    print("=" * 60)
    print(f"""
🚀 Run the following command to start training:

./scripts/start_training.sh \\
    --data "{output_path}" \\
    --timestamp-col "timestamp_us" \\
    --timestamp-unit us \\
    --bid-col "bid" \\
    --ask-col "ask" \\
    --output models/xauusd
""")


if __name__ == "__main__":
    main()
