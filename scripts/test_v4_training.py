#!/usr/bin/env python3
"""
v4 Training Pipeline Test Script

Tests the v4 training pipeline with a small dataset (50K ticks).

Usage:
    python scripts/test_v4_training.py [--max-ticks N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Test v4 training pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="data/XAUUSD_Ticks_50K.csv",
        help="Path to tick data CSV",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=50000,
        help="Maximum ticks to load",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/v4_test",
        help="Output directory",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="Number of CV folds (3 for faster testing)",
    )
    parser.add_argument(
        "--xgb-estimators",
        type=int,
        default=50,
        help="XGBoost n_estimators (50 for faster testing)",
    )
    args = parser.parse_args()
    
    # Setup logging
    from alphaos.core.logging import setup_logging, get_logger
    setup_logging(level="INFO", log_format="console")
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("v4 Training Pipeline Test")
    logger.info("=" * 60)
    
    # Import v4 modules
    from alphaos.v4 import (
        TrainingConfig,
        V4TrainingPipeline,
        SamplingConfig,
        SamplingMode,
        DenoiseConfig,
        DenoiseMode,
        PrimaryEngineConfig,
        MetaLabelConfig,
        TripleBarrierConfig,
    )
    
    # Create config with conservative parameters for testing
    logger.info("Creating training configuration...")
    
    sampling_config = SamplingConfig(
        mode=SamplingMode.VOLUME_BARS,
        target_volume=100,  # 100 ticks per bar
    )
    
    denoise_config = DenoiseConfig(
        mode=DenoiseMode.KALMAN,
    )
    
    primary_config = PrimaryEngineConfig(
        min_trend_duration=2,  # Lower threshold for more signals
        cooldown_bars=3,
        require_fvg=False,  # Relax FVG requirement for testing
    )

    # Tighten triple barrier for small dataset so we get both classes (0/1)
    triple_barrier = TripleBarrierConfig(
        upper_multiplier=1.0,
        lower_multiplier=1.0,
        vertical_bars=30,
        volatility_window=20,
        volatility_type="realized",
        min_barrier_pct=0.02,  # 0.02% minimum barrier (≈0.7 USD around 3,500)
        use_log_returns=True,
    )
    meta_label_cfg = MetaLabelConfig(triple_barrier_config=triple_barrier)
    
    config = TrainingConfig(
        sampling=sampling_config,
        denoise=denoise_config,
        primary=primary_config,
        meta_label=meta_label_cfg,
        cv_n_splits=args.cv_splits,
        xgb_n_estimators=args.xgb_estimators,
        xgb_max_depth=4,  # Shallower trees for small dataset
        xgb_learning_rate=0.15,
        epochs=30,
        early_stopping_patience=5,
        output_dir=args.output,
    )
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.save(output_path / "config.json")
    logger.info(f"Config saved to {output_path / 'config.json'}")
    
    # Create pipeline
    logger.info("Initializing V4TrainingPipeline...")
    pipeline = V4TrainingPipeline(config)
    
    # Load data
    logger.info(f"Loading data from {args.data}...")
    start_time = time.time()
    
    n_ticks = pipeline.load_tick_data(args.data, max_ticks=args.max_ticks)
    load_time = time.time() - start_time
    logger.info(f"Loaded {n_ticks:,} ticks in {load_time:.2f}s")
    
    # Run training
    logger.info("Starting training...")
    start_time = time.time()
    
    try:
        results = pipeline.train()
        train_time = time.time() - start_time
        
        logger.info(f"Training completed in {train_time:.2f}s")
        
        # Save schema
        pipeline.save_schema(output_path / "schema.json")
        logger.info(f"Schema saved to {output_path / 'schema.json'}")
        
        # Save results
        import json
        with open(output_path / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path / 'results.json'}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Data:           {args.data}")
        print(f"Ticks loaded:   {n_ticks:,}")
        print(f"Samples:        {results.get('n_samples', 'N/A')}")
        print(f"Features:       {results.get('n_features', 'N/A')}")
        print(f"Positive rate:  {results.get('positive_rate', 0):.2%}")
        print(f"Schema hash:    {results.get('schema_hash', 'N/A')}")
        print("-" * 60)
        
        if results.get("avg_results"):
            avg = results["avg_results"]
            print("Cross-Validation Results:")
            print(f"  Accuracy:     {avg.get('avg_accuracy', 0):.4f} ± {avg.get('std_accuracy', 0):.4f}")
            print(f"  AUC:          {avg.get('avg_auc', 0):.4f} ± {avg.get('std_auc', 0):.4f}")
            print(f"  Precision:    {avg.get('avg_precision', 0):.4f}")
            print(f"  Recall:       {avg.get('avg_recall', 0):.4f}")
        
        print("-" * 60)
        print(f"Load time:      {load_time:.2f}s")
        print(f"Train time:     {train_time:.2f}s")
        print(f"Total time:     {load_time + train_time:.2f}s")
        print(f"Output:         {output_path}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
