import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger("OnlineLGBM")

class OnlineLGBM:
    """
    Online Learning Wrapper for LightGBM.
    Supports Symbol-Specific Models (Multi-Booster).
    Optimized for incremental updates on M2 Pro.
    """
    def __init__(self, model_dir='ai-engine/models', default_model_path=None, params=None):
        self.model_dir = model_dir
        self.default_model_path = default_model_path
        self.boosters = {} # {symbol: booster}
        self.params = params if params else {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1
        }
        
    def get_booster(self, symbol):
        """Lazy load model for specific symbol, with fallback."""
        if not symbol:
            return None
            
        if symbol in self.boosters:
            return self.boosters[symbol]
            
        # 1. Try Specific Model
        model_path = os.path.join(self.model_dir, f"lgbm_{symbol}.txt")
        if os.path.exists(model_path):
            try:
                bst = lgb.Booster(model_file=model_path)
                self.boosters[symbol] = bst
                logger.info(f"✅ Loaded LightGBM model for {symbol} from {os.path.basename(model_path)}")
                return bst
            except Exception as e:
                logger.error(f"❌ Failed to load model for {symbol} from {model_path}: {e}")
        
        # 2. Try Default Model (Fallback)
        if self.default_model_path and os.path.exists(self.default_model_path):
            try:
                # Cache default model under 'DEFAULT' key if not loaded? 
                # Or just load it as the symbol's booster?
                # Better: Shared booster for memory efficiency? 
                # For now, load copy to allow divergent training
                bst = lgb.Booster(model_file=self.default_model_path)
                self.boosters[symbol] = bst
                logger.info(f"⚠️ Loaded Default Model for {symbol} (Fallback)")
                return bst
            except Exception as e:
                logger.error(f"❌ Failed to load default model: {e}")
                
        return None

    def predict(self, features_df, symbol):
        bst = self.get_booster(symbol)
        if not bst:
            # Cold Start: Return 0.0 until we have a model
            return np.zeros(len(features_df))
        
        feature_names = bst.feature_name()
        
        # Ensure column order matches model
        if feature_names:
            for col in feature_names:
                if col not in features_df.columns:
                    features_df[col] = 0.0
            X = features_df[feature_names]
        else:
            X = features_df
            
        return bst.predict(X)

    def update(self, X_new, y_new, symbol, save_checkpoint=True):
        """
        Incremental Update for specific symbol.
        """
        if len(X_new) == 0:
            return

        train_data = lgb.Dataset(X_new, label=y_new)
        bst = self.get_booster(symbol)
        
        try:
            # Continue training existing booster (or start new if bst is None but params provided)
            # Check lgb.train documentation: init_model can be None
            
            new_bst = lgb.train(
                self.params,
                train_data,
                num_boost_round=10, 
                init_model=bst, # Pass existing booster if any
                keep_training_booster=True
            )
            
            # Update reference
            self.boosters[symbol] = new_bst
            
            if save_checkpoint:
                model_path = os.path.join(self.model_dir, f"lgbm_{symbol}.txt")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                new_bst.save_model(model_path)
                logger.info(f"🔄 Online Model Updated for {symbol}")
                
        except Exception as e:
            logger.error(f"Update failed for {symbol}: {e}")
