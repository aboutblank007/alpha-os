import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging
import joblib

from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Trainer")

def train_scanner():
    # Load Data
    try:
        df = pd.read_csv("labeled_data.csv")
    except:
        logger.error("labeled_data.csv not found.")
        return

    # Features & Target
    # Exclude non-feature columns
    exclude_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'label', 'log_return']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['label']
    
    # Calculate Class Weights to handle imbalance
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    logger.info(f"Class Weights: {class_weights}")
    
    # Map weights to samples
    sample_weights = y.map(class_weights)

    # Time-based Split (No Shuffle!)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    w_train = sample_weights.iloc[:train_size]
    
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Create Dataset with Weights
    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Parameters for Multi-class
    params = {
        'objective': 'multiclass',
        'num_class': 3, # 0, 1, 2
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    
    # Train
    logger.info("Training LightGBM...")
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[test_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(50)
        ]
    )
    
    # Evaluate
    preds_prob = bst.predict(X_test)
    preds = np.argmax(preds_prob, axis=1)
    
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, preds))
    
    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_test, preds))
    
    # Save Model
    bst.save_model('ai-engine/models/lgbm_scanner_v1.txt')
    logger.info("Model saved to ai-engine/models/lgbm_scanner_v1.txt")

if __name__ == "__main__":
    train_scanner()

