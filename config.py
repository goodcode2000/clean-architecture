"""
Configuration file for BTC prediction app
"""
import os

# Data configuration
DATA_DIR = "data"
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions_history.csv")
MODEL_DIR = "models"
HISTORY_DAYS = 90
INTERVAL_MINUTES = 5

# API configuration
EXCHANGE = "binance"
SYMBOL = "BTC/USDT"

# Model configuration
ENSEMBLE_WEIGHTS = {
    'ets': 0.15,
    'garch': 0.15,
    'lightgbm': 0.30,
    'tcn': 0.20,
    'cnn': 0.20
}

# Training configuration
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42
RETRAIN_HOURS = 24
ROLLING_WINDOW_DAYS = 7

# Prediction offset target
TARGET_OFFSET_RANGE = (20, 100)  # USD

# Feature engineering
TECHNICAL_INDICATORS = {
    'rsi_period': 14,
    'bb_period': 20,
    'bb_std': 2,
    'ema_periods': [12, 26, 50],
    'macd_periods': [12, 26, 9]
}

