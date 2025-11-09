"""Configuration settings for BTC Predictor."""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    COINBASE_API_URL = "https://api.exchange.coinbase.com"
    COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
    
    # Data Configuration
    SYMBOL = "TAO-USD"  # TAO/USD trading pair
    DATA_INTERVAL_MINUTES = 5
    HISTORICAL_DAYS = 90
    PREDICTION_HORIZON_MINUTES = 5
    
    # Model Configuration
    ENSEMBLE_WEIGHTS = {
        'ets': 0.10,
        'svr': 0.15,
        'kalman': 0.05,
        'random_forest': 0.20,
        'lightgbm': 0.20,
        'xgboost': 0.20,  # XGBoost for rapid price changes
        'lstm': 0.10
    }
    
    # XGBoost minimum weight for rapid price changes
    XGBOOST_MIN_WEIGHT = 0.2
    
    # LSTM Configuration
    LSTM_LAYERS = [64, 32, 16]
    LSTM_SEQUENCE_LENGTH = 60  # 5 hours of 5-minute data
    LSTM_BATCH_SIZE = 32
    LSTM_EPOCHS = 100
    
    # Feature Engineering
    TECHNICAL_INDICATORS = [
        'bollinger_bands',
        'rsi',
        'macd',
        'moving_averages',
        'volatility'
    ]
    
    # Training Configuration
    RETRAIN_INTERVAL_HOURS = 6
    VALIDATION_SPLIT = 0.2
    CROSS_VALIDATION_FOLDS = 5
    
    # Offset Correction
    ERROR_ANALYSIS_WINDOW_DAYS = 7
    MIN_PREDICTIONS_FOR_CORRECTION = 100
    
    # API Server
    API_HOST = "0.0.0.0"
    API_PORT = 3001
    
    # File Paths
    DATA_DIR = "data"
    MODELS_DIR = "models/saved"
    LOGS_DIR = "logs"
    PREDICTIONS_FILE = "data/predictions.csv"
    HISTORICAL_DATA_FILE = "data/tao_historical.csv"
    
    # GPU Configuration
    USE_GPU = True
    GPU_MEMORY_LIMIT = 8192  # MB
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            "models/ensemble",
            "models/individual"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)