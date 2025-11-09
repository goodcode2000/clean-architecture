"""Configuration settings for BTC Predictor."""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    BINANCE_API_URL = "https://api.binance.com/api/v3"
    COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
    
    # Data Configuration
    SYMBOL = "TAOUSDT"  # TAO/USDT trading pair on Binance
    DATA_INTERVAL_MINUTES = 5
    HISTORICAL_DAYS = 90
    PREDICTION_HORIZON_MINUTES = 5
    
    # Model Configuration - Initial weights (will be adjusted dynamically)
    ENSEMBLE_WEIGHTS = {
        'ets': 0.15,
        'svr': 0.20,
        'kalman': 0.05,
        'random_forest': 0.30,
        'lightgbm': 0.20,  # LightGBM for rapid price changes
        'lstm': 0.10
    }
    
    # Dynamic weight adjustment settings
    ENABLE_DYNAMIC_WEIGHTS = True
    LIGHTGBM_MIN_WEIGHT = 0.2  # LightGBM minimum weight for rapid price changes
    WEIGHT_ADJUSTMENT_WINDOW = 20  # Number of recent predictions to consider
    WEIGHT_LEARNING_RATE = 0.1  # How quickly weights adapt to performance
    
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
    
    # Data Quality Settings
    MAX_NAN_PERCENTAGE = 0.3  # Maximum 30% NaN values per row
    FORWARD_FILL_LIMIT = 10  # Forward fill up to 10 periods
    USE_MEAN_FILL = True  # Fill remaining NaN with column means
    
    # Data Quality Settings
    MIN_TRAINING_SAMPLES = 1000  # Minimum samples needed for training
    MAX_NAN_PERCENTAGE = 0.3  # Maximum 30% NaN values allowed per row
    
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