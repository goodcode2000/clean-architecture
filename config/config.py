"""Configuration settings for BTC Predictor."""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    BINANCE_API_URL = "https://api.binance.com/api/v3"
    COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
    
    # Data Configuration
    DATA_INTERVAL_MINUTES = 5
    HISTORICAL_DAYS = 90
    PREDICTION_HORIZON_MINUTES = 5
    
    # Model Configuration
    ENSEMBLE_WEIGHTS = {
        'kalman': 0.10,
        'random_forest': 0.30,
        'lightgbm': 0.35,
        'lstm': 0.25
    }
    
    # LSTM Configuration
    LSTM_LAYERS = [32, 16]  # Reduced from [64, 32, 16] for faster training
    LSTM_SEQUENCE_LENGTH = 30  # Reduced from 60 for faster training (2.5 hours of data)
    LSTM_BATCH_SIZE = 64  # Increased from 32 for faster training
    LSTM_EPOCHS = 10  # Reduced from 100 for faster training (~5-10 seconds)
    
    # Feature Engineering
    TECHNICAL_INDICATORS = [
        'bollinger_bands',
        'rsi',
        'macd',
        'moving_averages',
        'volatility'
    ]
    
    # Enhanced Features
    USE_ENHANCED_FEATURES = False  # Use advanced features (sentiment, microstructure, etc.)
    INCLUDE_NEWS_SENTIMENT = False  # Requires NEWS_API_KEY in .env
    
    # Feature Categories
    FEATURE_CATEGORIES = {
        'market_sentiment': True,      # Fear & Greed, sentiment indicators
        'volume_profile': True,         # VWAP, OBV, MFI, order flow
        'trend_strength': True,         # ADX, Parabolic SAR, Ichimoku
        'microstructure': True,         # Spread, liquidity, price impact
        'statistical': True,            # Skewness, kurtosis, Hurst exponent
        'news_sentiment': False         # News-based sentiment (requires API key)
    }
    
    # Training Configuration
    RETRAIN_INTERVAL_HOURS = 1
    VALIDATION_SPLIT = 0.2
    CROSS_VALIDATION_FOLDS = 5
    
    # Offset Correction
    ERROR_ANALYSIS_WINDOW_DAYS = 7
    MIN_PREDICTIONS_FOR_CORRECTION = 100
    
    # API Server
    API_HOST = "0.0.0.0"
    API_PORT = 80
    
    # File Paths
    DATA_DIR = "data"
    MODELS_DIR = "models/saved"
    LOGS_DIR = "logs"
    PREDICTIONS_FILE = "data/predictions.csv"
    
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