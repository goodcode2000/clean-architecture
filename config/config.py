"""Configuration settings for BTC Predictor with advanced market data analysis."""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    BINANCE_API_URL = "https://api.binance.com/api/v3"
    COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
    
    # API Keys (set from environment variables)
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', '')
    TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET', '')
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '')
    REDDIT_USER_AGENT = 'BTCPredictor/1.0'
    
    # Data Configuration
    DATA_INTERVAL_MINUTES = 5
    HISTORICAL_DAYS = 18 * 30  # 18 months of data
    
    # Trading Pairs Configuration
    PRICE_SYMBOL = os.getenv('PRICE_SYMBOL', 'BTCUSDT')
    BINANCE_FALLBACK_SYMBOL = os.getenv('BINANCE_FALLBACK_SYMBOL', 'BTCUSDT')
    CORRELATED_PAIRS = ['ETHUSDT', 'BNBUSDT']  # Correlated assets
    
    # Prediction Configuration
    PREDICTION_HORIZON_MINUTES = 5
    SEQUENCE_LENGTH = 48  # Hours of data for sequence models
    
    # Model Configuration
    # Replace ETS with a volatility-aware GARCH model; add TFT as experimental (weight 0 by default)
    ENSEMBLE_WEIGHTS = {
        'garch': 0.15,
        'svr': 0.20,
        'kalman': 0.05,
        'random_forest': 0.25,
        'lightgbm': 0.25,
        'lstm': 0.10,
        'tft': 0.0
    }
    
    # LSTM Configuration
    LSTM_LAYERS = [64, 32, 16]
    LSTM_SEQUENCE_LENGTH = 60  # 5 hours of 5-minute data
    LSTM_BATCH_SIZE = 32
    LSTM_EPOCHS = 100
    
    # Feature Engineering
    TECHNICAL_INDICATORS = [
        # Trend Indicators
        'sma_20', 'sma_50', 'sma_200',
        'ema_12', 'ema_26',
        'macd', 'macd_signal', 'macd_hist',
        # Momentum Indicators
        'rsi', 'mfi', 'cci', 'roc',
        # Volatility Indicators
        'atr', 'bbands_upper', 'bbands_middle', 'bbands_lower',
        'garman_klass_vol', 'parkinson_vol', 'yang_zhang_vol',
        # Volume Indicators
        'obv', 'adl'
    ]
    
    # Feature Groups
    FEATURE_GROUPS = {
        'price': ['open', 'high', 'low', 'close', 'volume'],
        'technical': TECHNICAL_INDICATORS,
        'market_depth': ['bid_sum', 'ask_sum', 'imbalance'],
        'derivatives': ['funding_rate', 'open_interest'],
        'sentiment': ['sentiment_score', 'sentiment_magnitude']
    }
    
    # Market Depth Configuration
    MARKET_DEPTH_LEVELS = 100  # Number of order book levels
    
    # Sentiment Analysis Weights
    SENTIMENT_WEIGHTS = {
        'news': 0.4,
        'twitter': 0.3,
        'reddit': 0.3
    }
    
    # Training Configuration
    RETRAIN_INTERVAL_HOURS = 6
    VALIDATION_SPLIT = 0.2
    CROSS_VALIDATION_FOLDS = 5
    
    # Offset Correction
    ERROR_ANALYSIS_WINDOW_DAYS = 7
    MIN_PREDICTIONS_FOR_CORRECTION = 100
    
    # API Server
    API_HOST = "0.0.0.0"
    API_PORT = 5000
    
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