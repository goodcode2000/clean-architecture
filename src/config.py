"""
Configuration settings for the BTC Prediction Engine
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Config:
    """Configuration class for the prediction engine"""
    
    # API Configuration
    BINANCE_API_URL: str = "https://api.binance.com/api/v3"
    COINGECKO_API_URL: str = "https://api.coingecko.com/api/v3"
    
    # Data Collection Settings
    DATA_COLLECTION_INTERVAL: int = 300  # 5 minutes in seconds
    HISTORICAL_DAYS: int = 90  # 90-day rolling window
    
    # File Paths
    DATA_DIR: str = "data"
    LOGS_DIR: str = "logs"
    MODELS_DIR: str = "models"
    
    # CSV File Names
    PRICE_DATA_FILE: str = "btc_price_data.csv"
    PREDICTIONS_FILE: str = "predictions.csv"
    PERFORMANCE_FILE: str = "model_performance.csv"
    
    # Model Configuration
    PREDICTION_HORIZON: int = 5  # 5 minutes ahead
    RETRAIN_INTERVAL: int = 24 * 60 * 60  # 24 hours in seconds
    ROLLING_WINDOW_DAYS: int = 7  # 7-day rolling window for retraining
    
    # API Server Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # GPU Configuration
    USE_GPU: bool = True
    GPU_MEMORY_LIMIT: int = 16  # 16GB
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        
    @property
    def price_data_path(self) -> str:
        return os.path.join(self.DATA_DIR, self.PRICE_DATA_FILE)
        
    @property
    def predictions_path(self) -> str:
        return os.path.join(self.DATA_DIR, self.PREDICTIONS_FILE)
        
    @property
    def performance_path(self) -> str:
        return os.path.join(self.DATA_DIR, self.PERFORMANCE_FILE)
        
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration dictionary"""
        return {
            "binance_url": self.BINANCE_API_URL,
            "coingecko_url": self.COINGECKO_API_URL,
            "timeout": 30,
            "retries": 3
        }