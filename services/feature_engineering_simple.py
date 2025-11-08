"""Simplified feature engineering for BTC prediction - temporary fix."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class FeatureEngineer:
    """Simplified feature engineering for BTC price prediction."""
    
    def __init__(self):
        self.sequence_length = Config.LSTM_SEQUENCE_LENGTH
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create simplified features for the given price data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with basic features added
        """
        try:
            logger.info("Creating simplified features for BTC price data...")
            
            # Start with a copy of the original data
            features_df = df.copy()
            
            # Ensure timestamp is datetime
            if 'timestamp' in features_df.columns:
                features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
            
            # Basic price features
            features_df['returns'] = features_df['close'].pct_change()
            features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
            
            # Simple moving averages
            features_df['sma_5'] = features_df['close'].rolling(window=5).mean()
            features_df['sma_20'] = features_df['close'].rolling(window=20).mean()
            features_df['sma_50'] = features_df['close'].rolling(window=50).mean()
            
            # Price relative to moving averages
            features_df['price_to_sma_5'] = features_df['close'] / features_df['sma_5'] - 1
            features_df['price_to_sma_20'] = features_df['close'] / features_df['sma_20'] - 1
            
            # Simple volatility
            features_df['volatility_5'] = features_df['returns'].rolling(window=5).std()
            features_df['volatility_20'] = features_df['returns'].rolling(window=20).std()
            
            # Lag features
            features_df['close_lag_1'] = features_df['close'].shift(1)
            features_df['close_lag_2'] = features_df['close'].shift(2)
            features_df['close_lag_5'] = features_df['close'].shift(5)
            
            # Volume features
            features_df['volume_sma_5'] = features_df['volume'].rolling(window=5).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma_5']
            
            # Time features
            if 'timestamp' in features_df.columns:
                features_df['hour'] = features_df['timestamp'].dt.hour
                features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
                
                # Cyclical encoding
                features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
                features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
                features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
                features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
            
            # Remove rows with NaN values
            initial_rows = len(features_df)
            features_df = features_df.dropna()
            final_rows = len(features_df)
            
            if initial_rows != final_rows:
                logger.info(f"Removed {initial_rows - final_rows} rows with NaN values")
            
            logger.info(f"Simplified feature engineering completed: {len(features_df.columns)} features created")
            return features_df
            
        except Exception as e:
            logger.error(f"Simplified feature engineering failed: {e}")
            return df
    
    def prepare_sequences_for_lstm(self, df: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        
        Args:
            df: DataFrame with features
            target_column: Column to predict
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        try:
            # Get feature columns (exclude timestamp and target)
            feature_columns = [col for col in df.columns if col not in ['timestamp', target_column]]
            
            # Convert to numpy arrays
            features = df[feature_columns].values
            targets = df[target_column].values
            
            # Create sequences
            X_sequences = []
            y_sequences = []
            
            for i in range(self.sequence_length, len(features)):
                X_sequences.append(features[i-self.sequence_length:i])
                y_sequences.append(targets[i])
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            logger.info(f"Created {len(X_sequences)} sequences for LSTM training")
            return X_sequences, y_sequences
            
        except Exception as e:
            logger.error(f"Failed to prepare LSTM sequences: {e}")
            return np.array([]), np.array([])