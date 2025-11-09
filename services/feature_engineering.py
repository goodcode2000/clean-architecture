"""Feature engineering module for BTC price prediction."""
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
    """Creates features for BTC price prediction models."""
    
    def __init__(self):
        self.technical_indicators = Config.TECHNICAL_INDICATORS
        self.sequence_length = Config.LSTM_SEQUENCE_LENGTH
        
    def calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: DataFrame with price data
            window: Moving average window
            num_std: Number of standard deviations
            
        Returns:
            DataFrame with Bollinger Bands columns added
        """
        try:
            df = df.copy()
            
            # Calculate moving average and standard deviation
            df['bb_middle'] = df['close'].rolling(window=window).mean()
            bb_std = df['close'].rolling(window=window).std()
            
            # Calculate upper and lower bands
            df['bb_upper'] = df['bb_middle'] + (bb_std * num_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std * num_std)
            
            # Calculate Bollinger Band width and position
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Bollinger Band squeeze indicator
            df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(window=20).mean()
            
            logger.debug("Bollinger Bands calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate Bollinger Bands: {e}")
            return df
    
    def calculate_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            df: DataFrame with price data
            window: RSI calculation window
            
        Returns:
            DataFrame with RSI column added
        """
        try:
            df = df.copy()
            
            # Calculate price changes
            delta = df['close'].diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=window).mean()
            avg_losses = losses.rolling(window=window).mean()
            
            # Calculate RSI
            rs = avg_gains / avg_losses
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # RSI-based signals
            df['rsi_overbought'] = df['rsi'] > 70
            df['rsi_oversold'] = df['rsi'] < 30
            df['rsi_momentum'] = df['rsi'].diff()
            
            logger.debug("RSI calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate RSI: {e}")
            return df
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            df: DataFrame with price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            DataFrame with MACD columns added
        """
        try:
            df = df.copy()
            
            # Calculate EMAs
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            
            # Calculate MACD line
            df['macd'] = ema_fast - ema_slow
            
            # Calculate signal line
            df['macd_signal'] = df['macd'].ewm(span=signal).mean()
            
            # Calculate MACD histogram
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # MACD crossover signals
            df['macd_bullish'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
            df['macd_bearish'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
            
            logger.debug("MACD calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate MACD: {e}")
            return df
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various moving averages.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with moving average columns added
        """
        try:
            df = df.copy()
            
            # Simple Moving Averages
            periods = [5, 10, 20, 50, 100, 200]
            for period in periods:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                
                # Price relative to moving average
                df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
            
            # Exponential Moving Averages
            ema_periods = [12, 26, 50]
            for period in ema_periods:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1
            
            # Moving average crossovers
            df['sma_5_above_20'] = df['sma_5'] > df['sma_20']
            df['sma_20_above_50'] = df['sma_20'] > df['sma_50']
            df['golden_cross'] = (df['sma_50'] > df['sma_200']) & (df['sma_50'].shift(1) <= df['sma_200'].shift(1))
            df['death_cross'] = (df['sma_50'] < df['sma_200']) & (df['sma_50'].shift(1) >= df['sma_200'].shift(1))
            
            logger.debug("Moving averages calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate moving averages: {e}")
            return df
    
    def calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility-based indicators.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with volatility columns added
        """
        try:
            df = df.copy()
            
            # Price returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Rolling volatility (standard deviation of returns)
            windows = [5, 10, 20, 50]
            for window in windows:
                df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
                df[f'log_volatility_{window}'] = df['log_returns'].rolling(window=window).std()
            
            # Average True Range (ATR)
            df['high_low'] = df['high'] - df['low']
            df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
            df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
            
            df['true_range'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
            df['atr'] = df['true_range'].rolling(window=14).mean()
            
            # Volatility regime detection
            df['high_volatility'] = df['volatility_20'] > df['volatility_20'].rolling(window=50).quantile(0.75)
            df['low_volatility'] = df['volatility_20'] < df['volatility_20'].rolling(window=50).quantile(0.25)
            
            # Price range indicators
            df['price_range'] = (df['high'] - df['low']) / df['close']
            df['body_size'] = abs(df['close'] - df['open']) / df['close']
            df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
            df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
            
            logger.debug("Volatility indicators calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate volatility indicators: {e}")
            return df
    
    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """
        Create lagged features for time series analysis.
        
        Args:
            df: DataFrame with price data
            lags: List of lag periods to create
            
        Returns:
            DataFrame with lag features added
        """
        try:
            if lags is None:
                lags = [1, 2, 3, 5, 10, 20]
            
            df = df.copy()
            
            # Price lags
            for lag in lags:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            
            # Rolling statistics
            windows = [5, 10, 20]
            for window in windows:
                df[f'close_mean_{window}'] = df['close'].rolling(window=window).mean()
                df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
                df[f'close_min_{window}'] = df['close'].rolling(window=window).min()
                df[f'close_max_{window}'] = df['close'].rolling(window=window).max()
                
                # Position within rolling window
                df[f'close_position_{window}'] = (df['close'] - df[f'close_min_{window}']) / (df[f'close_max_{window}'] - df[f'close_min_{window}'])
            
            # Time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            logger.debug("Lag features created successfully")
            return df
            
        except Exception as e:
            logger.error(f"Failed to create lag features: {e}")
            return df
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Detect and flag outliers in the data.
        
        Args:
            df: DataFrame with features
            columns: Columns to check for outliers
            
        Returns:
            DataFrame with outlier flags added
        """
        try:
            if columns is None:
                columns = ['close', 'volume', 'returns']
            
            df = df.copy()
            
            for col in columns:
                if col in df.columns:
                    # Z-score method
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    df[f'{col}_outlier_zscore'] = z_scores > 3
                    
                    # IQR method
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[f'{col}_outlier_iqr'] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            logger.debug("Outlier detection completed")
            return df
            
        except Exception as e:
            logger.error(f"Failed to detect outliers: {e}")
            return df
    
    def normalize_features(self, df: pd.DataFrame, method: str = 'minmax') -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize features for model training.
        
        Args:
            df: DataFrame with features
            method: Normalization method ('minmax', 'zscore', 'robust')
            
        Returns:
            Tuple of (normalized_df, normalization_params)
        """
        try:
            df_norm = df.copy()
            normalization_params = {}
            
            # Get numeric columns (exclude timestamp and boolean columns)
            numeric_columns = df_norm.select_dtypes(include=[np.number]).columns
            exclude_columns = ['timestamp'] + [col for col in numeric_columns if df_norm[col].dtype == bool]
            numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
            
            for col in numeric_columns:
                if method == 'minmax':
                    min_val = df_norm[col].min()
                    max_val = df_norm[col].max()
                    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
                    normalization_params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
                    
                elif method == 'zscore':
                    mean_val = df_norm[col].mean()
                    std_val = df_norm[col].std()
                    df_norm[col] = (df_norm[col] - mean_val) / std_val
                    normalization_params[col] = {'method': 'zscore', 'mean': mean_val, 'std': std_val}
                    
                elif method == 'robust':
                    median_val = df_norm[col].median()
                    mad_val = (df_norm[col] - median_val).abs().median()
                    df_norm[col] = (df_norm[col] - median_val) / mad_val
                    normalization_params[col] = {'method': 'robust', 'median': median_val, 'mad': mad_val}
            
            logger.debug(f"Features normalized using {method} method")
            return df_norm, normalization_params
            
        except Exception as e:
            logger.error(f"Failed to normalize features: {e}")
            return df, {}
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features for the given price data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all features added
        """
        try:
            logger.info("Creating all features for TAO price data...")
            
            # Start with a copy of the original data
            features_df = df.copy()
            
            # Ensure timestamp is datetime
            if 'timestamp' in features_df.columns:
                features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
            
            # Calculate technical indicators
            if 'bollinger_bands' in self.technical_indicators:
                features_df = self.calculate_bollinger_bands(features_df)
            
            if 'rsi' in self.technical_indicators:
                features_df = self.calculate_rsi(features_df)
            
            if 'macd' in self.technical_indicators:
                features_df = self.calculate_macd(features_df)
            
            if 'moving_averages' in self.technical_indicators:
                features_df = self.calculate_moving_averages(features_df)
            
            if 'volatility' in self.technical_indicators:
                features_df = self.calculate_volatility_indicators(features_df)
            
            # Create lag features
            features_df = self.create_lag_features(features_df)
            
            # Detect outliers
            features_df = self.detect_outliers(features_df)
            
            # Remove rows with NaN values (from rolling calculations)
            # Only remove rows where critical columns have NaN
            initial_rows = len(features_df)
            
            # First, forward fill to handle minor gaps
            features_df = features_df.ffill(limit=5)
            
            # Then remove rows with too many NaN values (>50% of columns)
            threshold = len(features_df.columns) * 0.5
            features_df = features_df.dropna(thresh=int(threshold))
            
            final_rows = len(features_df)
            
            if initial_rows != final_rows:
                logger.info(f"Removed {initial_rows - final_rows} rows with NaN values")
            
            if len(features_df) == 0:
                logger.error(f"Feature engineering failed: All {initial_rows} rows removed by dropna()")
                return pd.DataFrame()  # Return empty DataFrame
            
            logger.info(f"Feature engineering completed: {len(features_df)} rows, {len(features_df.columns)} features created")
            return features_df
            
        except Exception as e:
            logger.error(f"Failed to create features: {e}")
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
            logger.info(f"Sequence shape: {X_sequences.shape}")
            
            return X_sequences, y_sequences
            
        except Exception as e:
            logger.error(f"Failed to prepare LSTM sequences: {e}")
            return np.array([]), np.array([])
    
    def get_feature_importance_columns(self) -> List[str]:
        """
        Get list of important feature columns for model training.
        
        Returns:
            List of feature column names
        """
        important_features = [
            # Price features
            'close', 'open', 'high', 'low', 'volume',
            
            # Technical indicators
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'bb_width',
            
            # Moving averages
            'sma_5', 'sma_20', 'sma_50', 'price_to_sma_20',
            'ema_12', 'ema_26', 'price_to_ema_12',
            
            # Volatility
            'volatility_20', 'atr', 'returns',
            
            # Lag features
            'close_lag_1', 'close_lag_2', 'close_lag_5',
            'returns_lag_1', 'returns_lag_2',
            
            # Rolling statistics
            'close_mean_5', 'close_std_5', 'close_position_20',
            
            # Time features
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        return important_features