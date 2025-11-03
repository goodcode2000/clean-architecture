"""
Feature engineering module for BTC prediction
Creates technical indicators, volatility features, and trend classification
"""
import pandas as pd
import numpy as np
import ta
from config import TECHNICAL_INDICATORS


class FeatureEngineer:
    def __init__(self):
        """Initialize feature engineering"""
        self.rsi_period = TECHNICAL_INDICATORS['rsi_period']
        self.bb_period = TECHNICAL_INDICATORS['bb_period']
        self.bb_std = TECHNICAL_INDICATORS['bb_std']
        self.ema_periods = TECHNICAL_INDICATORS['ema_periods']
        self.macd_periods = TECHNICAL_INDICATORS['macd_periods']
    
    def create_features(self, df):
        """
        Create all features from OHLCV data
        
        Args:
            df: DataFrame with OHLCV data (open, high, low, close, volume)
            
        Returns:
            DataFrame with features added
        """
        data = df.copy()
        
        # Price features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['price_change'] = data['close'].diff()
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        
        # Volatility features
        data['volatility'] = data['returns'].rolling(window=20).std()
        data['realized_volatility'] = data['returns'].abs().rolling(window=20).mean()
        
        # RSI
        data['rsi'] = ta.momentum.RSIIndicator(data['close'], window=self.rsi_period).rsi()
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(
            data['close'],
            window=self.bb_period,
            window_dev=self.bb_std
        )
        data['bb_upper'] = bb_indicator.bollinger_hband()
        data['bb_lower'] = bb_indicator.bollinger_lband()
        data['bb_middle'] = bb_indicator.bollinger_mavg()
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # EMA features
        for period in self.ema_periods:
            data[f'ema_{period}'] = ta.trend.EMAIndicator(data['close'], window=period).ema_indicator()
            data[f'price_ema_{period}_ratio'] = data['close'] / data[f'ema_{period}']
        
        # MACD
        macd = ta.trend.MACD(
            data['close'],
            window_slow=self.macd_periods[1],
            window_fast=self.macd_periods[0],
            window_sign=self.macd_periods[2]
        )
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_diff'] = macd.macd_diff()
        
        # Volume features
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        data['volume_price_trend'] = (data['volume'] * data['returns']).rolling(window=20).sum()
        
        # Trend classification
        data['trend_up'] = (data['close'] > data['close'].shift(5)).astype(int)
        data['trend_down'] = (data['close'] < data['close'].shift(5)).astype(int)
        data['trend_neutral'] = ((data['trend_up'] == 0) & (data['trend_down'] == 0)).astype(int)
        
        # Price acceleration
        data['price_acceleration'] = data['returns'].diff()
        
        # High-frequency micro-features (for offset reduction)
        data['price_spread'] = data['high'] - data['low']
        data['spread_ratio'] = data['price_spread'] / data['close']
        data['price_velocity'] = data['close'].diff()
        data['velocity_change'] = data['price_velocity'].diff()
        
        # Time-based features
        data['hour'] = data.index.hour
        data['minute'] = data.index.minute
        data['day_of_week'] = data.index.dayofweek
        
        # Remove NaN rows
        data = data.dropna()
        
        return data
    
    def get_feature_columns(self):
        """Get list of feature column names"""
        features = [
            'returns', 'log_returns', 'price_change', 'high_low_ratio', 'close_open_ratio',
            'volatility', 'realized_volatility', 'rsi',
            'bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position',
            'macd', 'macd_signal', 'macd_diff',
            'volume_ratio', 'volume_price_trend',
            'trend_up', 'trend_down', 'trend_neutral',
            'price_acceleration', 'price_spread', 'spread_ratio',
            'price_velocity', 'velocity_change',
            'hour', 'minute', 'day_of_week'
        ]
        
        # Add EMA features
        for period in self.ema_periods:
            features.extend([f'ema_{period}', f'price_ema_{period}_ratio'])
        
        return features


if __name__ == "__main__":
    import pandas as pd
    from data_fetcher import BTCDataFetcher
    
    fetcher = BTCDataFetcher()
    data = fetcher.get_historical_data(10)
    
    fe = FeatureEngineer()
    features = fe.create_features(data)
    print(f"Created {len(fe.get_feature_columns())} features")
    print(features[fe.get_feature_columns()].head())

