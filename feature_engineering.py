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
        # Ensure strictly 5-minute aligned index to avoid drift
        if not data.index.inferred_type == 'datetime64':
            data.index = pd.to_datetime(data.index)
        data = data[~data.index.duplicated(keep='last')]
        data = data.asfreq('5T', method='pad')
        
        # Kalman-smoothed price (simple 1D filter)
        def kalman_1d(series, process_var=1e-3, meas_var=1e-1):
            x_est = []
            x = series.iloc[0]
            P = 1.0
            Q = process_var
            R = meas_var
            for z in series:
                # Predict
                x = x
                P = P + Q
                # Update
                K = P / (P + R)
                x = x + K * (z - x)
                P = (1 - K) * P
                x_est.append(x)
            return pd.Series(x_est, index=series.index)

        data['kalman_close'] = kalman_1d(data['close'])

        # Price features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['price_change'] = data['close'].diff()
        data['kalman_returns'] = data['kalman_close'].pct_change()
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        
        # Volatility features (adaptive short + medium windows)
        data['volatility'] = data['returns'].rolling(window=20).std()
        data['volatility_short'] = data['returns'].rolling(window=10).std()
        data['realized_volatility'] = data['returns'].abs().rolling(window=20).mean()
        data['realized_volatility_short'] = data['returns'].abs().rolling(window=10).mean()
        
        # RSI (standard + short)
        data['rsi'] = ta.momentum.RSIIndicator(data['close'], window=self.rsi_period).rsi()
        data['rsi_short'] = ta.momentum.RSIIndicator(data['close'], window=max(7, self.rsi_period//2)).rsi()
        
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
        # Very short EMA for micro-trend
        data['ema_5'] = ta.trend.EMAIndicator(data['close'], window=5).ema_indicator()
        data['price_ema_5_ratio'] = data['close'] / data['ema_5']
        
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
        data['volume_ma_short'] = data['volume'].rolling(window=10).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        data['volume_ratio_short'] = data['volume'] / data['volume_ma_short']
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
            'returns', 'log_returns', 'price_change', 'kalman_returns', 'high_low_ratio', 'close_open_ratio',
            'volatility', 'volatility_short', 'realized_volatility', 'realized_volatility_short', 'rsi', 'rsi_short',
            'bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position',
            'macd', 'macd_signal', 'macd_diff',
            'volume_ratio', 'volume_ratio_short', 'volume_price_trend',
            'trend_up', 'trend_down', 'trend_neutral',
            'price_acceleration', 'price_spread', 'spread_ratio',
            'price_velocity', 'velocity_change',
            'hour', 'minute', 'day_of_week'
        ]
        
        # Add EMA features
        for period in self.ema_periods:
            features.extend([f'ema_{period}', f'price_ema_{period}_ratio'])
        features.extend(['ema_5', 'price_ema_5_ratio', 'kalman_close'])
        
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

