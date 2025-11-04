"""
Technical indicators calculation module.
"""
import pandas as pd
import numpy as np
from typing import Dict
# import talib  # Replaced with ta library
import ta

class TechnicalIndicators:
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for the given OHLCV dataframe."""
        result = df.copy()
        
        # Trend Indicators
        result['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        result['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        result['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
        result['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        result['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # MACD
        result['macd'] = ta.trend.macd(df['close'])
        result['macd_signal'] = ta.trend.macd_signal(df['close'])
        result['macd_hist'] = ta.trend.macd_diff(df['close'])
        
        # RSI
        result['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # Volatility Indicators
        result['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        result['bbands_upper'] = bb_indicator.bollinger_hband()
        result['bbands_middle'] = bb_indicator.bollinger_mavg()
        result['bbands_lower'] = bb_indicator.bollinger_lband()
        
        # Volume Indicators
        result['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        result['adl'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
        
        # Momentum Indicators
        result['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
        result['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=14)
        result['roc'] = ta.momentum.roc(df['close'], window=10)
        
        # Custom Indicators
        result['price_volatility'] = df['close'].pct_change().rolling(window=30).std()
        result['volume_volatility'] = df['volume'].pct_change().rolling(window=30).std()
        
        # Price Channels
        result['highest_high'] = df['high'].rolling(window=20).max()
        result['lowest_low'] = df['low'].rolling(window=20).min()
        
        return result

    @staticmethod
    def calculate_volatility_metrics(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate advanced volatility metrics."""
        metrics = {}
        
        # Garman-Klass Volatility
        log_hl = (df['high'] / df['low']).apply(np.log)
        log_co = (df['close'] / df['open']).apply(np.log)
        metrics['garman_klass_vol'] = np.sqrt(
            0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2
        )
        
        # Parkinson Volatility
        metrics['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * log_hl**2
        )
        
        # Yang-Zhang Volatility
        metrics['yang_zhang_vol'] = pd.Series(
            TechnicalIndicators._yang_zhang_volatility(
                df['open'], df['high'], df['low'], df['close']
            ),
            index=df.index
        )
        
        return metrics

    @staticmethod
    def _yang_zhang_volatility(open_price: pd.Series, high: pd.Series, 
                             low: pd.Series, close: pd.Series, 
                             window: int = 30) -> np.ndarray:
        """Calculate Yang-Zhang volatility."""
        rs = close / open_price
        log_rs = rs.apply(np.log)
        
        vo = log_rs.rolling(window=window).var()  # Overnight volatility
        vc = ((high / low).apply(np.log)).rolling(window=window).var()  # Close-to-close volatility
        
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        vyz = vo + k * vc
        
        return np.sqrt(252 * vyz)  # Annualized