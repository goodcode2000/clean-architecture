"""
Technical indicators calculation module.
"""
import pandas as pd
import numpy as np
from typing import Dict
import talib

class TechnicalIndicators:
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for the given OHLCV dataframe."""
        result = df.copy()
        
        # Trend Indicators
        result['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        result['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        result['sma_200'] = talib.SMA(df['close'], timeperiod=200)
        result['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        result['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'])
        result['macd'] = macd
        result['macd_signal'] = macd_signal
        result['macd_hist'] = macd_hist
        
        # RSI
        result['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # Volatility Indicators
        result['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        result['bbands_upper'], result['bbands_middle'], result['bbands_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Volume Indicators
        result['obv'] = talib.OBV(df['close'], df['volume'])
        result['adl'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        
        # Momentum Indicators
        result['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        result['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        result['roc'] = talib.ROC(df['close'], timeperiod=10)
        
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