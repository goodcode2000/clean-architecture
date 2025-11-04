"""
Market data collection module for advanced crypto market features.
"""
import pandas as pd
import requests
from typing import Dict, Optional
from config.config import Config

class MarketDataCollector:
    def __init__(self, exchange_id: str = 'binance'):
        self.base_url = "https://api.binance.com/api/v3"
    
    def fetch_ohlcv(self, symbol: str = 'BTCUSDT', timeframe: str = '1h', limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV data for given symbol."""
        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': limit
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
                return df.set_index('timestamp')
            else:
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    
    def fetch_order_book(self, symbol: str = 'BTCUSDT', limit: int = 100) -> Dict:
        """Fetch order book data."""
        try:
            url = f"{self.base_url}/depth"
            params = {'symbol': symbol, 'limit': limit}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                bids_df = pd.DataFrame(data['bids'], columns=['price', 'amount']).astype(float)
                asks_df = pd.DataFrame(data['asks'], columns=['price', 'amount']).astype(float)
                
                return {
                    'timestamp': pd.Timestamp.now(),
                    'bids': bids_df,
                    'asks': asks_df,
                    'bid_sum': bids_df['amount'].sum(),
                    'ask_sum': asks_df['amount'].sum(),
                    'imbalance': bids_df['amount'].sum() - asks_df['amount'].sum()
                }
            else:
                return self._empty_orderbook()
        except Exception:
            return self._empty_orderbook()
    
    def fetch_funding_rate(self, symbol: str = 'BTCUSDT') -> float:
        """Fetch current funding rate for perpetual futures."""
        try:
            # Binance futures API endpoint
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            params = {'symbol': symbol}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return float(data.get('lastFundingRate', 0.0))
            else:
                return 0.0
        except Exception:
            return 0.0

    def fetch_open_interest(self, symbol: str = 'BTCUSDT') -> float:
        """Fetch futures open interest."""
        try:
            url = "https://fapi.binance.com/fapi/v1/openInterest"
            params = {'symbol': symbol}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return float(data.get('openInterest', 0.0))
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _empty_orderbook(self) -> Dict:
        """Return empty orderbook structure."""
        return {
            'timestamp': pd.Timestamp.now(),
            'bids': pd.DataFrame(columns=['price', 'amount']),
            'asks': pd.DataFrame(columns=['price', 'amount']),
            'bid_sum': 0.0,
            'ask_sum': 0.0,
            'imbalance': 0.0
        }