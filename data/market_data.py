"""
Market data collection module for advanced crypto market features.
"""
import pandas as pd
import ccxt
from typing import Dict, Optional
from config.config import Config

class MarketDataCollector:
    def __init__(self, exchange_id: str = 'binance'):
        self.exchange = ccxt.create_market_instance(exchange_id)
    
    async def fetch_ohlcv(self, symbol: str = 'BTC/USDT', timeframe: str = '1h', limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV data for given symbol."""
        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.set_index('timestamp')
    
    async def fetch_order_book(self, symbol: str = 'BTC/USDT', limit: int = 100) -> Dict:
        """Fetch order book data."""
        orderbook = await self.exchange.fetch_order_book(symbol, limit=limit)
        bids_df = pd.DataFrame(orderbook['bids'], columns=['price', 'amount'])
        asks_df = pd.DataFrame(orderbook['asks'], columns=['price', 'amount'])
        
        return {
            'timestamp': pd.Timestamp.now(),
            'bids': bids_df,
            'asks': asks_df,
            'bid_sum': bids_df['amount'].sum(),
            'ask_sum': asks_df['amount'].sum(),
            'imbalance': bids_df['amount'].sum() - asks_df['amount'].sum()
        }
    
    async def fetch_funding_rate(self, symbol: str = 'BTC/USDT') -> float:
        """Fetch current funding rate for perpetual futures."""
        try:
            funding = await self.exchange.fetch_funding_rate(symbol)
            return funding['rate']
        except:
            return 0.0

    async def fetch_open_interest(self, symbol: str = 'BTC/USDT') -> float:
        """Fetch futures open interest."""
        try:
            stats = await self.exchange.fetch_open_interest(symbol)
            return stats['openInterest']
        except:
            return 0.0