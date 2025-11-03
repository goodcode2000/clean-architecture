"""
BTC data fetcher module
Fetches historical and real-time BTC price data
"""
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from config import EXCHANGE, SYMBOL, HISTORY_DAYS, INTERVAL_MINUTES, DATA_DIR


class BTCDataFetcher:
    def __init__(self):
        """Initialize exchange connection"""
        exchange_class = getattr(ccxt, EXCHANGE)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'timeout': 30000,
        })
        
    def get_historical_data(self, days=90):
        """
        Fetch historical BTC data for specified days
        
        Args:
            days: Number of days of historical data to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching {days} days of historical BTC data...")
        
        # Calculate timeframe
        timeframe = f'{INTERVAL_MINUTES}m'
        limit = days * 24 * (60 // INTERVAL_MINUTES)  # Total 5-min intervals
        
        try:
            # Fetch historical data
            ohlcv = self.exchange.fetch_ohlcv(
                SYMBOL,
                timeframe=timeframe,
                limit=min(limit, 1000)  # Exchange limit
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('datetime')
            
            # If we need more than 1000 candles, fetch in batches
            if limit > 1000:
                since = df.index[0] - timedelta(days=days)
                all_data = []
                
                for _ in range(days // 10 + 1):
                    ohlcv_batch = self.exchange.fetch_ohlcv(
                        SYMBOL,
                        timeframe=timeframe,
                        since=int(since.timestamp() * 1000),
                        limit=1000
                    )
                    if not ohlcv_batch:
                        break
                    
                    batch_df = pd.DataFrame(ohlcv_batch, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    batch_df['datetime'] = pd.to_datetime(batch_df['timestamp'], unit='ms')
                    batch_df = batch_df.set_index('datetime')
                    
                    all_data.append(batch_df)
                    since = batch_df.index[-1]
                    time.sleep(1)  # Rate limiting
                
                if all_data:
                    df = pd.concat(all_data)
                    df = df.sort_index()
                    df = df.drop_duplicates()
            
            # Keep only the last 'days' days
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df.index >= cutoff_date]
            
            print(f"Fetched {len(df)} data points")
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            raise
    
    def get_current_price(self):
        """
        Get current BTC price
        
        Returns:
            float: Current BTC price
        """
        try:
            ticker = self.exchange.fetch_ticker(SYMBOL)
            return ticker['last']
        except Exception as e:
            print(f"Error fetching current price: {e}")
            raise
    
    def get_latest_data_point(self):
        """
        Get the latest data point (last closed candle)
        
        Returns:
            Series with latest OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(SYMBOL, timeframe=f'{INTERVAL_MINUTES}m', limit=1)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('datetime')
            return df.iloc[-1]
        except Exception as e:
            print(f"Error fetching latest data point: {e}")
            raise


if __name__ == "__main__":
    fetcher = BTCDataFetcher()
    data = fetcher.get_historical_data(90)
    print(data.head())
    print(f"\nCurrent price: ${fetcher.get_current_price():.2f}")

