"""TAO price data collection from Coinbase API."""
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from loguru import logger
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class BTCDataCollector:
    """Collects TAO price data from Coinbase API."""
    
    def __init__(self):
        self.coinbase_url = Config.COINBASE_API_URL
        self.symbol = Config.SYMBOL
        self.data_dir = Config.DATA_DIR
        self.historical_days = Config.HISTORICAL_DAYS
        self.interval_minutes = Config.DATA_INTERVAL_MINUTES
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
    def get_coinbase_data(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Fetch TAO price data from Coinbase API.
        
        Args:
            start_time: Start time for data fetch
            end_time: End time for data fetch
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Coinbase candles endpoint: /products/{product_id}/candles
            endpoint = f"{self.coinbase_url}/products/{self.symbol}/candles"
            
            # Coinbase granularity in seconds (5 minutes = 300 seconds)
            granularity = self.interval_minutes * 60
            
            params = {
                'granularity': granularity
            }
            
            if start_time:
                params['start'] = start_time.isoformat()
            if end_time:
                params['end'] = end_time.isoformat()
            
            logger.info(f"Fetching data from Coinbase: {self.symbol}")
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                logger.warning("No data received from Coinbase")
                return None
            
            # Coinbase returns: [time, low, high, open, close, volume]
            df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Convert price columns to float
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in price_columns:
                df[col] = df[col].astype(float)
            
            # Reorder columns to match expected format
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Successfully fetched {len(df)} records from Coinbase")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Coinbase API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing Coinbase data: {e}")
            return None
    
    def get_coingecko_data(self, days: int = 90) -> Optional[pd.DataFrame]:
        """
        Fetch BTC price data from CoinGecko API as backup.
        
        Args:
            days: Number of days of historical data
            
        Returns:
            DataFrame with price data or None if failed
        """
        try:
            endpoint = f"{self.coingecko_url}/coins/bitcoin/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'minute' if days <= 1 else 'hourly'
            }
            
            logger.info("Fetching data from CoinGecko as backup")
            response = requests.get(endpoint, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract price data
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            if not prices:
                logger.warning("No price data received from CoinGecko")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add volume data if available
            if volumes:
                volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
                df = df.merge(volume_df, on='timestamp', how='left')
            else:
                df['volume'] = 0
            
            # For CoinGecko, we only have close prices, so we'll use them for OHLC
            df['open'] = df['close']
            df['high'] = df['close']
            df['low'] = df['close']
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Successfully fetched {len(df)} records from CoinGecko")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing CoinGecko data: {e}")
            return None
    
    def get_current_price(self) -> Optional[float]:
        """
        Get current TAO price from Coinbase API.
        
        Returns:
            Current TAO price or None if failed
        """
        try:
            endpoint = f"{self.coinbase_url}/products/{self.symbol}/ticker"
            
            response = requests.get(endpoint, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            current_price = float(data['price'])
            
            logger.debug(f"Current TAO price: ${current_price:,.2f}")
            return current_price
            
        except Exception as e:
            logger.error(f"Failed to get current price: {e}")
            return None
    
    def fetch_historical_data(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch historical TAO data for the specified period.
        
        Args:
            force_refresh: If True, fetch fresh data even if cached data exists
            
        Returns:
            DataFrame with historical data or None if failed
        """
        csv_file = os.path.join(self.data_dir, 'tao_historical.csv')
        
        # Check if we have recent cached data
        if not force_refresh and os.path.exists(csv_file):
            try:
                existing_df = pd.read_csv(csv_file)
                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                
                # Check if data is recent (within last hour)
                latest_time = existing_df['timestamp'].max()
                if datetime.now() - latest_time < timedelta(hours=1):
                    logger.info("Using cached historical data")
                    return existing_df
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")
        
        # Coinbase API can return up to 300 candles per request
        # For 90 days at 5-minute intervals, we need multiple requests
        max_candles_per_request = 300
        intervals_needed = (self.historical_days * 24 * 60) // self.interval_minutes
        
        all_data = []
        
        # Calculate number of chunks needed
        num_chunks = (intervals_needed + max_candles_per_request - 1) // max_candles_per_request
        
        logger.info(f"Need {intervals_needed} intervals, fetching in {num_chunks} chunks from Coinbase")
        
        # Fetch each chunk going backwards in time
        end_time = datetime.now()
        
        for i in range(num_chunks):
            # Calculate start time for this chunk
            chunk_duration_minutes = max_candles_per_request * self.interval_minutes
            start_time = end_time - timedelta(minutes=chunk_duration_minutes)
            
            try:
                logger.info(f"Fetching chunk {i+1}/{num_chunks}: {start_time} to {end_time}")
                df = self.get_coinbase_data(start_time=start_time, end_time=end_time)
                
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    logger.info(f"Successfully fetched chunk {i+1}/{num_chunks}: {len(df)} records")
                else:
                    logger.warning(f"No data received for chunk {i+1}/{num_chunks}")
                
            except Exception as e:
                logger.error(f"Failed to fetch chunk {i+1}: {e}")
            
            # Move end_time back for next chunk
            end_time = start_time
            
            # Rate limiting - wait between requests
            time.sleep(0.5)
        
        # If Coinbase failed, try CoinGecko as backup
        if not all_data:
            logger.warning("Coinbase data collection failed, trying CoinGecko")
            df = self.get_coingecko_data(days=self.historical_days)
            if df is not None:
                all_data.append(df)
        
        if not all_data:
            logger.error("Failed to collect data from all sources")
            return None
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        combined_df = combined_df.reset_index(drop=True)
        
        # Keep only the last 90 days
        cutoff_time = datetime.now() - timedelta(days=self.historical_days)
        combined_df = combined_df[combined_df['timestamp'] >= cutoff_time]
        
        logger.info(f"Collected {len(combined_df)} historical records")
        
        # Save to CSV
        try:
            combined_df.to_csv(csv_file, index=False)
            logger.info(f"Historical data saved to {csv_file}")
        except Exception as e:
            logger.error(f"Failed to save historical data: {e}")
        
        return combined_df
    
    def update_data(self) -> Optional[pd.DataFrame]:
        """
        Update existing data with latest records.
        
        Returns:
            Updated DataFrame or None if failed
        """
        csv_file = os.path.join(self.data_dir, 'tao_historical.csv')
        
        try:
            # Load existing data
            if os.path.exists(csv_file):
                existing_df = pd.read_csv(csv_file)
                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                latest_time = existing_df['timestamp'].max()
            else:
                # No existing data, fetch historical
                return self.fetch_historical_data()
            
            # Get new data since last update (last 24 hours)
            end_time = datetime.now()
            start_time = latest_time
            
            new_df = self.get_coinbase_data(start_time=start_time, end_time=end_time)
            
            if new_df is None:
                logger.warning("Failed to get new data, using existing")
                return existing_df
            
            # Filter for records newer than our latest
            new_records = new_df[new_df['timestamp'] > latest_time]
            
            if len(new_records) == 0:
                logger.info("No new records to add")
                return existing_df
            
            # Combine and deduplicate
            updated_df = pd.concat([existing_df, new_records], ignore_index=True)
            updated_df = updated_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Keep only last 90 days
            cutoff_time = datetime.now() - timedelta(days=self.historical_days)
            updated_df = updated_df[updated_df['timestamp'] >= cutoff_time]
            updated_df = updated_df.reset_index(drop=True)
            
            # Save updated data
            updated_df.to_csv(csv_file, index=False)
            
            logger.info(f"Added {len(new_records)} new records, total: {len(updated_df)}")
            return updated_df
            
        except Exception as e:
            logger.error(f"Failed to update data: {e}")
            return None