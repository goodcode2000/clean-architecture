"""BTC price data collection from cryptocurrency APIs."""
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
    """Collects BTC price data from multiple cryptocurrency APIs."""
    
    def __init__(self):
        self.binance_url = Config.BINANCE_API_URL
        self.coingecko_url = Config.COINGECKO_API_URL
        self.data_dir = Config.DATA_DIR
        self.historical_days = Config.HISTORICAL_DAYS
        self.interval_minutes = Config.DATA_INTERVAL_MINUTES
        self.price_symbol = getattr(Config, 'PRICE_SYMBOL', 'BTCUSD')
        self.fallback_symbol = getattr(Config, 'BINANCE_FALLBACK_SYMBOL', 'BTCUSDT')
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
    def get_binance_data(self, symbol: str = None, limit: int = 1000, end_time_ms: int = None) -> Optional[pd.DataFrame]:
        """
        Fetch BTC price data from Binance API.
        
        Args:
            symbol: Trading pair symbol (default: BTCUSDT)
            limit: Number of data points to fetch (max 1000)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            endpoint = f"{self.binance_url}/klines"
            if symbol is None:
                symbol = self.price_symbol

            params = {
                'symbol': symbol,
                'interval': f'{self.interval_minutes}m',
                'limit': limit
            }

            # If an end_time is provided, request older klines ending at that time
            if end_time_ms is not None:
                params['endTime'] = int(end_time_ms)
            
            logger.info(f"Fetching data from Binance: {symbol} (limit={limit})")
            response = requests.get(endpoint, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Convert price columns to float
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in price_columns:
                df[col] = df[col].astype(float)
            
            # Keep only necessary columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Successfully fetched {len(df)} records from Binance")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Binance API request failed for symbol {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing Binance data: {e}")
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
        Get current BTC price from Binance API.
        
        Returns:
            Current BTC price or None if failed
        """
        try:
            endpoint = f"{self.binance_url}/ticker/price"
            # Try configured symbol first, then fallback
            symbols_to_try = [self.price_symbol, self.fallback_symbol]
            for sym in symbols_to_try:
                try:
                    params = {'symbol': sym}
                    response = requests.get(endpoint, params=params, timeout=5)
                    response.raise_for_status()
                    data = response.json()
                    current_price = float(data['price'])
                    logger.debug(f"Current BTC price from Binance ({sym}): ${current_price:,.2f}")
                    return current_price
                except Exception:
                    logger.debug(f"Binance symbol {sym} not available or failed")

            # If Binance failed for both symbols, try CoinGecko
            try:
                cg_endpoint = f"{self.coingecko_url}/simple/price"
                cg_params = {'ids': 'bitcoin', 'vs_currencies': 'usd'}
                resp = requests.get(cg_endpoint, params=cg_params, timeout=5)
                resp.raise_for_status()
                price = resp.json().get('bitcoin', {}).get('usd')
                if price is not None:
                    logger.debug(f"Current BTC price from CoinGecko: ${price:,.2f}")
                    return float(price)
            except Exception as e:
                logger.error(f"CoinGecko current price failed: {e}")

            raise RuntimeError("Failed to obtain current price from Binance and CoinGecko")
            
            logger.debug(f"Current BTC price: ${current_price:,.2f}")
            return current_price
            
        except Exception as e:
            logger.error(f"Failed to get current price: {e}")
            return None
    
    def fetch_historical_data(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch historical BTC data for the specified period.
        
        Args:
            force_refresh: If True, fetch fresh data even if cached data exists
            
        Returns:
            DataFrame with historical data or None if failed
        """
        csv_file = os.path.join(self.data_dir, 'btc_historical.csv')
        
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
        
    # Calculate how many 5-minute intervals we need for configured days
    intervals_needed = (self.historical_days * 24 * 60) // self.interval_minutes
        
        # Binance has a limit of 1000 records per request
        max_per_request = 1000
        
        all_data = []
        
        # Fetch data in chunks if needed
        if intervals_needed <= max_per_request:
            df = self.get_binance_data(limit=intervals_needed)
            if df is not None:
                all_data.append(df)
        else:
            # For more than 1000 intervals, fetch in paginated chunks using endTime
            logger.info(f"Need {intervals_needed} intervals, fetching in chunks")

            # Start from current time and go backwards
            end_time_ms = int(time.time() * 1000)  # Current time in milliseconds
            remaining = intervals_needed
            attempts = 0

            while remaining > 0 and attempts < 10000:
                chunk_size = min(max_per_request, remaining)

                df = self.get_binance_data(limit=chunk_size, end_time_ms=end_time_ms)
                attempts += 1

                if df is None or df.empty:
                    logger.warning("Binance returned no data for requested chunk - stopping pagination")
                    break

                all_data.append(df)

                # Update remaining and end_time_ms to fetch older data in next iteration
                remaining -= len(df)
                oldest_ts = int(df['timestamp'].min().timestamp() * 1000)
                # Subtract one interval to avoid overlap
                end_time_ms = oldest_ts - (self.interval_minutes * 60 * 1000)

                # Small pause to respect rate limits
                time.sleep(0.12)
        
        # If Binance failed, try CoinGecko as backup
        if not all_data:
            logger.warning("Binance data collection failed, trying CoinGecko")
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
        csv_file = os.path.join(self.data_dir, 'btc_historical.csv')
        
        try:
            # Load existing data
            if os.path.exists(csv_file):
                existing_df = pd.read_csv(csv_file)
                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                latest_time = existing_df['timestamp'].max()
            else:
                # No existing data, fetch historical
                return self.fetch_historical_data()
            
            # Get new data since last update
            new_df = self.get_binance_data(limit=100)  # Get last 100 records
            
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