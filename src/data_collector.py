"""
BTC Data Collector with API integration for Binance and CoinGecko
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from .data_models import PriceData, MarketData
from .config import Config

logger = logging.getLogger(__name__)

class BTCDataCollector:
    """Collects BTC price data from multiple sources with retry mechanisms"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=10)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    async def _ensure_session(self):
        """Ensure session is available"""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=aiohttp.TCPConnector(limit=10)
            )
    
    async def fetch_current_price(self) -> Optional[Dict[str, Any]]:
        """Fetch current BTC price from Binance API"""
        await self._ensure_session()
        
        try:
            # Try Binance first
            binance_data = await self._fetch_binance_data()
            if binance_data:
                return binance_data
                
            # Fallback to CoinGecko
            logger.warning("Binance API failed, trying CoinGecko...")
            coingecko_data = await self._fetch_coingecko_data()
            return coingecko_data
            
        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
            return None
    
    async def _fetch_binance_data(self) -> Optional[Dict[str, Any]]:
        """Fetch data from Binance API"""
        try:
            # Get 24hr ticker statistics
            ticker_url = f"{self.config.BINANCE_API_URL}/ticker/24hr?symbol=BTCUSDT"
            
            async with self.session.get(ticker_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert Binance format to our format
                    return {
                        'timestamp': datetime.now(),
                        'open_price': float(data['openPrice']),
                        'high_price': float(data['highPrice']),
                        'low_price': float(data['lowPrice']),
                        'close_price': float(data['lastPrice']),
                        'volume': float(data['volume']),
                        'source': 'binance'
                    }
                else:
                    logger.error(f"Binance API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Binance API fetch error: {e}")
            return None
    
    async def _fetch_coingecko_data(self) -> Optional[Dict[str, Any]]:
        """Fetch data from CoinGecko API as fallback"""
        try:
            # Get current price and 24h data
            url = f"{self.config.COINGECKO_API_URL}/simple/price"
            params = {
                'ids': 'bitcoin',
                'vs_currencies': 'usd',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    btc_data = data.get('bitcoin', {})
                    
                    current_price = btc_data.get('usd', 0)
                    volume = btc_data.get('usd_24h_vol', 0)
                    
                    # CoinGecko doesn't provide OHLC in simple price endpoint
                    # Use current price as approximation
                    return {
                        'timestamp': datetime.now(),
                        'open_price': current_price,  # Approximation
                        'high_price': current_price,  # Approximation
                        'low_price': current_price,   # Approximation
                        'close_price': current_price,
                        'volume': volume,
                        'source': 'coingecko'
                    }
                else:
                    logger.error(f"CoinGecko API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"CoinGecko API fetch error: {e}")
            return None
    
    async def fetch_historical_data(self, days: int = 90) -> List[PriceData]:
        """Fetch historical BTC data for the specified number of days"""
        await self._ensure_session()
        
        try:
            # Use Binance klines for historical data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            url = f"{self.config.BINANCE_API_URL}/klines"
            params = {
                'symbol': 'BTCUSDT',
                'interval': '5m',  # 5-minute intervals
                'startTime': int(start_time.timestamp() * 1000),
                'endTime': int(end_time.timestamp() * 1000),
                'limit': 1000  # Maximum allowed by Binance
            }
            
            historical_data = []
            
            # Fetch data in chunks if needed
            current_start = start_time
            while current_start < end_time:
                params['startTime'] = int(current_start.timestamp() * 1000)
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        klines = await response.json()
                        
                        for kline in klines:
                            price_data = PriceData(
                                timestamp=datetime.fromtimestamp(kline[0] / 1000),
                                open_price=float(kline[1]),
                                high_price=float(kline[2]),
                                low_price=float(kline[3]),
                                close_price=float(kline[4]),
                                volume=float(kline[5])
                            )
                            historical_data.append(price_data)
                        
                        # Update start time for next chunk
                        if klines:
                            last_timestamp = klines[-1][0] / 1000
                            current_start = datetime.fromtimestamp(last_timestamp) + timedelta(minutes=5)
                        else:
                            break
                    else:
                        logger.error(f"Historical data fetch error: {response.status}")
                        break
                        
                # Rate limiting
                await asyncio.sleep(0.1)
            
            logger.info(f"Fetched {len(historical_data)} historical data points")
            return historical_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []
    
    async def fetch_order_book(self) -> Dict[str, float]:
        """Fetch order book depth for micro-features"""
        await self._ensure_session()
        
        try:
            url = f"{self.config.BINANCE_API_URL}/depth"
            params = {'symbol': 'BTCUSDT', 'limit': 100}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Calculate order book metrics
                    bids = [[float(price), float(qty)] for price, qty in data['bids'][:10]]
                    asks = [[float(price), float(qty)] for price, qty in data['asks'][:10]]
                    
                    bid_depth = sum(qty for _, qty in bids)
                    ask_depth = sum(qty for _, qty in asks)
                    
                    return {
                        'bid_depth': bid_depth,
                        'ask_depth': ask_depth,
                        'depth_ratio': bid_depth / ask_depth if ask_depth > 0 else 0,
                        'spread': asks[0][0] - bids[0][0] if bids and asks else 0
                    }
                    
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            
        return {}
    
    async def validate_and_clean_data(self, price_data: Dict[str, Any]) -> Optional[PriceData]:
        """Validate and clean price data"""
        try:
            # Basic validation
            required_fields = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
            for field in required_fields:
                if field not in price_data or price_data[field] <= 0:
                    logger.warning(f"Invalid data: {field} is missing or invalid")
                    return None
            
            # Price consistency checks
            if not (price_data['low_price'] <= price_data['close_price'] <= price_data['high_price']):
                logger.warning("Price data inconsistency detected")
                return None
            
            # Create PriceData object
            return PriceData(
                timestamp=price_data.get('timestamp', datetime.now()),
                open_price=price_data['open_price'],
                high_price=price_data['high_price'],
                low_price=price_data['low_price'],
                close_price=price_data['close_price'],
                volume=price_data['volume']
            )
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return None