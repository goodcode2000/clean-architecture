"""
High-frequency data processor for 1-minute resolution data and micro-features
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .data_models import PriceData, MarketData
from .data_collector import BTCDataCollector
from .config import Config

logger = logging.getLogger(__name__)

@dataclass
class HighFrequencyData:
    """High-frequency data point"""
    timestamp: datetime
    price: float
    volume: float
    bid_price: float
    ask_price: float
    bid_volume: float
    ask_volume: float
    trade_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'volume': self.volume,
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'bid_volume': self.bid_volume,
            'ask_volume': self.ask_volume,
            'trade_count': self.trade_count
        }

class HighFrequencyCollector:
    """Collects high-frequency BTC data at 1-minute intervals"""
    
    def __init__(self, config: Config):
        self.config = config
        self.btc_collector = BTCDataCollector(config)
        self.last_collection_time = None
        
    async def collect_1min_data(self) -> Optional[HighFrequencyData]:
        """Collect 1-minute resolution data"""
        try:
            # Get current price data
            price_data = await self.btc_collector.fetch_current_price()
            if not price_data:
                return None
            
            # Get order book data
            order_book = await self.btc_collector.fetch_order_book()
            
            # Create high-frequency data point
            hf_data = HighFrequencyData(
                timestamp=datetime.now(),
                price=float(price_data['close_price']),
                volume=float(price_data['volume']),
                bid_price=order_book.get('bids', [[0, 0]])[0][0] if order_book.get('bids') else 0,
                ask_price=order_book.get('asks', [[0, 0]])[0][0] if order_book.get('asks') else 0,
                bid_volume=order_book.get('bid_depth', 0),
                ask_volume=order_book.get('ask_depth', 0),
                trade_count=1  # Placeholder - would need trade data from API
            )
            
            self.last_collection_time = datetime.now()
            return hf_data
            
        except Exception as e:
            logger.error(f"High-frequency data collection failed: {e}")
            return None
    
    async def collect_historical_1min(self, hours: int = 24) -> List[HighFrequencyData]:
        """Collect historical 1-minute data"""
        try:
            # This would typically use a different API endpoint for historical minute data
            # For now, we'll simulate by collecting current data points
            historical_data = []
            
            # In a real implementation, you would fetch from Binance klines with 1m interval
            # For demonstration, we'll create synthetic data based on current price
            current_data = await self.collect_1min_data()
            if not current_data:
                return []
            
            # Generate synthetic historical data (in production, use real API)
            base_price = current_data.price
            for i in range(hours * 60):  # 60 minutes per hour
                timestamp = datetime.now() - timedelta(minutes=i)
                
                # Add some random variation (this would be real data in production)
                price_variation = np.random.normal(0, base_price * 0.001)  # 0.1% std dev
                
                hf_point = HighFrequencyData(
                    timestamp=timestamp,
                    price=base_price + price_variation,
                    volume=current_data.volume * (0.8 + 0.4 * np.random.random()),
                    bid_price=base_price + price_variation - 1,
                    ask_price=base_price + price_variation + 1,
                    bid_volume=current_data.bid_volume * (0.8 + 0.4 * np.random.random()),
                    ask_volume=current_data.ask_volume * (0.8 + 0.4 * np.random.random()),
                    trade_count=np.random.randint(1, 10)
                )
                
                historical_data.append(hf_point)
            
            historical_data.reverse()  # Chronological order
            logger.info(f"Collected {len(historical_data)} high-frequency data points")
            return historical_data
            
        except Exception as e:
            logger.error(f"Historical high-frequency data collection failed: {e}")
            return []

class MicroFeatureExtractor:
    """Extracts micro-features from high-frequency data"""
    
    @staticmethod
    def calculate_vwap(hf_data: List[HighFrequencyData], window: int = 20) -> np.ndarray:
        """Volume Weighted Average Price"""
        if len(hf_data) < window:
            window = len(hf_data)
        
        vwap = np.zeros(len(hf_data))
        
        for i in range(len(hf_data)):
            start_idx = max(0, i - window + 1)
            window_data = hf_data[start_idx:i+1]
            
            total_volume = sum(d.volume for d in window_data)
            if total_volume > 0:
                vwap[i] = sum(d.price * d.volume for d in window_data) / total_volume
            else:
                vwap[i] = hf_data[i].price
        
        return vwap
    
    @staticmethod
    def calculate_spread_features(hf_data: List[HighFrequencyData]) -> Dict[str, np.ndarray]:
        """Calculate bid-ask spread features"""
        spreads = []
        spread_pct = []
        mid_prices = []
        
        for data in hf_data:
            if data.bid_price > 0 and data.ask_price > 0:
                spread = data.ask_price - data.bid_price
                mid_price = (data.bid_price + data.ask_price) / 2
                spread_percentage = spread / mid_price if mid_price > 0 else 0
            else:
                spread = 0
                spread_percentage = 0
                mid_price = data.price
            
            spreads.append(spread)
            spread_pct.append(spread_percentage)
            mid_prices.append(mid_price)
        
        return {
            'spread': np.array(spreads),
            'spread_pct': np.array(spread_pct),
            'mid_price': np.array(mid_prices)
        }
    
    @staticmethod
    def calculate_order_flow_imbalance(hf_data: List[HighFrequencyData]) -> np.ndarray:
        """Calculate order flow imbalance"""
        imbalance = []
        
        for data in hf_data:
            total_volume = data.bid_volume + data.ask_volume
            if total_volume > 0:
                ofi = (data.bid_volume - data.ask_volume) / total_volume
            else:
                ofi = 0
            imbalance.append(ofi)
        
        return np.array(imbalance)
    
    @staticmethod
    def calculate_price_impact(hf_data: List[HighFrequencyData], window: int = 10) -> np.ndarray:
        """Calculate price impact measures"""
        if len(hf_data) < 2:
            return np.zeros(len(hf_data))
        
        prices = np.array([d.price for d in hf_data])
        volumes = np.array([d.volume for d in hf_data])
        
        price_changes = np.diff(prices)
        price_changes = np.concatenate([[0], price_changes])
        
        impact = np.zeros(len(hf_data))
        
        for i in range(1, len(hf_data)):
            if volumes[i] > 0:
                impact[i] = abs(price_changes[i]) / (volumes[i] + 1e-10)
            else:
                impact[i] = 0
        
        # Smooth with rolling average
        impact_smooth = pd.Series(impact).rolling(window=window, min_periods=1).mean().values
        
        return impact_smooth
    
    @staticmethod
    def calculate_trade_intensity(hf_data: List[HighFrequencyData], window: int = 15) -> np.ndarray:
        """Calculate trade intensity features"""
        trade_counts = np.array([d.trade_count for d in hf_data])
        
        # Rolling sum of trade counts
        intensity = pd.Series(trade_counts).rolling(window=window, min_periods=1).sum().values
        
        return intensity
    
    @staticmethod
    def calculate_volatility_clustering(hf_data: List[HighFrequencyData], window: int = 30) -> np.ndarray:
        """Calculate volatility clustering measure"""
        if len(hf_data) < 2:
            return np.zeros(len(hf_data))
        
        prices = np.array([d.price for d in hf_data])
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        returns = np.concatenate([[0], returns])
        
        # Rolling volatility
        vol = pd.Series(returns).rolling(window=window, min_periods=1).std().values
        
        # Volatility clustering (correlation of volatility with lagged volatility)
        clustering = np.zeros(len(vol))
        for i in range(1, len(vol)):
            if i >= window:
                recent_vol = vol[i-window:i]
                if len(recent_vol) > 1:
                    clustering[i] = np.corrcoef(recent_vol[:-1], recent_vol[1:])[0, 1]
        
        return clustering

class HighFrequencyProcessor:
    """Main high-frequency data processor"""
    
    def __init__(self, config: Config):
        self.config = config
        self.collector = HighFrequencyCollector(config)
        self.feature_extractor = MicroFeatureExtractor()
        self.data_buffer: List[HighFrequencyData] = []
        self.max_buffer_size = 1440  # 24 hours of 1-minute data
        
    async def start_collection(self):
        """Start continuous 1-minute data collection"""
        logger.info("Starting high-frequency data collection...")
        
        while True:
            try:
                # Collect 1-minute data
                hf_data = await self.collector.collect_1min_data()
                
                if hf_data:
                    # Add to buffer
                    self.data_buffer.append(hf_data)
                    
                    # Maintain buffer size
                    if len(self.data_buffer) > self.max_buffer_size:
                        self.data_buffer.pop(0)
                    
                    logger.debug(f"Collected HF data: ${hf_data.price:,.2f}, Buffer size: {len(self.data_buffer)}")
                
                # Wait for next minute
                await asyncio.sleep(60)  # 1 minute
                
            except Exception as e:
                logger.error(f"High-frequency collection error: {e}")
                await asyncio.sleep(10)  # Wait 10 seconds before retry
    
    def get_recent_data(self, minutes: int = 60) -> List[HighFrequencyData]:
        """Get recent high-frequency data"""
        if minutes >= len(self.data_buffer):
            return self.data_buffer.copy()
        else:
            return self.data_buffer[-minutes:].copy()
    
    def extract_micro_features(self, minutes: int = 60) -> Dict[str, Any]:
        """Extract micro-features from recent data"""
        recent_data = self.get_recent_data(minutes)
        
        if not recent_data:
            return {}
        
        try:
            features = {}
            
            # VWAP features
            vwap = self.feature_extractor.calculate_vwap(recent_data)
            features['vwap'] = vwap[-1] if len(vwap) > 0 else 0
            features['price_vwap_ratio'] = recent_data[-1].price / (vwap[-1] + 1e-10) if len(vwap) > 0 else 1
            
            # Spread features
            spread_features = self.feature_extractor.calculate_spread_features(recent_data)
            features['spread'] = spread_features['spread'][-1]
            features['spread_pct'] = spread_features['spread_pct'][-1]
            features['mid_price'] = spread_features['mid_price'][-1]
            
            # Order flow imbalance
            ofi = self.feature_extractor.calculate_order_flow_imbalance(recent_data)
            features['order_flow_imbalance'] = ofi[-1]
            features['ofi_mean'] = np.mean(ofi[-10:]) if len(ofi) >= 10 else ofi[-1]
            
            # Price impact
            impact = self.feature_extractor.calculate_price_impact(recent_data)
            features['price_impact'] = impact[-1]
            features['impact_mean'] = np.mean(impact[-10:]) if len(impact) >= 10 else impact[-1]
            
            # Trade intensity
            intensity = self.feature_extractor.calculate_trade_intensity(recent_data)
            features['trade_intensity'] = intensity[-1]
            
            # Volatility clustering
            clustering = self.feature_extractor.calculate_volatility_clustering(recent_data)
            features['volatility_clustering'] = clustering[-1]
            
            # Additional micro-features
            prices = np.array([d.price for d in recent_data])
            volumes = np.array([d.volume for d in recent_data])
            
            # Price momentum at different scales
            if len(prices) >= 5:
                features['momentum_1min'] = (prices[-1] - prices[-2]) / (prices[-2] + 1e-10)
                features['momentum_5min'] = (prices[-1] - prices[-5]) / (prices[-5] + 1e-10)
            
            if len(prices) >= 15:
                features['momentum_15min'] = (prices[-1] - prices[-15]) / (prices[-15] + 1e-10)
            
            # Volume features
            features['volume_current'] = recent_data[-1].volume
            features['volume_mean'] = np.mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1]
            features['volume_ratio'] = recent_data[-1].volume / (np.mean(volumes[-10:]) + 1e-10) if len(volumes) >= 10 else 1
            
            # Bid-ask features
            features['bid_ask_ratio'] = recent_data[-1].bid_volume / (recent_data[-1].ask_volume + 1e-10)
            
            logger.debug(f"Extracted {len(features)} micro-features")
            return features
            
        except Exception as e:
            logger.error(f"Micro-feature extraction failed: {e}")
            return {}
    
    def resample_to_5min(self) -> List[PriceData]:
        """Resample 1-minute data to 5-minute intervals with exact 300s windows"""
        if len(self.data_buffer) < 5:
            return []
        
        try:
            # Group data into 5-minute windows
            resampled_data = []
            
            # Start from the most recent complete 5-minute window
            latest_time = self.data_buffer[-1].timestamp
            window_start = latest_time.replace(second=0, microsecond=0)
            window_start = window_start.replace(minute=(window_start.minute // 5) * 5)
            
            current_window_start = window_start
            
            # Process windows going backwards
            for _ in range(min(288, len(self.data_buffer) // 5)):  # Max 24 hours of 5-min data
                window_end = current_window_start + timedelta(minutes=5)
                
                # Find data points in this window
                window_data = [
                    d for d in self.data_buffer 
                    if current_window_start <= d.timestamp < window_end
                ]
                
                if window_data:
                    # Create OHLCV data for this window
                    prices = [d.price for d in window_data]
                    volumes = [d.volume for d in window_data]
                    
                    price_data = PriceData(
                        timestamp=current_window_start,
                        open_price=prices[0],
                        high_price=max(prices),
                        low_price=min(prices),
                        close_price=prices[-1],
                        volume=sum(volumes)
                    )
                    
                    resampled_data.append(price_data)
                
                # Move to previous window
                current_window_start -= timedelta(minutes=5)
            
            # Reverse to get chronological order
            resampled_data.reverse()
            
            logger.debug(f"Resampled to {len(resampled_data)} 5-minute intervals")
            return resampled_data
            
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            return []
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about the data buffer"""
        if not self.data_buffer:
            return {'buffer_empty': True}
        
        prices = [d.price for d in self.data_buffer]
        volumes = [d.volume for d in self.data_buffer]
        
        return {
            'buffer_size': len(self.data_buffer),
            'time_range': {
                'start': self.data_buffer[0].timestamp.isoformat(),
                'end': self.data_buffer[-1].timestamp.isoformat()
            },
            'price_range': {
                'min': min(prices),
                'max': max(prices),
                'current': prices[-1]
            },
            'volume_stats': {
                'mean': np.mean(volumes),
                'std': np.std(volumes),
                'current': volumes[-1]
            }
        }