"""
Feature Engineering System for BTC Price Prediction
Implements technical indicators, micro-features, and Kalman filtering
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
from pykalman import KalmanFilter
from .data_models import PriceData, MarketData

logger = logging.getLogger(__name__)

@dataclass
class FeatureSet:
    """Container for engineered features"""
    timestamp: datetime
    price_features: Dict[str, float]
    technical_indicators: Dict[str, float]
    micro_features: Dict[str, float]
    volatility_features: Dict[str, float]
    trend_features: Dict[str, float]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert all features to a flat dictionary"""
        features = {}
        features.update(self.price_features)
        features.update(self.technical_indicators)
        features.update(self.micro_features)
        features.update(self.volatility_features)
        features.update(self.trend_features)
        return features

class TechnicalIndicators:
    """Technical indicators calculation engine"""
    
    @staticmethod
    def sma(prices: np.ndarray, window: int) -> np.ndarray:
        """Simple Moving Average"""
        return pd.Series(prices).rolling(window=window, min_periods=1).mean().values
    
    @staticmethod
    def ema(prices: np.ndarray, window: int) -> np.ndarray:
        """Exponential Moving Average"""
        return pd.Series(prices).ewm(span=window, adjust=False).mean().values
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands (Upper, Middle, Lower)"""
        sma = TechnicalIndicators.sma(prices, window)
        std = pd.Series(prices).rolling(window=window, min_periods=1).std().values
        
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        return upper, sma, lower
    
    @staticmethod
    def rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Pad with zeros to match original length
        gains = np.concatenate([[0], gains])
        losses = np.concatenate([[0], losses])
        
        avg_gains = pd.Series(gains).ewm(span=window, adjust=False).mean().values
        avg_losses = pd.Series(losses).ewm(span=window, adjust=False).mean().values
        
        rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD (MACD Line, Signal Line, Histogram)"""
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic_oscillator(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_window: int = 14, d_window: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator (%K, %D)"""
        lowest_low = pd.Series(low).rolling(window=k_window, min_periods=1).min().values
        highest_high = pd.Series(high).rolling(window=k_window, min_periods=1).max().values
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        d_percent = TechnicalIndicators.sma(k_percent, d_window)
        
        return k_percent, d_percent
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        true_range[0] = high_low[0]  # First value
        
        atr = pd.Series(true_range).ewm(span=window, adjust=False).mean().values
        return atr

class VolatilityFeatures:
    """Volatility and risk measures"""
    
    @staticmethod
    def realized_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
        """Realized volatility using rolling standard deviation"""
        return pd.Series(returns).rolling(window=window, min_periods=1).std().values * np.sqrt(252)  # Annualized
    
    @staticmethod
    def garch_volatility_estimate(returns: np.ndarray, alpha: float = 0.1, beta: float = 0.85) -> np.ndarray:
        """Simple GARCH(1,1) volatility estimate"""
        n = len(returns)
        volatility = np.zeros(n)
        
        # Initialize with sample variance
        volatility[0] = np.var(returns[:min(20, n)])
        
        for i in range(1, n):
            volatility[i] = (1 - alpha - beta) * np.var(returns[:i+1]) + \
                           alpha * returns[i-1]**2 + \
                           beta * volatility[i-1]
        
        return np.sqrt(volatility)
    
    @staticmethod
    def volatility_regime(volatility: np.ndarray, threshold_low: float = 0.2, threshold_high: float = 0.4) -> np.ndarray:
        """Classify volatility regime (0: Low, 1: Medium, 2: High)"""
        regime = np.zeros(len(volatility))
        regime[volatility > threshold_low] = 1
        regime[volatility > threshold_high] = 2
        return regime

class MicroFeatures:
    """High-frequency micro-features for enhanced prediction"""
    
    @staticmethod
    def price_momentum(prices: np.ndarray, windows: List[int] = [5, 10, 20]) -> Dict[str, np.ndarray]:
        """Price momentum over multiple windows"""
        momentum = {}
        for window in windows:
            momentum[f'momentum_{window}'] = (prices - np.roll(prices, window)) / (np.roll(prices, window) + 1e-10)
        return momentum
    
    @staticmethod
    def volume_price_trend(prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Volume Price Trend indicator"""
        price_change = np.diff(prices) / (prices[:-1] + 1e-10)
        volume_change = volumes[1:] * price_change
        vpt = np.cumsum(np.concatenate([[0], volume_change]))
        return vpt
    
    @staticmethod
    def order_flow_imbalance(bid_depth: np.ndarray, ask_depth: np.ndarray) -> Dict[str, np.ndarray]:
        """Order flow imbalance metrics"""
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / (total_depth + 1e-10)
        
        return {
            'order_imbalance': imbalance,
            'depth_ratio': bid_depth / (ask_depth + 1e-10),
            'total_depth': total_depth
        }
    
    @staticmethod
    def jump_detection(prices: np.ndarray, threshold: float = 0.02) -> np.ndarray:
        """Detect price jumps exceeding threshold"""
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        jumps = np.abs(returns) > threshold
        return np.concatenate([[False], jumps]).astype(float)

class KalmanFeatures:
    """Kalman filtering for noise reduction and trend extraction"""
    
    def __init__(self):
        self.kf = None
        self.state_means = None
        self.state_covariances = None
    
    def fit_kalman_filter(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit Kalman filter to price series"""
        try:
            # Simple local level model
            self.kf = KalmanFilter(
                transition_matrices=np.array([[1.0]]),
                observation_matrices=np.array([[1.0]]),
                initial_state_mean=prices[0],
                initial_state_covariance=1.0,
                observation_covariance=1.0,
                transition_covariance=0.1
            )
            
            self.state_means, self.state_covariances = self.kf.em(prices, n_iter=10).smooth()
            
            # Extract trend and noise
            trend = self.state_means.flatten()
            noise = prices - trend
            
            return trend, noise
            
        except Exception as e:
            logger.warning(f"Kalman filter failed: {e}, using simple smoothing")
            # Fallback to simple exponential smoothing
            trend = pd.Series(prices).ewm(span=10).mean().values
            noise = prices - trend
            return trend, noise
    
    def get_kalman_features(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract Kalman-based features"""
        trend, noise = self.fit_kalman_filter(prices)
        
        return {
            'kalman_trend': trend,
            'kalman_noise': noise,
            'trend_strength': np.abs(trend) / (np.abs(prices) + 1e-10),
            'noise_ratio': np.abs(noise) / (np.abs(prices) + 1e-10)
        }

class FeatureEngineer:
    """Main feature engineering orchestrator"""
    
    def __init__(self, cache_size: int = 1000):
        self.technical_indicators = TechnicalIndicators()
        self.volatility_features = VolatilityFeatures()
        self.micro_features = MicroFeatures()
        self.kalman_features = KalmanFeatures()
        
        # Feature cache for performance optimization
        self.feature_cache = {}
        self.cache_size = cache_size
        
    async def engineer_features(self, price_data: List[PriceData], market_data: Optional[List[MarketData]] = None) -> List[FeatureSet]:
        """Engineer complete feature set from price data"""
        if not price_data:
            return []
        
        try:
            # Convert to numpy arrays for efficient computation
            timestamps = [pd.timestamp for pd in price_data]
            prices = np.array([pd.close_price for pd in price_data])
            highs = np.array([pd.high_price for pd in price_data])
            lows = np.array([pd.low_price for pd in price_data])
            volumes = np.array([pd.volume for pd in price_data])
            
            # Calculate returns
            returns = np.diff(prices) / (prices[:-1] + 1e-10)
            returns = np.concatenate([[0], returns])  # Pad to match length
            
            feature_sets = []
            
            for i, timestamp in enumerate(timestamps):
                # Use sliding window for feature calculation
                window_start = max(0, i - 100)  # Use last 100 points or available data
                window_prices = prices[window_start:i+1]
                window_highs = highs[window_start:i+1]
                window_lows = lows[window_start:i+1]
                window_volumes = volumes[window_start:i+1]
                window_returns = returns[window_start:i+1]
                
                if len(window_prices) < 2:
                    continue
                
                # Price features
                price_features = self._calculate_price_features(window_prices, i)
                
                # Technical indicators
                technical_indicators = self._calculate_technical_indicators(
                    window_prices, window_highs, window_lows, window_volumes
                )
                
                # Volatility features
                volatility_features = self._calculate_volatility_features(window_returns, window_prices)
                
                # Micro features
                micro_features = self._calculate_micro_features(
                    window_prices, window_volumes, market_data, i
                )
                
                # Trend features (including Kalman)
                trend_features = self._calculate_trend_features(window_prices)
                
                feature_set = FeatureSet(
                    timestamp=timestamp,
                    price_features=price_features,
                    technical_indicators=technical_indicators,
                    micro_features=micro_features,
                    volatility_features=volatility_features,
                    trend_features=trend_features
                )
                
                feature_sets.append(feature_set)
            
            logger.info(f"Engineered features for {len(feature_sets)} data points")
            return feature_sets
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return []
    
    def _calculate_price_features(self, prices: np.ndarray, index: int) -> Dict[str, float]:
        """Calculate basic price-based features"""
        current_price = prices[-1]
        
        features = {
            'price': current_price,
            'price_log': np.log(current_price + 1e-10),
            'price_normalized': current_price / (np.mean(prices) + 1e-10),
        }
        
        if len(prices) > 1:
            features.update({
                'price_change': prices[-1] - prices[-2],
                'price_change_pct': (prices[-1] - prices[-2]) / (prices[-2] + 1e-10),
                'price_range': np.max(prices) - np.min(prices),
                'price_position': (current_price - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-10)
            })
        
        return features
    
    def _calculate_technical_indicators(self, prices: np.ndarray, highs: np.ndarray, 
                                      lows: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """Calculate technical indicators"""
        indicators = {}
        
        try:
            # Moving averages
            if len(prices) >= 5:
                sma_5 = self.technical_indicators.sma(prices, 5)[-1]
                sma_20 = self.technical_indicators.sma(prices, min(20, len(prices)))[-1]
                ema_12 = self.technical_indicators.ema(prices, min(12, len(prices)))[-1]
                
                indicators.update({
                    'sma_5': sma_5,
                    'sma_20': sma_20,
                    'ema_12': ema_12,
                    'price_sma5_ratio': prices[-1] / (sma_5 + 1e-10),
                    'price_sma20_ratio': prices[-1] / (sma_20 + 1e-10)
                })
            
            # Bollinger Bands
            if len(prices) >= 10:
                bb_upper, bb_middle, bb_lower = self.technical_indicators.bollinger_bands(
                    prices, min(20, len(prices))
                )
                bb_position = (prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1] + 1e-10)
                
                indicators.update({
                    'bb_upper': bb_upper[-1],
                    'bb_middle': bb_middle[-1],
                    'bb_lower': bb_lower[-1],
                    'bb_position': bb_position,
                    'bb_width': (bb_upper[-1] - bb_lower[-1]) / (bb_middle[-1] + 1e-10)
                })
            
            # RSI
            if len(prices) >= 10:
                rsi = self.technical_indicators.rsi(prices, min(14, len(prices)))[-1]
                indicators['rsi'] = rsi
                indicators['rsi_overbought'] = 1.0 if rsi > 70 else 0.0
                indicators['rsi_oversold'] = 1.0 if rsi < 30 else 0.0
            
            # MACD
            if len(prices) >= 20:
                macd_line, signal_line, histogram = self.technical_indicators.macd(prices)
                indicators.update({
                    'macd_line': macd_line[-1],
                    'macd_signal': signal_line[-1],
                    'macd_histogram': histogram[-1],
                    'macd_bullish': 1.0 if macd_line[-1] > signal_line[-1] else 0.0
                })
            
            # Stochastic
            if len(prices) >= 10:
                stoch_k, stoch_d = self.technical_indicators.stochastic_oscillator(
                    highs, lows, prices, min(14, len(prices))
                )
                indicators.update({
                    'stoch_k': stoch_k[-1],
                    'stoch_d': stoch_d[-1],
                    'stoch_overbought': 1.0 if stoch_k[-1] > 80 else 0.0,
                    'stoch_oversold': 1.0 if stoch_k[-1] < 20 else 0.0
                })
            
            # ATR
            if len(prices) >= 10:
                atr = self.technical_indicators.atr(highs, lows, prices, min(14, len(prices)))[-1]
                indicators['atr'] = atr
                indicators['atr_normalized'] = atr / (prices[-1] + 1e-10)
            
        except Exception as e:
            logger.warning(f"Technical indicators calculation failed: {e}")
        
        return indicators
    
    def _calculate_volatility_features(self, returns: np.ndarray, prices: np.ndarray) -> Dict[str, float]:
        """Calculate volatility-based features"""
        features = {}
        
        try:
            if len(returns) >= 5:
                # Basic volatility measures
                vol_5 = np.std(returns[-5:]) * np.sqrt(252)  # Annualized
                vol_20 = np.std(returns[-min(20, len(returns)):]) * np.sqrt(252)
                
                features.update({
                    'volatility_5d': vol_5,
                    'volatility_20d': vol_20,
                    'volatility_ratio': vol_5 / (vol_20 + 1e-10)
                })
                
                # Realized volatility
                realized_vol = self.volatility_features.realized_volatility(returns)[-1]
                features['realized_volatility'] = realized_vol
                
                # GARCH estimate
                garch_vol = self.volatility_features.garch_volatility_estimate(returns)[-1]
                features['garch_volatility'] = garch_vol
                
                # Volatility regime
                vol_regime = self.volatility_features.volatility_regime(np.array([realized_vol]))[0]
                features['volatility_regime'] = vol_regime
                
                # Jump detection
                jumps = self.micro_features.jump_detection(prices)
                features['recent_jump'] = np.sum(jumps[-5:])  # Jumps in last 5 periods
        
        except Exception as e:
            logger.warning(f"Volatility features calculation failed: {e}")
        
        return features
    
    def _calculate_micro_features(self, prices: np.ndarray, volumes: np.ndarray, 
                                market_data: Optional[List[MarketData]], index: int) -> Dict[str, float]:
        """Calculate high-frequency micro-features"""
        features = {}
        
        try:
            # Price momentum
            momentum_features = self.micro_features.price_momentum(prices)
            for key, values in momentum_features.items():
                features[key] = values[-1]
            
            # Volume-price trend
            if len(volumes) >= 2:
                vpt = self.micro_features.volume_price_trend(prices, volumes)[-1]
                features['volume_price_trend'] = vpt
                
                # Volume features
                features.update({
                    'volume': volumes[-1],
                    'volume_sma': np.mean(volumes[-min(10, len(volumes)):]),
                    'volume_ratio': volumes[-1] / (np.mean(volumes[-min(10, len(volumes)):]) + 1e-10)
                })
            
            # Order book features (if available)
            if market_data and index < len(market_data) and market_data[index]:
                order_book = market_data[index].order_book_depth
                if order_book:
                    bid_depth = order_book.get('bid_depth', 0)
                    ask_depth = order_book.get('ask_depth', 0)
                    
                    if bid_depth > 0 and ask_depth > 0:
                        imbalance_features = self.micro_features.order_flow_imbalance(
                            np.array([bid_depth]), np.array([ask_depth])
                        )
                        for key, values in imbalance_features.items():
                            features[f'orderbook_{key}'] = values[0]
                        
                        features['spread'] = order_book.get('spread', 0)
            
        except Exception as e:
            logger.warning(f"Micro features calculation failed: {e}")
        
        return features
    
    def _calculate_trend_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate trend-based features including Kalman filtering"""
        features = {}
        
        try:
            if len(prices) >= 10:
                # Kalman filter features
                kalman_features = self.kalman_features.get_kalman_features(prices)
                for key, values in kalman_features.items():
                    features[key] = values[-1]
                
                # Trend direction
                if len(prices) >= 5:
                    short_trend = np.polyfit(range(5), prices[-5:], 1)[0]
                    long_trend = np.polyfit(range(min(20, len(prices))), prices[-min(20, len(prices)):], 1)[0]
                    
                    features.update({
                        'trend_short': short_trend,
                        'trend_long': long_trend,
                        'trend_alignment': 1.0 if short_trend * long_trend > 0 else 0.0,
                        'trend_strength': abs(short_trend) / (prices[-1] + 1e-10)
                    })
        
        except Exception as e:
            logger.warning(f"Trend features calculation failed: {e}")
        
        return features
    
    def cache_features(self, key: str, features: FeatureSet):
        """Cache features for performance optimization"""
        if len(self.feature_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.feature_cache))
            del self.feature_cache[oldest_key]
        
        self.feature_cache[key] = features
    
    def get_cached_features(self, key: str) -> Optional[FeatureSet]:
        """Retrieve cached features"""
        return self.feature_cache.get(key)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        # This would be populated based on actual feature calculation
        feature_names = [
            # Price features
            'price', 'price_log', 'price_normalized', 'price_change', 'price_change_pct',
            'price_range', 'price_position',
            
            # Technical indicators
            'sma_5', 'sma_20', 'ema_12', 'price_sma5_ratio', 'price_sma20_ratio',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'bb_width',
            'rsi', 'rsi_overbought', 'rsi_oversold',
            'macd_line', 'macd_signal', 'macd_histogram', 'macd_bullish',
            'stoch_k', 'stoch_d', 'stoch_overbought', 'stoch_oversold',
            'atr', 'atr_normalized',
            
            # Volatility features
            'volatility_5d', 'volatility_20d', 'volatility_ratio',
            'realized_volatility', 'garch_volatility', 'volatility_regime', 'recent_jump',
            
            # Micro features
            'momentum_5', 'momentum_10', 'momentum_20',
            'volume_price_trend', 'volume', 'volume_sma', 'volume_ratio',
            'orderbook_order_imbalance', 'orderbook_depth_ratio', 'orderbook_total_depth', 'spread',
            
            # Trend features
            'kalman_trend', 'kalman_noise', 'trend_strength', 'noise_ratio',
            'trend_short', 'trend_long', 'trend_alignment', 'trend_strength'
        ]
        
        return feature_names