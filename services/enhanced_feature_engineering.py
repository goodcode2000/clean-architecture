"""Enhanced feature engineering with market sentiment, news, and advanced features."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
import requests
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from services.feature_engineering import FeatureEngineer

class EnhancedFeatureEngineer(FeatureEngineer):
    """
    Enhanced feature engineering with:
    - Market sentiment indicators
    - News sentiment analysis
    - On-chain metrics
    - Advanced technical indicators
    - Market microstructure features
    """
    
    def __init__(self):
        super().__init__()
        self.news_api_key = os.getenv('NEWS_API_KEY', None)
        self.sentiment_cache = {}
        
    def calculate_market_sentiment_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market sentiment indicators from price action.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with sentiment indicators added
        """
        try:
            df = df.copy()
            
            # Fear & Greed Index (simplified version based on price action)
            # Components: Momentum, Volatility, Volume, Market Strength
            
            # 1. Momentum component (25%)
            df['momentum_7d'] = df['close'].pct_change(periods=7 * 288)  # 7 days in 5-min intervals
            df['momentum_30d'] = df['close'].pct_change(periods=30 * 288)
            df['momentum_score'] = (
                (df['momentum_7d'] > 0).astype(int) * 50 +
                (df['momentum_30d'] > 0).astype(int) * 50
            )
            
            # 2. Volatility component (25%)
            current_vol = df['volatility_20']
            avg_vol = df['volatility_20'].rolling(window=50).mean()
            df['volatility_ratio'] = current_vol / avg_vol
            df['volatility_score'] = 100 - np.clip(df['volatility_ratio'] * 50, 0, 100)
            
            # 3. Volume component (25%)
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['volume_score'] = np.clip(df['volume_ratio'] * 50, 0, 100)
            
            # 4. Market strength component (25%)
            df['new_highs'] = (df['close'] >= df['close'].rolling(window=50).max()).astype(int)
            df['new_lows'] = (df['close'] <= df['close'].rolling(window=50).min()).astype(int)
            df['strength_score'] = df['new_highs'] * 100
            
            # Combined Fear & Greed Index (0-100)
            df['fear_greed_index'] = (
                df['momentum_score'] * 0.25 +
                df['volatility_score'] * 0.25 +
                df['volume_score'] * 0.25 +
                df['strength_score'] * 0.25
            )
            
            # Sentiment classification
            df['extreme_fear'] = df['fear_greed_index'] < 25
            df['fear'] = (df['fear_greed_index'] >= 25) & (df['fear_greed_index'] < 45)
            df['neutral'] = (df['fear_greed_index'] >= 45) & (df['fear_greed_index'] < 55)
            df['greed'] = (df['fear_greed_index'] >= 55) & (df['fear_greed_index'] < 75)
            df['extreme_greed'] = df['fear_greed_index'] >= 75
            
            # Sentiment momentum
            df['sentiment_change'] = df['fear_greed_index'].diff()
            df['sentiment_acceleration'] = df['sentiment_change'].diff()
            
            logger.debug("Market sentiment indicators calculated")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate market sentiment: {e}")
            return df
    
    def calculate_volume_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume profile and order flow features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume profile features
        """
        try:
            df = df.copy()
            
            # Volume-weighted average price (VWAP)
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            df['price_to_vwap'] = (df['close'] - df['vwap']) / df['vwap']
            
            # On-Balance Volume (OBV)
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            df['obv_ma'] = df['obv'].rolling(window=20).mean()
            df['obv_divergence'] = df['obv'] - df['obv_ma']
            
            # Volume Rate of Change
            df['volume_roc'] = df['volume'].pct_change(periods=10)
            
            # Accumulation/Distribution Line
            clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            df['ad_line'] = (clv * df['volume']).cumsum()
            df['ad_line_ma'] = df['ad_line'].rolling(window=20).mean()
            
            # Money Flow Index (MFI)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
            
            positive_mf = positive_flow.rolling(window=14).sum()
            negative_mf = negative_flow.rolling(window=14).sum()
            
            mfi_ratio = positive_mf / negative_mf
            df['mfi'] = 100 - (100 / (1 + mfi_ratio))
            
            # Volume-based signals
            df['volume_spike'] = df['volume'] > (df['volume_ma'] * 2)
            df['volume_dry_up'] = df['volume'] < (df['volume_ma'] * 0.5)
            
            # Buying/Selling pressure
            df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
            
            logger.debug("Volume profile features calculated")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate volume profile: {e}")
            return df
    
    def calculate_trend_strength_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend strength and direction indicators.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with trend indicators
        """
        try:
            df = df.copy()
            
            # Average Directional Index (ADX)
            high_diff = df['high'].diff()
            low_diff = -df['low'].diff()
            
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            
            tr = df['true_range']
            atr = tr.rolling(window=14).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(window=14).mean()
            df['plus_di'] = plus_di
            df['minus_di'] = minus_di
            
            # Trend strength classification
            df['strong_trend'] = df['adx'] > 25
            df['weak_trend'] = df['adx'] < 20
            df['bullish_trend'] = (df['plus_di'] > df['minus_di']) & df['strong_trend']
            df['bearish_trend'] = (df['minus_di'] > df['plus_di']) & df['strong_trend']
            
            # Parabolic SAR
            df['psar'] = self._calculate_parabolic_sar(df)
            df['price_above_psar'] = df['close'] > df['psar']
            
            # Ichimoku Cloud components
            high_9 = df['high'].rolling(window=9).max()
            low_9 = df['low'].rolling(window=9).min()
            df['tenkan_sen'] = (high_9 + low_9) / 2
            
            high_26 = df['high'].rolling(window=26).max()
            low_26 = df['low'].rolling(window=26).min()
            df['kijun_sen'] = (high_26 + low_26) / 2
            
            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
            
            high_52 = df['high'].rolling(window=52).max()
            low_52 = df['low'].rolling(window=52).min()
            df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
            
            df['price_above_cloud'] = df['close'] > df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
            df['price_below_cloud'] = df['close'] < df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
            
            logger.debug("Trend strength indicators calculated")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate trend strength: {e}")
            return df
    
    def _calculate_parabolic_sar(self, df: pd.DataFrame, af_start: float = 0.02, 
                                  af_increment: float = 0.02, af_max: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR indicator."""
        try:
            psar = df['close'].copy()
            bull = True
            af = af_start
            ep = df['low'].iloc[0]
            hp = df['high'].iloc[0]
            lp = df['low'].iloc[0]
            
            for i in range(1, len(df)):
                if bull:
                    psar.iloc[i] = psar.iloc[i-1] + af * (hp - psar.iloc[i-1])
                else:
                    psar.iloc[i] = psar.iloc[i-1] + af * (lp - psar.iloc[i-1])
                
                reverse = False
                
                if bull:
                    if df['low'].iloc[i] < psar.iloc[i]:
                        bull = False
                        reverse = True
                        psar.iloc[i] = hp
                        lp = df['low'].iloc[i]
                        af = af_start
                else:
                    if df['high'].iloc[i] > psar.iloc[i]:
                        bull = True
                        reverse = True
                        psar.iloc[i] = lp
                        hp = df['high'].iloc[i]
                        af = af_start
                
                if not reverse:
                    if bull:
                        if df['high'].iloc[i] > hp:
                            hp = df['high'].iloc[i]
                            af = min(af + af_increment, af_max)
                    else:
                        if df['low'].iloc[i] < lp:
                            lp = df['low'].iloc[i]
                            af = min(af + af_increment, af_max)
            
            return psar
            
        except Exception as e:
            logger.error(f"Failed to calculate Parabolic SAR: {e}")
            return df['close']
    
    def calculate_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market microstructure features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with microstructure features
        """
        try:
            df = df.copy()
            
            # Spread indicators
            df['hl_spread'] = (df['high'] - df['low']) / df['close']
            df['oc_spread'] = abs(df['open'] - df['close']) / df['close']
            
            # Price impact
            df['price_impact'] = df['returns'] / np.log1p(df['volume'])
            
            # Amihud illiquidity measure
            df['illiquidity'] = abs(df['returns']) / df['volume']
            df['illiquidity_ma'] = df['illiquidity'].rolling(window=20).mean()
            
            # Roll's spread estimator
            df['roll_spread'] = 2 * np.sqrt(abs(df['returns'].rolling(window=2).cov(df['returns'].shift(1))))
            
            # Effective spread
            df['effective_spread'] = 2 * abs(df['close'] - (df['high'] + df['low']) / 2)
            
            # Quote imbalance
            df['quote_imbalance'] = (df['high'] - df['close']) / (df['high'] - df['low'])
            
            # Trade intensity
            df['trade_intensity'] = df['volume'] / df['hl_spread']
            
            logger.debug("Market microstructure features calculated")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate microstructure features: {e}")
            return df
    
    def calculate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced statistical features.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with statistical features
        """
        try:
            df = df.copy()
            
            # Skewness and Kurtosis
            windows = [20, 50, 100]
            for window in windows:
                df[f'returns_skew_{window}'] = df['returns'].rolling(window=window).skew()
                df[f'returns_kurt_{window}'] = df['returns'].rolling(window=window).kurt()
            
            # Hurst exponent (simplified)
            df['hurst_exponent'] = self._calculate_hurst_exponent(df['close'], window=100)
            
            # Autocorrelation
            for lag in [1, 5, 10]:
                df[f'returns_autocorr_{lag}'] = df['returns'].rolling(window=50).apply(
                    lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
                )
            
            # Entropy (price distribution)
            df['price_entropy'] = df['returns'].rolling(window=50).apply(
                lambda x: -np.sum(x * np.log(x + 1e-10)) if len(x) > 0 else 0
            )
            
            # Fractal dimension
            df['fractal_dimension'] = self._calculate_fractal_dimension(df['close'], window=50)
            
            # Coefficient of variation
            df['cv_20'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
            df['cv_50'] = df['close'].rolling(window=50).std() / df['close'].rolling(window=50).mean()
            
            logger.debug("Statistical features calculated")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate statistical features: {e}")
            return df
    
    def _calculate_hurst_exponent(self, series: pd.Series, window: int = 100) -> pd.Series:
        """Calculate Hurst exponent for trend persistence."""
        try:
            def hurst(ts):
                if len(ts) < 20:
                    return 0.5
                lags = range(2, min(20, len(ts) // 2))
                tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            
            return series.rolling(window=window).apply(hurst)
            
        except Exception as e:
            logger.error(f"Failed to calculate Hurst exponent: {e}")
            return pd.Series([0.5] * len(series), index=series.index)
    
    def _calculate_fractal_dimension(self, series: pd.Series, window: int = 50) -> pd.Series:
        """Calculate fractal dimension."""
        try:
            def fractal_dim(ts):
                if len(ts) < 10:
                    return 1.5
                n = len(ts)
                L = []
                x = np.arange(n)
                for k in range(1, min(10, n // 2)):
                    Lk = np.sum(np.abs(ts[k:] - ts[:-k]))
                    L.append(Lk / k)
                if len(L) > 1:
                    return 1 - np.polyfit(np.log(range(1, len(L) + 1)), np.log(L), 1)[0]
                return 1.5
            
            return series.rolling(window=window).apply(fractal_dim)
            
        except Exception as e:
            logger.error(f"Failed to calculate fractal dimension: {e}")
            return pd.Series([1.5] * len(series), index=series.index)
    
    def fetch_news_sentiment(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch news sentiment data for Bitcoin.
        
        Args:
            start_date: Start date for news
            end_date: End date for news
            
        Returns:
            DataFrame with news sentiment scores
        """
        try:
            if not self.news_api_key:
                logger.warning("News API key not configured, skipping news sentiment")
                return pd.DataFrame()
            
            # Check cache
            cache_key = f"{start_date.date()}_{end_date.date()}"
            if cache_key in self.sentiment_cache:
                logger.debug("Using cached news sentiment data")
                return self.sentiment_cache[cache_key]
            
            # Fetch news from API (example using NewsAPI)
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': 'bitcoin OR BTC OR cryptocurrency',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                
                # Process articles and calculate sentiment
                sentiment_data = []
                for article in articles:
                    published_at = pd.to_datetime(article['publishedAt'])
                    title = article.get('title', '')
                    description = article.get('description', '')
                    
                    # Simple sentiment analysis (can be enhanced with NLP models)
                    sentiment_score = self._analyze_text_sentiment(title + ' ' + description)
                    
                    sentiment_data.append({
                        'timestamp': published_at,
                        'sentiment_score': sentiment_score,
                        'title': title
                    })
                
                sentiment_df = pd.DataFrame(sentiment_data)
                
                # Cache the results
                self.sentiment_cache[cache_key] = sentiment_df
                
                logger.info(f"Fetched {len(sentiment_df)} news articles")
                return sentiment_df
            else:
                logger.warning(f"Failed to fetch news: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to fetch news sentiment: {e}")
            return pd.DataFrame()
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text (simplified version).
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1 to 1)
        """
        try:
            # Simple keyword-based sentiment (can be replaced with proper NLP model)
            positive_keywords = ['bullish', 'surge', 'rally', 'gain', 'rise', 'up', 'high', 
                               'positive', 'growth', 'increase', 'profit', 'boom', 'soar']
            negative_keywords = ['bearish', 'crash', 'fall', 'drop', 'down', 'low', 
                               'negative', 'decline', 'loss', 'plunge', 'dump', 'fear']
            
            text_lower = text.lower()
            
            positive_count = sum(1 for word in positive_keywords if word in text_lower)
            negative_count = sum(1 for word in negative_keywords if word in text_lower)
            
            total_count = positive_count + negative_count
            if total_count == 0:
                return 0.0
            
            sentiment = (positive_count - negative_count) / total_count
            return sentiment
            
        except Exception as e:
            logger.error(f"Failed to analyze sentiment: {e}")
            return 0.0
    
    def add_news_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add news sentiment features to price data.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with news sentiment features
        """
        try:
            if len(df) == 0:
                return df
            
            df = df.copy()
            
            # Fetch news for the date range
            start_date = df['timestamp'].min()
            end_date = df['timestamp'].max()
            
            news_df = self.fetch_news_sentiment(start_date, end_date)
            
            if len(news_df) == 0:
                # Add default sentiment features
                df['news_sentiment'] = 0.0
                df['news_sentiment_ma'] = 0.0
                df['news_count'] = 0
                logger.warning("No news data available, using default values")
                return df
            
            # Aggregate news sentiment by hour
            news_df['hour'] = news_df['timestamp'].dt.floor('H')
            hourly_sentiment = news_df.groupby('hour').agg({
                'sentiment_score': ['mean', 'std', 'count']
            }).reset_index()
            
            hourly_sentiment.columns = ['timestamp', 'news_sentiment', 'news_sentiment_std', 'news_count']
            
            # Merge with price data
            df['hour'] = df['timestamp'].dt.floor('H')
            df = df.merge(hourly_sentiment, left_on='hour', right_on='timestamp', 
                         how='left', suffixes=('', '_news'))
            
            # Fill missing values
            df['news_sentiment'] = df['news_sentiment'].fillna(method='ffill').fillna(0)
            df['news_sentiment_std'] = df['news_sentiment_std'].fillna(0)
            df['news_count'] = df['news_count'].fillna(0)
            
            # Calculate rolling sentiment features
            df['news_sentiment_ma'] = df['news_sentiment'].rolling(window=24).mean()
            df['news_sentiment_change'] = df['news_sentiment'].diff()
            
            # Sentiment divergence from price
            df['sentiment_price_divergence'] = (
                (df['news_sentiment'] > 0) & (df['returns'] < 0) |
                (df['news_sentiment'] < 0) & (df['returns'] > 0)
            ).astype(int)
            
            # Drop temporary column
            df = df.drop('hour', axis=1)
            
            logger.info("News sentiment features added")
            return df
            
        except Exception as e:
            logger.error(f"Failed to add news sentiment features: {e}")
            # Add default features
            df['news_sentiment'] = 0.0
            df['news_sentiment_ma'] = 0.0
            df['news_count'] = 0
            return df
    
    def create_all_enhanced_features(self, df: pd.DataFrame, include_news: bool = False) -> pd.DataFrame:
        """
        Create all enhanced features including sentiment and advanced indicators.
        
        Args:
            df: DataFrame with OHLCV data
            include_news: Whether to include news sentiment (requires API key)
            
        Returns:
            DataFrame with all enhanced features
        """
        try:
            logger.info("Creating enhanced features for BTC price prediction...")
            
            # Start with base features
            features_df = super().create_all_features(df)
            
            # Add enhanced features
            features_df = self.calculate_market_sentiment_indicators(features_df)
            features_df = self.calculate_volume_profile_features(features_df)
            features_df = self.calculate_trend_strength_indicators(features_df)
            features_df = self.calculate_market_microstructure_features(features_df)
            features_df = self.calculate_statistical_features(features_df)
            
            # Add news sentiment if requested
            if include_news:
                features_df = self.add_news_sentiment_features(features_df)
            
            # Remove rows with NaN values
            initial_rows = len(features_df)
            features_df = features_df.dropna()
            final_rows = len(features_df)
            
            if initial_rows != final_rows:
                logger.info(f"Removed {initial_rows - final_rows} rows with NaN values")
            
            logger.info(f"Enhanced feature engineering completed: {len(features_df.columns)} features created")
            return features_df
            
        except Exception as e:
            logger.error(f"Failed to create enhanced features: {e}")
            return df
    
    def get_enhanced_feature_importance_columns(self) -> List[str]:
        """
        Get list of important enhanced feature columns.
        
        Returns:
            List of feature column names
        """
        base_features = super().get_feature_importance_columns()
        
        enhanced_features = [
            # Market sentiment
            'fear_greed_index', 'sentiment_change',
            
            # Volume profile
            'vwap', 'price_to_vwap', 'obv', 'mfi',
            'buying_pressure', 'selling_pressure',
            
            # Trend strength
            'adx', 'plus_di', 'minus_di',
            'price_above_psar', 'price_above_cloud',
            
            # Microstructure
            'hl_spread', 'price_impact', 'illiquidity',
            
            # Statistical
            'returns_skew_20', 'hurst_exponent', 'fractal_dimension',
            
            # News sentiment (if available)
            'news_sentiment', 'news_sentiment_ma'
        ]
        
        return base_features + enhanced_features
