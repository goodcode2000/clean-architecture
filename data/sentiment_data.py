"""
Sentiment data collection - simplified version without external APIs.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import requests
from config.config import Config

class SentimentCollector:
    def __init__(self, config: Config):
        self.config = config
        # Simple sentiment word lists
        self.positive_words = ['bullish', 'moon', 'pump', 'buy', 'hodl', 'green', 'up', 'rise', 'gain']
        self.negative_words = ['bearish', 'dump', 'sell', 'crash', 'red', 'down', 'fall', 'loss', 'fear']

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Simple sentiment analysis using word counting."""
        if not text:
            return {'polarity': 0.0, 'subjectivity': 0.5}
        
        text_lower = text.lower()
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            polarity = 0.0
            subjectivity = 0.1
        else:
            polarity = (positive_count - negative_count) / max(total_sentiment_words, 1)
            subjectivity = min(total_sentiment_words / len(text_lower.split()), 1.0)
        
        return {
            'polarity': max(-1.0, min(1.0, polarity)),
            'subjectivity': max(0.0, min(1.0, subjectivity))
        }

    def fetch_news_sentiment(self, query: str = 'bitcoin OR crypto', days: int = 1) -> pd.DataFrame:
        """Generate mock sentiment data for news (simplified version)."""
        # Generate some mock sentiment data based on market volatility
        timestamps = pd.date_range(
            start=pd.Timestamp.now() - pd.Timedelta(days=days),
            end=pd.Timestamp.now(),
            freq='1H'
        )
        
        sentiments = []
        for ts in timestamps:
            # Simple sentiment based on time patterns (mock data)
            hour = ts.hour
            base_sentiment = 0.1 * np.sin(hour * np.pi / 12)  # Daily cycle
            noise = np.random.normal(0, 0.2)
            
            sentiments.append({
                'timestamp': ts,
                'source': 'mock_news',
                'polarity': max(-1.0, min(1.0, base_sentiment + noise)),
                'subjectivity': np.random.uniform(0.3, 0.8)
            })

        return pd.DataFrame(sentiments)

    def fetch_social_sentiment(self, query: str = 'bitcoin', limit: int = 100) -> Dict[str, pd.DataFrame]:
        """Generate mock social sentiment data (simplified version)."""
        timestamps = pd.date_range(
            start=pd.Timestamp.now() - pd.Timedelta(hours=24),
            end=pd.Timestamp.now(),
            freq='1H'
        )
        
        twitter_data = []
        reddit_data = []
        
        for ts in timestamps:
            # Mock Twitter sentiment
            twitter_sentiment = np.random.normal(0, 0.3)
            twitter_data.append({
                'timestamp': ts,
                'polarity': max(-1.0, min(1.0, twitter_sentiment)),
                'subjectivity': np.random.uniform(0.4, 0.9)
            })
            
            # Mock Reddit sentiment
            reddit_sentiment = np.random.normal(0, 0.25)
            reddit_data.append({
                'timestamp': ts,
                'polarity': max(-1.0, min(1.0, reddit_sentiment)),
                'subjectivity': np.random.uniform(0.3, 0.7)
            })

        return {
            'twitter': pd.DataFrame(twitter_data),
            'reddit': pd.DataFrame(reddit_data)
        }

    def aggregate_sentiment(self, news_df: pd.DataFrame, social_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Aggregate sentiment from all sources with weights."""
        try:
            dfs = []
            
            # Resample each source to hourly and calculate weighted average
            if not news_df.empty:
                news_hourly = news_df.set_index('timestamp').resample('1H').mean()
                news_hourly['weight'] = 0.4  # News weight
                dfs.append(news_hourly)
                
            if 'twitter' in social_data and not social_data['twitter'].empty:
                twitter_hourly = social_data['twitter'].set_index('timestamp').resample('1H').mean()
                twitter_hourly['weight'] = 0.3  # Twitter weight
                dfs.append(twitter_hourly)
                
            if 'reddit' in social_data and not social_data['reddit'].empty:
                reddit_hourly = social_data['reddit'].set_index('timestamp').resample('1H').mean()
                reddit_hourly['weight'] = 0.3  # Reddit weight
                dfs.append(reddit_hourly)

            if not dfs:
                # Return empty DataFrame if no data
                return pd.DataFrame(columns=['sentiment_score', 'sentiment_magnitude'])

            # Combine all sources
            combined = pd.concat(dfs)
            return combined.groupby(level=0).apply(
                lambda x: pd.Series({
                    'sentiment_score': np.average(x['polarity'], weights=x['weight']),
                    'sentiment_magnitude': np.average(x['subjectivity'], weights=x['weight'])
                })
            )
        except Exception:
            # Return neutral sentiment if aggregation fails
            return pd.DataFrame({
                'sentiment_score': [0.0],
                'sentiment_magnitude': [0.5]
            }, index=[pd.Timestamp.now()])