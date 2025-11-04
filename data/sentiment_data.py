"""
Sentiment data collection from various sources like Twitter, Reddit, and news.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from textblob import TextBlob
from newsapi import NewsApiClient
import tweepy
import praw
from config.config import Config

class SentimentCollector:
    def __init__(self, config: Config):
        self.config = config
        self.news_api = NewsApiClient(api_key=config.NEWS_API_KEY)
        self.twitter_auth = tweepy.OAuthHandler(
            config.TWITTER_API_KEY,
            config.TWITTER_API_SECRET
        )
        self.twitter_api = tweepy.API(self.twitter_auth)
        self.reddit = praw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent=config.REDDIT_USER_AGENT
        )

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using TextBlob."""
        analysis = TextBlob(text)
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }

    async def fetch_news_sentiment(self, query: str = 'bitcoin OR crypto', days: int = 1) -> pd.DataFrame:
        """Fetch and analyze news articles sentiment."""
        articles = self.news_api.get_everything(
            q=query,
            language='en',
            from_param=pd.Timestamp.now() - pd.Timedelta(days=days)
        )

        sentiments = []
        for article in articles['articles']:
            text = f"{article['title']} {article['description']}"
            sentiment = self.analyze_sentiment(text)
            sentiments.append({
                'timestamp': pd.to_datetime(article['publishedAt']),
                'source': article['source']['name'],
                'polarity': sentiment['polarity'],
                'subjectivity': sentiment['subjectivity']
            })

        return pd.DataFrame(sentiments)

    async def fetch_social_sentiment(self, query: str = 'bitcoin', limit: int = 100) -> Dict[str, pd.DataFrame]:
        """Fetch and analyze social media sentiment."""
        # Twitter
        tweets = []
        for tweet in tweepy.Cursor(self.twitter_api.search_tweets, q=query).items(limit):
            sentiment = self.analyze_sentiment(tweet.text)
            tweets.append({
                'timestamp': tweet.created_at,
                'polarity': sentiment['polarity'],
                'subjectivity': sentiment['subjectivity']
            })

        # Reddit
        subreddit = self.reddit.subreddit('cryptocurrency+bitcoin')
        submissions = []
        for submission in subreddit.hot(limit=limit):
            sentiment = self.analyze_sentiment(submission.title + " " + submission.selftext)
            submissions.append({
                'timestamp': pd.to_datetime(submission.created_utc, unit='s'),
                'polarity': sentiment['polarity'],
                'subjectivity': sentiment['subjectivity']
            })

        return {
            'twitter': pd.DataFrame(tweets),
            'reddit': pd.DataFrame(submissions)
        }

    def aggregate_sentiment(self, news_df: pd.DataFrame, social_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Aggregate sentiment from all sources with weights."""
        dfs = []
        
        # Resample each source to hourly and calculate weighted average
        if not news_df.empty:
            news_hourly = news_df.set_index('timestamp').resample('1H').mean()
            news_hourly['weight'] = 0.4  # News weight
            dfs.append(news_hourly)
            
        if not social_data['twitter'].empty:
            twitter_hourly = social_data['twitter'].set_index('timestamp').resample('1H').mean()
            twitter_hourly['weight'] = 0.3  # Twitter weight
            dfs.append(twitter_hourly)
            
        if not social_data['reddit'].empty:
            reddit_hourly = social_data['reddit'].set_index('timestamp').resample('1H').mean()
            reddit_hourly['weight'] = 0.3  # Reddit weight
            dfs.append(reddit_hourly)

        # Combine all sources
        combined = pd.concat(dfs)
        return combined.groupby(level=0).apply(
            lambda x: pd.Series({
                'sentiment_score': np.average(x['polarity'], weights=x['weight']),
                'sentiment_magnitude': np.average(x['subjectivity'], weights=x['weight'])
            })
        )