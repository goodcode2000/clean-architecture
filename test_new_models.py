#!/usr/bin/env python3
"""
Test script for new ARIMA and XGBoost models.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.arima_model import ARIMAPredictor
from models.xgboost_model import XGBoostPredictor
from services.dynamic_weighting import DynamicWeightManager
from loguru import logger

def create_test_data(n_samples=200):
    """Create synthetic BTC price data for testing."""
    # Generate realistic BTC price data
    np.random.seed(42)
    
    # Start with a base price
    base_price = 100000
    
    # Generate price changes with some trend and volatility
    price_changes = np.random.normal(0, 0.02, n_samples)  # 2% volatility
    
    # Add some trend
    trend = np.linspace(0, 0.1, n_samples)  # 10% upward trend over period
    price_changes += trend / n_samples
    
    # Calculate cumulative prices
    prices = [base_price]
    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create DataFrame
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(minutes=5*n_samples),
        periods=n_samples + 1,
        freq='5min'
    )
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices,
        'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'volume': np.random.uniform(1000, 10000, n_samples + 1)
    })
    
    # Add some basic features for XGBoost
    df['price_change'] = df['close'].pct_change()
    df['volatil