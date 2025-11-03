#!/usr/bin/env python3
"""
Initialize historical price data for BTC prediction engine
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_price_data(days=90, frequency='1min'):
    """Generate synthetic BTC price data with realistic patterns"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate timestamps
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq=frequency)
    
    # Initial price and parameters
    base_price = 30000  # Base BTC price
    volatility = 0.02   # Daily volatility
    trend = 0.0001     # Slight upward trend
    
    # Generate prices using geometric Brownian motion
    n_steps = len(timestamps)
    dt = 1/n_steps
    prices = np.zeros(n_steps)
    prices[0] = base_price
    
    # Add price movements with realistic patterns
    for i in range(1, n_steps):
        # Random walk with drift
        drift = trend * prices[i-1]
        diffusion = volatility * prices[i-1] * np.random.normal(0, np.sqrt(dt))
        
        # Add some occasional jumps (5% chance)
        if np.random.random() < 0.05:
            jump = np.random.choice([-1, 1]) * np.random.uniform(0.01, 0.03) * prices[i-1]
        else:
            jump = 0
            
        prices[i] = prices[i-1] + drift + diffusion + jump
        
    # Ensure prices stay positive
    prices = np.maximum(prices, 100)
    
    # Create DataFrame with OHLCV data
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open_price': prices,
        'high_price': prices * (1 + np.random.uniform(0, 0.01, n_steps)),
        'low_price': prices * (1 - np.random.uniform(0, 0.01, n_steps)),
        'close_price': prices * (1 + np.random.normal(0, 0.005, n_steps)),
        'volume': np.random.lognormal(10, 1, n_steps)
    })
    
    # Ensure OHLC relationships are maintained
    df['high_price'] = df[['open_price', 'close_price', 'high_price']].max(axis=1)
    df['low_price'] = df[['open_price', 'close_price', 'low_price']].min(axis=1)
    
    return df

def main():
    """Main function to generate and save data"""
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate 90 days of minute-level data
    print("Generating synthetic BTC price data...")
    df = generate_synthetic_price_data(days=90, frequency='1min')
    
    # Save to CSV
    csv_path = os.path.join(data_dir, 'btc_price_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} price records")
    print(f"Data saved to: {csv_path}")
    
    # Create empty files for predictions and performance
    predictions_path = os.path.join(data_dir, 'predictions.csv')
    performance_path = os.path.join(data_dir, 'model_performance.csv')
    
    # Create predictions.csv with headers
    if not os.path.exists(predictions_path):
        pd.DataFrame(columns=[
            'id', 'timestamp', 'current_price', 'predicted_price', 
            'confidence_score', 'model_contributions', 'features_used', 
            'prediction_horizon'
        ]).to_csv(predictions_path, index=False)
        print(f"Created: {predictions_path}")
    
    # Create model_performance.csv with headers
    if not os.path.exists(performance_path):
        pd.DataFrame(columns=[
            'model_name', 'timestamp', 'mae', 'rmse', 
            'accuracy_within_threshold', 'rapid_change_detection_rate'
        ]).to_csv(performance_path, index=False)
        print(f"Created: {performance_path}")

if __name__ == "__main__":
    main()