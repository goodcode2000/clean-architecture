import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def main():
    """Initialize data for BTC prediction engine"""
    try:
        # Get the absolute path to the data directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        print(f"Initializing data in: {data_dir}")
        
        # Generate timestamps for 90 days of minute data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        dates = pd.date_range(start=start_date, end=end_date, freq='1min')
        
        print("Generating synthetic price data...")
        n_points = len(dates)
        
        # Generate base price using geometric Brownian motion
        dt = 1.0/n_points
        mu = 0.1  # Annual drift term
        sigma = 0.5  # Annual volatility
        
        # Generate random walks
        W = np.random.standard_normal(size=n_points)
        W = np.cumsum(W)*np.sqrt(dt)  # Cumulative sum for the Brownian motion
        
        # Calculate price path
        initial_price = 30000
        prices = initial_price * np.exp((mu - 0.5*sigma**2)*np.arange(n_points)*dt + sigma*W)
        
        # Ensure all prices are valid
        prices = np.maximum(100, np.minimum(100000, prices))  # Cap between $100 and $100,000
        
        # Create DataFrame with all required fields
        df = pd.DataFrame({
            'timestamp': dates,
            'open_price': prices,
            'close_price': [p * (1 + np.random.normal(0, 0.0002)) for p in prices],
            'high_price': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'low_price': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'volume': np.random.lognormal(10, 1, len(dates))
        })
        
        # Ensure high/low prices are consistent
        df['high_price'] = df[['open_price', 'close_price', 'high_price']].max(axis=1)
        df['low_price'] = df[['open_price', 'close_price', 'low_price']].min(axis=1)
        
        # Save price data
        price_file = os.path.join(data_dir, 'btc_price_data.csv')
        df.to_csv(price_file, index=False)
        print(f"Generated {len(df)} price records in {price_file}")
        
        # Create empty predictions file
        predictions_file = os.path.join(data_dir, 'predictions.csv')
        pd.DataFrame(columns=[
            'id', 'timestamp', 'current_price', 'predicted_price', 
            'confidence_score', 'model_contributions', 'features_used', 
            'prediction_horizon'
        ]).to_csv(predictions_file, index=False)
        print(f"Created predictions file: {predictions_file}")
        
        # Create empty performance file
        performance_file = os.path.join(data_dir, 'model_performance.csv')
        pd.DataFrame(columns=[
            'model_name', 'timestamp', 'mae', 'rmse', 
            'accuracy_within_threshold', 'rapid_change_detection_rate'
        ]).to_csv(performance_file, index=False)
        print(f"Created performance file: {performance_file}")
        
        # Verify data
        df_check = pd.read_csv(price_file)
        print(f"\nVerification:")
        print(f"Total records: {len(df_check)}")
        print(f"Date range: {df_check['timestamp'].iloc[0]} to {df_check['timestamp'].iloc[-1]}")
        print(f"Price range: ${df_check['close_price'].min():.2f} to ${df_check['close_price'].max():.2f}")
        
    except Exception as e:
        print(f"Error initializing data: {e}")
        raise

if __name__ == "__main__":
    main()