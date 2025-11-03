import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

# Generate timestamps for 90 days of minute data
end_date = datetime.now()
start_date = end_date - timedelta(days=90)
dates = pd.date_range(start=start_date, end=end_date, freq='1min')

# Generate synthetic price data with realistic patterns
initial_price = 30000
prices = [initial_price]
for _ in range(len(dates)-1):
    # Add random price movement with momentum
    change_pct = np.random.normal(0, 0.001)  # 0.1% standard deviation
    momentum = np.sum([p - prices[max(0, len(prices)-5)] for p in prices[-5:]]) / 1000
    new_price = prices[-1] * (1 + change_pct + momentum)
    prices.append(max(100, new_price))  # Ensure price stays above 100

# Create DataFrame with all required columns
df = pd.DataFrame({
    'timestamp': dates,
    'open_price': prices,
    'close_price': [p * (1 + np.random.normal(0, 0.0002)) for p in prices],
    'volume': np.random.lognormal(10, 1, len(dates))
})

# Add high/low prices
df['high_price'] = df[['open_price', 'close_price']].max(axis=1) * (1 + abs(np.random.normal(0, 0.001, len(df))))
df['low_price'] = df[['open_price', 'close_price']].min(axis=1) * (1 - abs(np.random.normal(0, 0.001, len(df))))

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Save to CSV
df.to_csv('data/btc_price_data.csv', index=False)
print(f"Generated {len(df)} price records spanning {(end_date - start_date).days} days")

# Create empty predictions and performance files with headers
pd.DataFrame(columns=[
    'id', 'timestamp', 'current_price', 'predicted_price', 
    'confidence_score', 'model_contributions', 'features_used', 
    'prediction_horizon'
]).to_csv('data/predictions.csv', index=False)

pd.DataFrame(columns=[
    'model_name', 'timestamp', 'mae', 'rmse', 
    'accuracy_within_threshold', 'rapid_change_detection_rate'
]).to_csv('data/model_performance.csv', index=False)

print("Data initialization complete!")