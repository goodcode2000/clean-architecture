#!/usr/bin/env bash
# Minimal setup script for prediction-engine only
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Prediction Engine root: $ROOT_DIR"

VENV_DIR="$ROOT_DIR/venv"
PYTHON="python3"

echo "Creating virtual environment at $VENV_DIR (if missing)"
if [ ! -d "$VENV_DIR" ]; then
  $PYTHON -m venv "$VENV_DIR"
fi

echo "Activating venv"
source "$VENV_DIR/bin/activate"

echo "Upgrading pip and build tools"
python -m pip install --upgrade pip setuptools wheel

echo "Installing prediction-engine requirements"
pip install -r "$ROOT_DIR/requirements.txt"

echo "Creating runtime directories"
mkdir -p "$ROOT_DIR/data"
mkdir -p "$ROOT_DIR/logs"
mkdir -p "$ROOT_DIR/models"

# Seed synthetic historical data if price CSV is missing
PRICE_FILE="$ROOT_DIR/data/btc_price_data.csv"
if [ ! -f "$PRICE_FILE" ] || [ $(wc -l < "$PRICE_FILE") -lt 50 ]; then
  echo "Creating synthetic historical price data..."
  python - << 'EOF'
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate synthetic price data
np.random.seed(42)
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
dates = pd.date_range(start=start_date, end=end_date, freq='1min')

# Generate random walk prices
price = 30000  # Starting BTC price
prices = [price]
for _ in range(len(dates)-1):
    change = np.random.normal(0, 50)  # Random price change
    price += change
    prices.append(max(1, price))  # Ensure price stays positive

# Create and save DataFrame
df = pd.DataFrame({
    'timestamp': dates,
    'price': prices
})
df.to_csv('data/btc_price_data.csv', index=False)
print(f"Created synthetic price data with {len(df)} records")
EOF
else
  echo "Found existing price data ($PRICE_FILE)"
fi

echo "Setup complete. Activate the venv with: source venv/bin/activate"
echo "To run prediction engine: python main.py --mode api --host 0.0.0.0 --port 8000"

exit 0