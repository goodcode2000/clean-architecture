# BTC Price Prediction App 1 - Ubuntu VPS

This is the first application that runs on Ubuntu VPS with 16GB GPU for training and predicting BTC prices using ensemble learning.

## Features

- **Ensemble Models**: ETS + GARCH + LightGBM + TCN + CNN
- **Real-time Updates**: Predictions updated every 5 minutes
- **Offset Tracking**: Tracks and corrects prediction offsets (target: 20-50 USD)
- **Rapid Change Detection**: Predicts rapid price movements (>2%)
- **Feature Engineering**: Technical indicators, volatility features, trend classification

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data models
```

## Usage

```bash
# Run the application
python main.py
```

The app will:
1. Perform initial training with 90 days of historical data
2. Load existing models if available
3. Show current real price
4. Make predictions every 5 minutes
5. Display predictions and offset statistics in terminal
6. Retrain models every 24 hours

## Configuration

Edit `config.py` to adjust:
- Model weights
- Training intervals
- Feature engineering parameters
- Prediction targets

## Output

- Terminal display showing:
  - Current real price
  - Predicted price 5 minutes later
  - Individual model predictions
  - Offset statistics
- CSV file at `data/predictions_history.csv` with all predictions for offset analysis

## Model Components

- **ETS**: Captures level, trend, seasonality
- **GARCH**: Models time-varying volatility
- **LightGBM**: Handles complex nonlinear interactions
- **TCN**: Captures long-range temporal dependencies
- **CNN**: Extracts local temporal patterns

