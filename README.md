# Enhanced BTC Price Predictor

## 🚀 Advanced Cryptocurrency Price Prediction System
A comprehensive Bitcoin price prediction system combining multiple data sources, advanced ML models, and real-time analysis.

## Core Features

### 📊 Multi-Source Data Analysis
- ✅ **Price Data**: OHLCV for BTC and correlated assets (ETH, BNB)
- 🗣️ **Sentiment Analysis**: 
  - News articles
  - Twitter sentiment
  - Reddit discussions
- 📈 **Market Depth**: 
  - Order book analysis
  - Buy/sell imbalance
  - Liquidity tracking
- 💹 **Derivatives Data**:
  - Funding rates
  - Open interest
  - Options data
- 📉 **Technical Indicators**:
  - Advanced indicators (RSI, MACD, BB)
  - Volatility metrics (ATR, Yang-Zhang)
  - Custom market signals

### 🤖 Advanced Models
1. **Enhanced LSTM**:
   - Multi-head attention
   - Feature-specific processing
   - Uncertainty estimation
   
2. **GARCH Model**:
   - Volatility forecasting
   - Risk assessment
   
3. **LightGBM & Random Forest**:
   - High-speed training
   - Feature importance
   
4. **SVR with SHAP**:
   - Dynamic feature importance
   - Kernel-based learning
   
5. **Experimental Models**:
   - Temporal Fusion Transformer
   - RL Trading Agent

## Quick Start

### 1. Clone Repository
```bash
git clone clean-architecture
cd btc-predictor-app
```

### 2. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

pip install -r requirements_simple.txt
```

### 3. Run System
```bash
# For VPS (port 80)
sudo python simple_predictor.py

# For local testing (port 5000)
python simple_predictor.py
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/health` | System health check |
| `/api/current-price` | Current BTC price |
| `/api/latest-prediction` | Most recent prediction |
| `/api/prediction-history` | Last 20 predictions |
| `/api/historical-data` | Price history for charts |
| `/api/system-status` | System status and metrics |

## Data Files
- `data/predictions.csv` - All predictions with accuracy
- Price history stored in memory
- Automatic CSV export every prediction

## System Requirements
- Python 3.8+
- Internet connection for Binance API
- 1GB RAM minimum
- Port 80 access (for VPS deployment)

## Configuration

The system requires setup of various components:

### Data Sources
- **Price Data:** 
  - Primary: Binance (BTC/USDT)
  - Secondary: Correlated pairs (ETH/USDT, BNB/USDT)
  - Fallback: CoinGecko API

### API Keys
Required in `.env` file:
```env
NEWS_API_KEY=your_key_here
TWITTER_API_KEY=your_key_here
TWITTER_API_SECRET=your_key_here
REDDIT_CLIENT_ID=your_id_here
REDDIT_CLIENT_SECRET=your_secret_here
```

### Model Settings
```python
# In config/config.py
ENSEMBLE_WEIGHTS = {
    'lstm': 0.30,
    'garch': 0.15,
    'svr': 0.15,
    'random_forest': 0.20,
    'lightgbm': 0.20
}

# Feature Groups
FEATURE_GROUPS = {
    'price': ['open', 'high', 'low', 'close', 'volume'],
    'technical': TECHNICAL_INDICATORS,
    'market_depth': ['bid_sum', 'ask_sum', 'imbalance'],
    'derivatives': ['funding_rate', 'open_interest'],
    'sentiment': ['sentiment_score', 'sentiment_magnitude']
}
```

## Deployment on VPS
```bash
# Install system dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Clone and setup
git clone <your-repo>
cd btc-predictor-app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_simple.txt

# Run with sudo for port 80
sudo python simple_predictor.py
```

## Expected Output
```
🚀 SIMPLE BTC PREDICTOR STARTING 🚀
✅ API Server: http://0.0.0.0:80
✅ Data Collection: Every 5 minutes
✅ Predictions: Every 5 minutes

[14:30:15] BTC Price: $69,425.75
Model trained with 45 samples
Prediction: $69,480.25 (Current: $69,425.75)
```

## Troubleshooting
- **Port 80 permission denied:** Use `sudo python simple_predictor.py`
- **API connection failed:** Check firewall settings
- **No predictions:** Wait 15 minutes for initial model training

## License
MIT License - Free to use and modify