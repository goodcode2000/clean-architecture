# BTC Price Predictor - Standalone System

## 🚀 Complete BTC Price Prediction System
Real-time Bitcoin price prediction with ML models and REST API.

## Features
- ✅ **Real-time BTC data** from Binance API
- ✅ **ML predictions** every 5 minutes  
- ✅ **REST API** for data access
- ✅ **Automatic model training** and retraining
- ✅ **Prediction accuracy tracking**
- ✅ **CSV data export**

## Quick Start

### 1. Clone Repository
```bash
git clone <your-repo-url>
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
The system works out-of-the-box with default settings:
- **Data Source:** Binance BTCUSDT
- **Prediction Interval:** 5 minutes
- **Model:** RandomForest with auto-retraining
- **API Port:** 80 (VPS) or 5000 (local)

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