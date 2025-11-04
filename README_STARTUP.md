# 🚀 BTC Predictor - Quick Start Guide

## Fixed Issues ✅

1. **Port Configuration**: Changed from port 80 to 5000 (no sudo required)
2. **Missing Test File**: Created `test_complete_system.py`
3. **Dependencies**: Created minimal requirements file
4. **Symbol Configuration**: Fixed BTCUSDT symbol consistency
5. **Startup Scripts**: Enhanced with better error handling

## Quick Start Options

### Option 1: Automated Setup (Recommended)
```bash
cd btc-predictor-app
python quick_start.py
```
This will:
- Check Python version
- Create virtual environment
- Install dependencies
- Run tests
- Start the system

### Option 2: Manual Setup
```bash
cd btc-predictor-app

# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate
# OR Activate (Windows)
venv\Scripts\activate

# Install minimal requirements
pip install -r requirements_minimal.txt

# Run tests
python test_complete_system.py

# Start simple version
python simple_predictor.py
```

### Option 3: Using Start Script
```bash
cd btc-predictor-app
chmod +x start.sh
./start.sh          # Local mode (port 5000)
./start.sh vps      # VPS mode (port 80, requires sudo)
```

## System Verification

### 1. Run Tests First
```bash
python test_complete_system.py
```
This tests:
- All imports work
- Configuration is correct
- Data collection works
- Directories are created
- API connectivity

### 2. Check API Endpoints
Once running, test these URLs:
- Health: http://localhost:5000/api/health
- Current Price: http://localhost:5000/api/current-price
- System Status: http://localhost:5000/api/system-status

### 3. Monitor Logs
The system will show:
- Current BTC price updates every 5 minutes
- Prediction results
- Model training status
- API server status

## Troubleshooting

### Common Issues:

**1. Import Errors**
```bash
pip install -r requirements_minimal.txt
```

**2. Port Already in Use**
```bash
# Change port in environment
export API_PORT=5001
python simple_predictor.py
```

**3. Network Issues**
- Check internet connection
- Verify Binance API is accessible
- Try different API endpoints

**4. Permission Issues (VPS)**
```bash
# For port 80 on VPS
sudo python simple_predictor.py
# OR use higher port
export API_PORT=8080
python simple_predictor.py
```

## System Architecture

### Simple Mode (simple_predictor.py)
- ✅ Real BTC price fetching
- ✅ Basic RandomForest prediction
- ✅ 5-minute intervals
- ✅ REST API with 6 endpoints
- ✅ Automatic model retraining

### Full Mode (main.py)
- ✅ Ensemble ML models (5 algorithms)
- ✅ Advanced feature engineering
- ✅ Offset correction system
- ✅ Terminal monitoring
- ✅ Comprehensive logging
- ✅ 9 REST API endpoints

## Next Steps

1. **Start with Simple Mode**: Get familiar with basic functionality
2. **Verify Data Flow**: Check that prices update every 5 minutes
3. **Test Predictions**: Wait for model training and first predictions
4. **Monitor Performance**: Check prediction accuracy over time
5. **Upgrade to Full Mode**: When ready for advanced features

## Files Created/Modified

- ✅ `test_complete_system.py` - System verification
- ✅ `quick_start.py` - Automated setup
- ✅ `requirements_minimal.txt` - Essential packages only
- ✅ `config/config.py` - Fixed port to 5000
- ✅ `simple_predictor.py` - Fixed port configuration
- ✅ `start.sh` - Enhanced startup script

## Success Indicators

When working correctly, you should see:
```
🚀 SIMPLE BTC PREDICTOR STARTING 🚀
✅ API Server: http://0.0.0.0:5000
✅ Data Collection: Every 5 minutes
✅ Predictions: Every 5 minutes
✅ Model: RandomForest with auto-retraining
[HH:MM:SS] BTC Price: $XX,XXX.XX
Prediction: $XX,XXX.XX (Current: $XX,XXX.XX)
```

The system is now ready to start! 🎉