#!/bin/bash
# Enhanced BTC Predictor Startup Script

echo "🚀 Starting Enhanced BTC Predictor System"
echo "📊 Models: ARIMA + SVM + Boruta+RF + XGBoost + LSTM"
echo "🎯 Target: $15-30 USD accuracy with directional focus"
echo "="*60

# Create necessary directories
mkdir -p data
mkdir -p models
mkdir -p logs

# Check Python version
python3 --version

# Install/upgrade requirements
echo "📦 Installing requirements..."
pip3 install -r requirements_enhanced.txt

# Start the enhanced predictor
echo "🔥 Starting Enhanced Precision Predictor..."
python3 enhanced_precision_predictor.py

echo "✅ Enhanced BTC Predictor started successfully!"
echo "🌐 API available at: http://localhost:8080/api/"
echo "📊 Health check: http://localhost:8080/api/health"