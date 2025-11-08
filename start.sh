#!/bin/bash
# BTC Predictor Startup Script

echo "🚀 Starting BTC Price Predictor System"
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements_simple.txt

# Create data directory
mkdir -p data
mkdir -p logs

# Start the system
echo "Starting BTC Predictor..."
echo "Press Ctrl+C to stop"
echo "======================================"

# Run with sudo for port 80 (VPS)
if [ "$1" = "vps" ]; then
    sudo python simple_predictor.py
else
    # Local development (port 5000)
    python simple_predictor.py
fi