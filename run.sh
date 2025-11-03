#!/bin/bash

# Startup script for BTC Prediction App 1
# Run on Ubuntu VPS

echo "Starting BTC Price Prediction App 1..."
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p data models

# Run the application
echo "Starting application..."
python main.py

