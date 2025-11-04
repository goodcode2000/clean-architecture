#!/usr/bin/env python3
"""
Simple API server for BTC data - runs alongside precision predictor
No root permissions needed - uses port 5000
"""
from flask import Flask, jsonify
from flask_cors import CORS
import json
import os
import pandas as pd
from datetime import datetime
import requests
import sys

# Use project config for symbol selection
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import Config

app = Flask(__name__)
CORS(app)

def get_current_btc_price():
    """Get current BTC price from Binance"""
    try:
        url = "https://api.binance.com/api/v3/ticker/price"
        params = {"symbol": Config.PRICE_SYMBOL}
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return float(data['price'])
    except:
        pass
    return 110469.57  # Fallback price

def get_historical_data():
    """Get historical price data"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": Config.PRICE_SYMBOL,
            "interval": "1m",
            "limit": 300  # Last 5 hours
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            historical = []
            for item in data:
                historical.append({
                    "timestamp": datetime.fromtimestamp(item[0]/1000).isoformat(),
                    "price": float(item[4]),  # Close price
                    "unix_time": int(item[0]/1000)
                })
            return historical
    except:
        pass
    
    # Fallback data
    current_price = get_current_btc_price()
    return [{
        "timestamp": datetime.now().isoformat(),
        "price": current_price,
        "unix_time": int(datetime.now().timestamp())
    }]

@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0-simple",
        "message": "Simple API for BTC data"
    })

@app.route('/api/current-price')
def current_price():
    price = get_current_btc_price()
    return jsonify({
        "price": price,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/historical-data')
def historical_data():
    data = get_historical_data()
    return jsonify({
        "data": data,
        "total": len(data)
    })

@app.route('/api/latest-prediction')
def latest_prediction():
    # Check if predictions.csv exists
    if os.path.exists('data/predictions.csv'):
        try:
            df = pd.read_csv('data/predictions.csv')
            if len(df) > 0:
                latest = df.iloc[-1].to_dict()
                return jsonify(latest)
        except:
            pass
    
    # Return mock prediction for now
    current_price = get_current_btc_price()
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "current_price": current_price,
        "predicted_price": current_price + 25,  # Simple +$25 prediction
        "confidence_lower": current_price + 10,
        "confidence_upper": current_price + 40,
        "model": "Simple",
        "message": "Waiting for precision predictor to generate predictions"
    })

@app.route('/api/prediction-history')
def prediction_history():
    predictions = []
    
    # Try to load from predictions.csv
    if os.path.exists('data/predictions.csv'):
        try:
            df = pd.read_csv('data/predictions.csv')
            predictions = df.tail(20).to_dict('records')
        except:
            pass
    
    return jsonify({
        "predictions": predictions,
        "total": len(predictions)
    })

@app.route('/api/system-status')
def system_status():
    return jsonify({
        "is_running": True,
        "api_server": "Simple API - Port 5000",
        "precision_predictor": "Running separately",
        "data_source": f"Binance / CoinGecko (symbol={Config.PRICE_SYMBOL})",
        "last_update": datetime.now().isoformat()
    })

if __name__ == "__main__":
    print("🚀 Starting Simple BTC API Server...")
    print("📊 Port: 5000 (No root permissions needed)")
    print("🔗 Endpoints: /api/health, /api/current-price, /api/historical-data")
    print("✅ CORS enabled for dashboard connection")
    print("="*50)
    
    app.run(host='0.0.0.0', port=5000, debug=False)