#!/usr/bin/env python3
"""
Simple BTC Price Predictor - Complete Working Version
Fetches real BTC data and makes actual predictions every 5 minutes
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
from flask import Flask, jsonify
from flask_cors import CORS
import threading
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import sys

# Use project config for symbol selection
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import Config

class SimpleBTCPredictor:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Data storage
        self.price_history = []
        self.predictions = []
        self.current_price = 0
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Setup API routes
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask API routes"""
        
        @self.app.route('/api/health')
        def health():
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            })
        
        @self.app.route('/api/current-price')
        def current_price():
            return jsonify({
                "price": self.current_price,
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/api/latest-prediction')
        def latest_prediction():
            if self.predictions:
                return jsonify(self.predictions[-1])
            return jsonify({"error": "No predictions available"})
        
        @self.app.route('/api/prediction-history')
        def prediction_history():
            return jsonify({
                "predictions": self.predictions[-20:],  # Last 20 predictions
                "total": len(self.predictions)
            })
        
        @self.app.route('/api/historical-data')
        def historical_data():
            # Return last 60 data points (5 hours of 5-minute data)
            recent_data = self.price_history[-60:] if len(self.price_history) >= 60 else self.price_history
            return jsonify({
                "data": recent_data,
                "total": len(recent_data)
            })
        
        @self.app.route('/api/system-status')
        def system_status():
            return jsonify({
                "is_running": True,
                "model_trained": self.is_trained,
                "data_points": len(self.price_history),
                "predictions_made": len(self.predictions),
                "last_update": datetime.now().isoformat()
            })
    
    def fetch_btc_price(self):
        """Fetch current BTC price from Binance"""
        try:
            url = "https://api.binance.com/api/v3/ticker/price"
            params = {"symbol": Config.PRICE_SYMBOL}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                price = float(data['price'])
                timestamp = datetime.now()
                
                # Store price data
                price_data = {
                    "timestamp": timestamp.isoformat(),
                    "price": price,
                    "unix_time": int(timestamp.timestamp())
                }
                
                self.price_history.append(price_data)
                self.current_price = price
                
                # Keep only last 500 data points (about 41 hours)
                if len(self.price_history) > 500:
                    self.price_history = self.price_history[-500:]
                
                print(f"[{timestamp.strftime('%H:%M:%S')}] BTC Price: ${price:,.2f}")
                return True
            else:
                print(f"Error fetching price: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error fetching BTC price: {e}")
            return False
    
    def create_features(self, prices):
        """Create simple features for prediction"""
        if len(prices) < 20:
            return None
            
        # Convert to numpy array
        price_array = np.array(prices)
        
        # Create features
        features = []
        
        # Current price
        features.append(price_array[-1])
        
        # Price changes (returns)
        for i in [1, 2, 3, 5, 10]:
            if len(price_array) > i:
                change = (price_array[-1] - price_array[-1-i]) / price_array[-1-i]
                features.append(change)
            else:
                features.append(0)
        
        # Moving averages
        for window in [5, 10, 20]:
            if len(price_array) >= window:
                ma = np.mean(price_array[-window:])
                features.append(ma)
                features.append((price_array[-1] - ma) / ma)  # Distance from MA
            else:
                features.append(price_array[-1])
                features.append(0)
        
        # Volatility (standard deviation)
        for window in [5, 10]:
            if len(price_array) >= window:
                vol = np.std(price_array[-window:])
                features.append(vol)
            else:
                features.append(0)
        
        # Price momentum
        if len(price_array) >= 3:
            momentum = price_array[-1] - price_array[-3]
            features.append(momentum)
        else:
            features.append(0)
        
        return np.array(features).reshape(1, -1)
    
    def train_model(self):
        """Train the prediction model"""
        if len(self.price_history) < 50:
            print("Not enough data to train model (need at least 50 points)")
            return False
        
        try:
            # Extract prices
            prices = [p['price'] for p in self.price_history]
            
            # Create training data
            X_train = []
            y_train = []
            
            # Use sliding window to create training samples
            for i in range(20, len(prices) - 1):  # Need at least 20 points for features
                features = self.create_features(prices[:i+1])
                if features is not None:
                    X_train.append(features[0])
                    y_train.append(prices[i+1])  # Predict next price
            
            if len(X_train) < 10:
                print("Not enough training samples")
                return False
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            
            print(f"Model trained with {len(X_train)} samples")
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def make_prediction(self):
        """Make a price prediction"""
        if not self.is_trained or len(self.price_history) < 20:
            return None
        
        try:
            # Get recent prices
            prices = [p['price'] for p in self.price_history]
            
            # Create features for prediction
            features = self.create_features(prices)
            if features is None:
                return None
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            predicted_price = self.model.predict(features_scaled)[0]
            
            # Calculate confidence interval (simple approach)
            recent_errors = []
            if len(self.predictions) > 10:
                for pred in self.predictions[-10:]:
                    if 'actual_price' in pred:
                        error = abs(pred['predicted_price'] - pred['actual_price'])
                        recent_errors.append(error)
            
            if recent_errors:
                avg_error = np.mean(recent_errors)
                confidence_lower = predicted_price - avg_error
                confidence_upper = predicted_price + avg_error
            else:
                # Default confidence interval (±2%)
                confidence_lower = predicted_price * 0.98
                confidence_upper = predicted_price * 1.02
            
            # Create prediction record
            prediction = {
                "timestamp": datetime.now().isoformat(),
                "current_price": self.current_price,
                "predicted_price": round(predicted_price, 2),
                "confidence_lower": round(confidence_lower, 2),
                "confidence_upper": round(confidence_upper, 2),
                "prediction_for": (datetime.now() + timedelta(minutes=5)).isoformat(),
                "model": "RandomForest"
            }
            
            self.predictions.append(prediction)
            
            # Keep only last 100 predictions
            if len(self.predictions) > 100:
                self.predictions = self.predictions[-100:]
            
            # Save predictions to file
            self.save_predictions()
            
            print(f"Prediction: ${predicted_price:.2f} (Current: ${self.current_price:.2f})")
            return prediction
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def update_prediction_accuracy(self):
        """Update predictions with actual prices for accuracy calculation"""
        current_time = datetime.now()
        
        for prediction in self.predictions:
            if 'actual_price' not in prediction:
                pred_time = datetime.fromisoformat(prediction['timestamp'].replace('Z', '+00:00').replace('+00:00', ''))
                
                # Check if 5 minutes have passed
                if current_time - pred_time >= timedelta(minutes=5):
                    prediction['actual_price'] = self.current_price
                    error = abs(prediction['predicted_price'] - self.current_price)
                    prediction['error'] = round(error, 2)
                    prediction['error_percentage'] = round((error / self.current_price) * 100, 2)
    
    def save_predictions(self):
        """Save predictions to CSV file"""
        try:
            if self.predictions:
                df = pd.DataFrame(self.predictions)
                df.to_csv('data/predictions.csv', index=False)
        except Exception as e:
            print(f"Error saving predictions: {e}")
    
    def data_collection_loop(self):
        """Main data collection and prediction loop"""
        print("Starting BTC data collection and prediction system...")
        
        while True:
            try:
                # Fetch current price
                if self.fetch_btc_price():
                    
                    # Train model if we have enough data and not trained yet
                    if not self.is_trained and len(self.price_history) >= 50:
                        self.train_model()
                    
                    # Retrain model periodically
                    elif len(self.price_history) % 50 == 0:  # Retrain every 50 data points
                        self.train_model()
                    
                    # Make prediction if model is trained
                    if self.is_trained:
                        self.make_prediction()
                        self.update_prediction_accuracy()
                
                # Wait 5 minutes
                print("Waiting 5 minutes for next update...")
                time.sleep(300)  # 5 minutes
                
            except KeyboardInterrupt:
                print("Stopping system...")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def start_api_server(self):
        """Start the Flask API server"""
        # Use port 5000 (can be changed with environment variable)
        port = int(os.getenv('API_PORT', '5000'))
        print(f"Starting API server on port {port}...")
        self.app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    
    def run(self):
        """Start the complete system"""
        print("="*60)
        print("🚀 SIMPLE BTC PREDICTOR STARTING 🚀")
        print("="*60)
        
        # Start API server in separate thread
        api_thread = threading.Thread(target=self.start_api_server, daemon=True)
        api_thread.start()
        
        # Give API server time to start
        time.sleep(2)
        
        port = int(os.getenv('API_PORT', '5000'))
        print(f"✅ API Server: http://0.0.0.0:{port}")
        print("✅ Data Collection: Every 5 minutes")
        print("✅ Predictions: Every 5 minutes")
        print("✅ Model: RandomForest with auto-retraining")
        print("="*60)
        
        # Start main data collection loop
        self.data_collection_loop()

if __name__ == "__main__":
    predictor = SimpleBTCPredictor()
    predictor.run()