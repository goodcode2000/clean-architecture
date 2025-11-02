#!/usr/bin/env python3
"""
High-Precision BTC Predictor - Target: $20-50 USD accuracy
Includes rapid change detection and ultra-short-term predictions
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class PrecisionBTCPredictor:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # High-frequency data storage (1-minute intervals)
        self.price_history = []
        self.predictions = []
        self.current_price = 0
        
        # Multiple models for different conditions
        self.models = {
            'stable': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42),
            'volatile': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            'rapid': GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
        }
        
        self.scalers = {
            'stable': RobustScaler(),
            'volatile': StandardScaler(), 
            'rapid': RobustScaler()
        }
        
        self.models_trained = {'stable': False, 'volatile': False, 'rapid': False}
        
        # Market condition detection
        self.market_state = 'stable'
        self.volatility_threshold = 0.002  # 0.2% for rapid detection
        self.rapid_change_threshold = 0.001  # 0.1% for ultra-sensitive detection
        
        # Accuracy tracking
        self.target_accuracy = 50  # $50 USD max error
        self.recent_errors = []
        
        os.makedirs('data', exist_ok=True)
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask API routes"""
        
        @self.app.route('/api/health')
        def health():
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0-precision",
                "target_accuracy": "$20-50 USD"
            })
        
        @self.app.route('/api/current-price')
        def current_price():
            return jsonify({
                "price": self.current_price,
                "timestamp": datetime.now().isoformat(),
                "market_state": self.market_state
            })
        
        @self.app.route('/api/latest-prediction')
        def latest_prediction():
            if self.predictions:
                latest = self.predictions[-1].copy()
                latest['target_accuracy'] = "$20-50 USD"
                latest['market_state'] = self.market_state
                return jsonify(latest)
            return jsonify({"error": "No predictions available"})
        
        @self.app.route('/api/prediction-history')
        def prediction_history():
            recent_preds = self.predictions[-20:]
            accurate_preds = [p for p in recent_preds if 'error' in p and p['error'] <= 50]
            
            return jsonify({
                "predictions": recent_preds,
                "total": len(self.predictions),
                "accurate_predictions": len(accurate_preds),
                "accuracy_rate": f"{(len(accurate_preds)/len(recent_preds)*100):.1f}%" if recent_preds else "0%"
            })
        
        @self.app.route('/api/historical-data')
        def historical_data():
            # Return last 300 data points (5 hours of 1-minute data)
            recent_data = self.price_history[-300:] if len(self.price_history) >= 300 else self.price_history
            return jsonify({
                "data": recent_data,
                "total": len(recent_data),
                "interval": "1-minute"
            })
        
        @self.app.route('/api/system-status')
        def system_status():
            avg_error = np.mean(self.recent_errors) if self.recent_errors else 0
            return jsonify({
                "is_running": True,
                "models_trained": self.models_trained,
                "data_points": len(self.price_history),
                "predictions_made": len(self.predictions),
                "market_state": self.market_state,
                "average_error_usd": round(avg_error, 2),
                "target_met": avg_error <= 50 if self.recent_errors else False,
                "last_update": datetime.now().isoformat()
            })
    
    def fetch_btc_price_high_freq(self):
        """Fetch BTC price every minute for high precision"""
        try:
            # Get detailed price data
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "limit": 1
            }
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()[0]
                
                price_data = {
                    "timestamp": datetime.now().isoformat(),
                    "open": float(data[1]),
                    "high": float(data[2]),
                    "low": float(data[3]),
                    "close": float(data[4]),
                    "volume": float(data[5]),
                    "unix_time": int(datetime.now().timestamp())
                }
                
                self.price_history.append(price_data)
                self.current_price = price_data['close']
                
                # Keep only last 1000 data points (about 16 hours)
                if len(self.price_history) > 1000:
                    self.price_history = self.price_history[-1000:]
                
                # Detect market conditions
                self.detect_market_conditions()
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] BTC: ${self.current_price:,.2f} | State: {self.market_state}")
                return True
            else:
                print(f"Error fetching price: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error fetching BTC price: {e}")
            return False
    
    def detect_market_conditions(self):
        """Detect current market state for model selection"""
        if len(self.price_history) < 10:
            return
        
        # Get recent prices
        recent_prices = [p['close'] for p in self.price_history[-10:]]
        
        # Calculate short-term volatility
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns)
        
        # Calculate rapid changes
        latest_change = abs(returns[-1]) if len(returns) > 0 else 0
        
        # Determine market state
        if latest_change > self.rapid_change_threshold:
            self.market_state = 'rapid'
        elif volatility > self.volatility_threshold:
            self.market_state = 'volatile'
        else:
            self.market_state = 'stable'
    
    def create_precision_features(self, data_window):
        """Create high-precision features for ultra-accurate predictions"""
        if len(data_window) < 30:
            return None
        
        # Extract OHLCV data
        opens = np.array([d['open'] for d in data_window])
        highs = np.array([d['high'] for d in data_window])
        lows = np.array([d['low'] for d in data_window])
        closes = np.array([d['close'] for d in data_window])
        volumes = np.array([d['volume'] for d in data_window])
        
        features = []
        
        # Current OHLC
        features.extend([opens[-1], highs[-1], lows[-1], closes[-1]])
        
        # Micro price movements (last few minutes)
        for i in [1, 2, 3, 5]:
            if len(closes) > i:
                change = (closes[-1] - closes[-1-i]) / closes[-1-i]
                features.append(change)
            else:
                features.append(0)
        
        # Ultra-short moving averages
        for window in [3, 5, 10, 15, 20]:
            if len(closes) >= window:
                ma = np.mean(closes[-window:])
                features.append(ma)
                features.append((closes[-1] - ma) / ma)
            else:
                features.append(closes[-1])
                features.append(0)
        
        # Price action patterns
        if len(closes) >= 5:
            # Recent high/low
            recent_high = np.max(highs[-5:])
            recent_low = np.min(lows[-5:])
            features.append((closes[-1] - recent_low) / (recent_high - recent_low))
            
            # Momentum indicators
            momentum_3 = closes[-1] - closes[-4] if len(closes) >= 4 else 0
            momentum_5 = closes[-1] - closes[-6] if len(closes) >= 6 else 0
            features.extend([momentum_3, momentum_5])
        else:
            features.extend([0.5, 0, 0])
        
        # Volume analysis
        if len(volumes) >= 10:
            vol_ma = np.mean(volumes[-10:])
            vol_ratio = volumes[-1] / vol_ma if vol_ma > 0 else 1
            features.append(vol_ratio)
            
            # Volume-price correlation
            vol_change = (volumes[-1] - volumes[-2]) / volumes[-2] if len(volumes) >= 2 and volumes[-2] > 0 else 0
            price_change = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0
            features.append(vol_change * price_change)
        else:
            features.extend([1, 0])
        
        # Volatility measures
        for window in [5, 10]:
            if len(closes) >= window:
                returns = np.diff(closes[-window:]) / closes[-window:-1]
                vol = np.std(returns)
                features.append(vol)
            else:
                features.append(0)
        
        # Intrabar analysis (high-low spread)
        if len(highs) >= 3:
            spreads = (highs[-3:] - lows[-3:]) / closes[-3:]
            avg_spread = np.mean(spreads)
            current_spread = (highs[-1] - lows[-1]) / closes[-1]
            features.extend([avg_spread, current_spread])
        else:
            features.extend([0, 0])
        
        # Time-based features (minute of hour, etc.)
        now = datetime.now()
        features.extend([
            now.minute / 60,  # Minute of hour (normalized)
            now.hour / 24,    # Hour of day (normalized)
            now.weekday() / 7  # Day of week (normalized)
        ])
        
        return np.array(features).reshape(1, -1)
    
    def train_precision_models(self):
        """Train models for different market conditions"""
        if len(self.price_history) < 100:
            print("Need at least 100 data points for precision training")
            return False
        
        try:
            print("Training precision models...")
            
            # Prepare training data for each market condition
            for condition in ['stable', 'volatile', 'rapid']:
                X_train = []
                y_train = []
                
                # Create training samples
                for i in range(30, len(self.price_history) - 1):
                    window = self.price_history[:i+1]
                    features = self.create_precision_features(window)
                    
                    if features is not None:
                        # Determine if this sample matches the condition
                        current_data = self.price_history[i]
                        next_data = self.price_history[i+1]
                        
                        # Calculate change for condition matching
                        price_change = abs((next_data['close'] - current_data['close']) / current_data['close'])
                        
                        include_sample = False
                        if condition == 'stable' and price_change <= self.rapid_change_threshold:
                            include_sample = True
                        elif condition == 'volatile' and self.rapid_change_threshold < price_change <= self.volatility_threshold:
                            include_sample = True
                        elif condition == 'rapid' and price_change > self.volatility_threshold:
                            include_sample = True
                        
                        if include_sample:
                            X_train.append(features[0])
                            y_train.append(next_data['close'])
                
                if len(X_train) >= 20:  # Need minimum samples
                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    
                    # Scale features
                    X_train_scaled = self.scalers[condition].fit_transform(X_train)
                    
                    # Train model
                    self.models[condition].fit(X_train_scaled, y_train)
                    self.models_trained[condition] = True
                    
                    print(f"✅ {condition.capitalize()} model trained with {len(X_train)} samples")
                else:
                    print(f"⚠️ Not enough {condition} samples: {len(X_train)}")
            
            return any(self.models_trained.values())
            
        except Exception as e:
            print(f"Error training models: {e}")
            return False
    
    def make_precision_prediction(self):
        """Make ultra-precise prediction based on market conditions"""
        if not any(self.models_trained.values()) or len(self.price_history) < 30:
            return None
        
        try:
            # Select appropriate model based on market state
            if not self.models_trained[self.market_state]:
                # Fallback to any trained model
                available_models = [k for k, v in self.models_trained.items() if v]
                if not available_models:
                    return None
                model_type = available_models[0]
            else:
                model_type = self.market_state
            
            # Create features
            features = self.create_precision_features(self.price_history)
            if features is None:
                return None
            
            # Scale features
            features_scaled = self.scalers[model_type].transform(features)
            
            # Make prediction
            predicted_price = self.models[model_type].predict(features_scaled)[0]
            
            # Calculate precision confidence interval (tighter bounds)
            if len(self.recent_errors) > 5:
                recent_mae = np.mean(self.recent_errors[-10:])
                confidence_margin = min(recent_mae * 1.2, 30)  # Cap at $30
            else:
                confidence_margin = 25  # Default $25 margin
            
            confidence_lower = predicted_price - confidence_margin
            confidence_upper = predicted_price + confidence_margin
            
            # Create prediction record
            prediction = {
                "timestamp": datetime.now().isoformat(),
                "current_price": self.current_price,
                "predicted_price": round(predicted_price, 2),
                "confidence_lower": round(confidence_lower, 2),
                "confidence_upper": round(confidence_upper, 2),
                "prediction_for": (datetime.now() + timedelta(minutes=2)).isoformat(),  # 2-minute prediction
                "model_used": model_type,
                "market_state": self.market_state,
                "confidence_margin": round(confidence_margin, 2)
            }
            
            self.predictions.append(prediction)
            
            # Keep only last 200 predictions
            if len(self.predictions) > 200:
                self.predictions = self.predictions[-200:]
            
            # Save predictions
            self.save_predictions()
            
            print(f"🎯 Prediction: ${predicted_price:.2f} (±${confidence_margin:.0f}) | Model: {model_type}")
            return prediction
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def update_precision_accuracy(self):
        """Update predictions with actual prices and track precision"""
        current_time = datetime.now()
        
        for prediction in self.predictions:
            if 'actual_price' not in prediction:
                pred_time = datetime.fromisoformat(prediction['timestamp'].replace('Z', '+00:00').replace('+00:00', ''))
                
                # Check if 2 minutes have passed
                if current_time - pred_time >= timedelta(minutes=2):
                    prediction['actual_price'] = self.current_price
                    error = abs(prediction['predicted_price'] - self.current_price)
                    
                    prediction['error'] = round(error, 2)
                    prediction['error_percentage'] = round((error / self.current_price) * 100, 4)
                    prediction['target_met'] = error <= 50
                    
                    # Track recent errors
                    self.recent_errors.append(error)
                    if len(self.recent_errors) > 50:
                        self.recent_errors = self.recent_errors[-50:]
                    
                    status = "✅ TARGET MET" if error <= 50 else "❌ TARGET MISSED"
                    print(f"📊 Accuracy: ${error:.2f} USD error | {status}")
    
    def save_predictions(self):
        """Save predictions to CSV"""
        try:
            if self.predictions:
                df = pd.DataFrame(self.predictions)
                df.to_csv('data/predictions.csv', index=False)
        except Exception as e:
            print(f"Error saving predictions: {e}")
    
    def precision_loop(self):
        """Main high-precision prediction loop"""
        print("🎯 Starting High-Precision BTC Predictor...")
        print("Target Accuracy: $20-50 USD")
        
        minute_counter = 0
        
        while True:
            try:
                # Fetch price every minute
                if self.fetch_btc_price_high_freq():
                    
                    # Train models initially and periodically
                    if not any(self.models_trained.values()) and len(self.price_history) >= 100:
                        self.train_precision_models()
                    elif minute_counter % 60 == 0 and len(self.price_history) >= 100:  # Retrain hourly
                        self.train_precision_models()
                    
                    # Make prediction every 2 minutes
                    if minute_counter % 2 == 0 and any(self.models_trained.values()):
                        self.make_precision_prediction()
                        self.update_precision_accuracy()
                    
                    minute_counter += 1
                
                # Wait 1 minute
                time.sleep(60)
                
            except KeyboardInterrupt:
                print("Stopping precision predictor...")
                break
            except Exception as e:
                print(f"Error in precision loop: {e}")
                time.sleep(30)
    
    def start_api_server(self):
        """Start Flask API server"""
        print("🚀 Starting Precision API server on port 80...")
        self.app.run(host='0.0.0.0', port=80, debug=False, threaded=True)
    
    def run(self):
        """Start the precision prediction system"""
        print("="*70)
        print("🎯 HIGH-PRECISION BTC PREDICTOR STARTING 🎯")
        print("="*70)
        print("🎯 Target Accuracy: $20-50 USD")
        print("⚡ Prediction Interval: 2 minutes")
        print("📊 Data Collection: 1-minute intervals")
        print("🤖 Models: Stable/Volatile/Rapid conditions")
        print("="*70)
        
        # Start API server
        api_thread = threading.Thread(target=self.start_api_server, daemon=True)
        api_thread.start()
        time.sleep(2)
        
        print("✅ API Server: http://0.0.0.0:80")
        print("✅ High-frequency data collection active")
        print("✅ Multi-model precision system ready")
        print("="*70)
        
        # Start precision prediction loop
        self.precision_loop()

if __name__ == "__main__":
    predictor = PrecisionBTCPredictor()
    predictor.run()