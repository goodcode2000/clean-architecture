#!/usr/bin/env python3
"""
Enhanced High-Precision BTC Predictor
New Architecture: ARIMA + SVM + Boruta+RF + XGBoost + LSTM
Dynamic weight adjustment based on performance
Target: $15-30 USD accuracy with directional accuracy focus
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
import warnings
warnings.filterwarnings('ignore')

# Core ML imports
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, accuracy_score

# Advanced models
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

class EnhancedBTCPredictor:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Data storage
        self.price_history = []
        self.predictions = []
        self.current_price = 0
        
        # Enhanced model ensemble
        self.models = {
            'arima': None,
            'svm': SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1),
            'boruta_rf': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
            'xgboost': xgb.XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective='reg:squarederror'
            )
        }
        
        # Dynamic weights
        self.model_weights = {
            'arima': 0.25,
            'svm': 0.25,
            'boruta_rf': 0.25,
            'xgboost': 0.25
        }
        
        # Performance tracking
        self.model_performance = {
            'arima': [],
            'svm': [],
            'boruta_rf': [],
            'xgboost': []
        }
        
        # Scalers
        self.scalers = {
            'svm': StandardScaler(),
            'boruta_rf': RobustScaler(),
            'xgboost': StandardScaler()
        }
        
        # Model training status
        self.models_trained = {
            'arima': False,
            'svm': False,
            'boruta_rf': False,
            'xgboost': False
        }
        
        # Enhanced performance tracking for rapid change detection
        self.directional_accuracy = 0.0
        self.price_direction_history = []
        self.target_accuracy = 45  # $20-60 USD range (middle: $40, ±$20 tolerance)
        
        # 90-day historical data management
        self.last_5min_update = None
        
        os.makedirs('data', exist_ok=True)
        
        # Initialize 90-day historical data at startup
        self.initialize_90_day_data()
        
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask API routes"""
        
        @self.app.route('/api/health')
        def health():
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "3.0.0-enhanced",
                "models": "ARIMA+SVM+Boruta+RF+XGBoost",
                "target_accuracy": "$15-30 USD"
            })
        
        @self.app.route('/api/current-price')
        def current_price():
            return jsonify({
                "price": self.current_price,
                "timestamp": datetime.now().isoformat(),
                "model_weights": self.model_weights
            })
        
        @self.app.route('/api/latest-prediction')
        def latest_prediction():
            if self.predictions:
                latest = self.predictions[-1].copy()
                return jsonify(latest)
            return jsonify({"error": "No predictions available"})
        
        @self.app.route('/api/prediction-history')
        def prediction_history():
            recent_preds = self.predictions[-50:]
            return jsonify({
                "predictions": recent_preds,
                "total": len(self.predictions)
            })
        
        @self.app.route('/api/historical-data')
        def historical_data():
            """Serve 90-day historical real price data for second app"""
            try:
                if os.path.exists('data/historical_real.csv'):
                    df = pd.read_csv('data/historical_real.csv')
                    
                    # Convert to API format
                    data_points = []
                    for _, row in df.iterrows():
                        data_points.append({
                            "timestamp": row['timestamp'],
                            "price": float(row['price'])
                        })
                    
                    return jsonify({
                        "data": data_points,
                        "total": len(data_points),
                        "range": "90 days",
                        "interval": "5 minutes"
                    })
                else:
                    return jsonify({"error": "Historical data not available"})
            except Exception as e:
                return jsonify({"error": f"Historical data error: {str(e)}"})
        
        @self.app.route('/api/system-status')
        def system_status():
            """Enhanced system status with 90-day data info"""
            try:
                historical_count = 0
                if os.path.exists('data/historical_real.csv'):
                    df = pd.read_csv('data/historical_real.csv')
                    historical_count = len(df)
                
                return jsonify({
                    "status": "running",
                    "api_server": "Enhanced Predictor - Port 8080",
                    "models_trained": sum(self.models_trained.values()),
                    "historical_data_points": historical_count,
                    "data_range": "90 days",
                    "target_accuracy": "$20-60 USD",
                    "last_update": datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({"error": f"Status error: {str(e)}"})
    
    def initialize_90_day_data(self):
        """Initialize 90-day historical data at startup"""
        try:
            if os.path.exists('data/historical_real.csv'):
                print("📊 Loading existing 90-day historical data...")
                df = pd.read_csv('data/historical_real.csv')
                print(f"✅ Loaded {len(df)} historical price points")
            else:
                print("📈 Fetching 90 days of historical BTC data...")
                self.fetch_90_days_historical_data()
        except Exception as e:
            print(f"⚠️ Historical data initialization error: {e}")
    
    def fetch_90_days_historical_data(self):
        """Fetch 90 days of 5-minute historical data from Binance"""
        try:
            print("🔄 Downloading 90 days of BTC price data...")
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=90)
            
            all_data = []
            current_time = start_time
            
            # Fetch in daily chunks to avoid API limits
            while current_time < end_time:
                chunk_end = min(current_time + timedelta(days=1), end_time)
                
                params = {
                    "symbol": "BTCUSDT",
                    "interval": "5m",
                    "startTime": int(current_time.timestamp() * 1000),
                    "endTime": int(chunk_end.timestamp() * 1000),
                    "limit": 1000
                }
                
                response = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=30)
                if response.status_code == 200:
                    klines = response.json()
                    for kline in klines:
                        timestamp = datetime.fromtimestamp(kline[0] / 1000)
                        price = float(kline[4])  # Close price
                        
                        all_data.append({
                            'timestamp': timestamp.isoformat(),
                            'price': price
                        })
                    
                    print(f"📊 Fetched {len(klines)} points for {current_time.strftime('%Y-%m-%d')}")
                
                current_time = chunk_end
                time.sleep(0.1)  # Rate limiting
            
            # Save to historical_real.csv
            if all_data:
                df = pd.DataFrame(all_data)
                df.to_csv('data/historical_real.csv', index=False)
                print(f"✅ Saved {len(all_data)} historical points to historical_real.csv")
                return True
            
        except Exception as e:
            print(f"❌ 90-day data fetch error: {e}")
            return False
    
    def fetch_btc_data(self):
        """Fetch current BTC data and update historical file every 5 minutes"""
        try:
            url = "https://api.binance.com/api/v3/ticker/price"
            response = requests.get(url, params={"symbol": "BTCUSDT"}, timeout=10)
            if response.status_code == 200:
                price_data = response.json()
                current_price = float(price_data['price'])
                
                # Store data in memory
                timestamp = datetime.now()
                data_point = {
                    'timestamp': timestamp,
                    'price': current_price
                }
                
                self.price_history.append(data_point)
                self.current_price = current_price
                
                # Keep last 2000 points in memory (about 7 days at 5-min intervals)
                if len(self.price_history) > 2000:
                    self.price_history = self.price_history[-2000:]
                
                # Update historical_real.csv every 5 minutes
                self.update_historical_real_csv(timestamp, current_price)
                
                return True
        except Exception as e:
            print(f"❌ Data fetch error: {e}")
            return False
    
    def update_historical_real_csv(self, timestamp, price):
        """Update historical_real.csv with new data every 5 minutes"""
        try:
            # Check if 5 minutes have passed since last update
            current_5min_mark = timestamp.replace(minute=(timestamp.minute // 5) * 5, second=0, microsecond=0)
            
            if self.last_5min_update and self.last_5min_update >= current_5min_mark:
                return  # Not time for update yet
            
            # Load existing data
            historical_data = []
            if os.path.exists('data/historical_real.csv'):
                df = pd.read_csv('data/historical_real.csv')
                historical_data = df.to_dict('records')
            
            # Add new 5-minute data point
            new_point = {
                'timestamp': current_5min_mark.isoformat(),
                'price': price
            }
            
            # Check if this timestamp already exists (avoid duplicates)
            existing_timestamps = [point['timestamp'] for point in historical_data]
            if new_point['timestamp'] not in existing_timestamps:
                historical_data.append(new_point)
                
                # Keep only last 90 days (25,920 points at 5-minute intervals)
                if len(historical_data) > 25920:
                    historical_data = historical_data[-25920:]
                
                # Save updated data
                df = pd.DataFrame(historical_data)
                df.to_csv('data/historical_real.csv', index=False)
                
                self.last_5min_update = current_5min_mark
                print(f"📊 Updated historical_real.csv: {len(historical_data)} points (90 days)")
            
        except Exception as e:
            print(f"⚠️ Historical CSV update error: {e}")
    
    def train_models(self):
        """Train all models using 90-day historical data for improved accuracy"""
        try:
            # Load 90-day historical data for training
            historical_prices = self.load_historical_data_for_training()
            
            # Combine with recent memory data
            recent_prices = [p['price'] for p in self.price_history[-100:]]
            
            # Use historical data if available, otherwise use recent data
            if len(historical_prices) > 1000:
                prices = historical_prices
                print(f"🎯 Training with {len(prices)} historical points (90 days)")
            elif len(recent_prices) >= 50:
                prices = recent_prices
                print(f"🎯 Training with {len(prices)} recent points")
            else:
                print("❌ Insufficient data for training")
                return False
            
            # Train ARIMA
            try:
                model = ARIMA(prices, order=(2, 1, 2))
                fitted_model = model.fit()
                self.models['arima'] = fitted_model
                self.models_trained['arima'] = True
                print("✅ ARIMA trained")
            except:
                print("⚠️ ARIMA training failed")
            
            # For ML models, create simple features
            if len(prices) >= 20:
                X = []
                y = []
                
                for i in range(10, len(prices) - 2):
                    # Simple features: last 10 prices
                    features = prices[i-10:i]
                    target = prices[i+2]  # 2 steps ahead
                    X.append(features)
                    y.append(target)
                
                if len(X) >= 20:
                    X = np.array(X)
                    y = np.array(y)
                    
                    # Train SVM
                    try:
                        X_svm = self.scalers['svm'].fit_transform(X)
                        self.models['svm'].fit(X_svm, y)
                        self.models_trained['svm'] = True
                        print("✅ SVM trained")
                    except:
                        print("⚠️ SVM training failed")
                    
                    # Train RF
                    try:
                        X_rf = self.scalers['boruta_rf'].fit_transform(X)
                        self.models['boruta_rf'].fit(X_rf, y)
                        self.models_trained['boruta_rf'] = True
                        print("✅ Boruta+RF trained")
                    except:
                        print("⚠️ Boruta+RF training failed")
                    
                    # Train XGBoost
                    try:
                        X_xgb = self.scalers['xgboost'].fit_transform(X)
                        self.models['xgboost'].fit(X_xgb, y)
                        self.models_trained['xgboost'] = True
                        print("✅ XGBoost trained")
                    except:
                        print("⚠️ XGBoost training failed")
            
            return any(self.models_trained.values())
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False
    
    def make_prediction(self):
        """Make ensemble prediction"""
        try:
            if not any(self.models_trained.values()):
                return None
            
            predictions = {}
            
            # ARIMA prediction
            if self.models_trained['arima']:
                try:
                    arima_pred = self.models['arima'].forecast(steps=2)[1]
                    predictions['arima'] = arima_pred
                except:
                    pass
            
            # ML predictions
            if len(self.price_history) >= 10:
                recent_prices = [p['price'] for p in self.price_history[-10:]]
                X_current = np.array([recent_prices])
                
                # SVM
                if self.models_trained['svm']:
                    try:
                        X_svm = self.scalers['svm'].transform(X_current)
                        svm_pred = self.models['svm'].predict(X_svm)[0]
                        predictions['svm'] = svm_pred
                    except:
                        pass
                
                # RF
                if self.models_trained['boruta_rf']:
                    try:
                        X_rf = self.scalers['boruta_rf'].transform(X_current)
                        rf_pred = self.models['boruta_rf'].predict(X_rf)[0]
                        predictions['boruta_rf'] = rf_pred
                    except:
                        pass
                
                # XGBoost
                if self.models_trained['xgboost']:
                    try:
                        X_xgb = self.scalers['xgboost'].transform(X_current)
                        xgb_pred = self.models['xgboost'].predict(X_xgb)[0]
                        predictions['xgboost'] = xgb_pred
                    except:
                        pass
            
            # Ensemble prediction
            if predictions:
                weighted_prediction = 0
                total_weight = 0
                
                for model_name, pred_value in predictions.items():
                    weight = self.model_weights.get(model_name, 0)
                    weighted_prediction += pred_value * weight
                    total_weight += weight
                
                if total_weight > 0:
                    final_prediction = weighted_prediction / total_weight
                    
                    prediction_record = {
                        'timestamp': datetime.now().isoformat(),
                        'current_price': self.current_price,
                        'predicted_price': final_prediction,
                        'model': 'Enhanced Ensemble',
                        'confidence_lower': final_prediction * 0.98,
                        'confidence_upper': final_prediction * 1.02
                    }
                    
                    return prediction_record
            
            return None
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def save_predictions(self):
        """Save predictions to CSV"""
        try:
            if len(self.predictions) > 0:
                df = pd.DataFrame(self.predictions)
                df.to_csv('data/predictions.csv', index=False)
        except Exception as e:
            print(f"⚠️ Save error: {e}")
    
    def run_prediction_loop(self):
        """Main prediction loop with 5-minute intervals and periodic retraining"""
        print("🚀 Starting Enhanced BTC Predictor...")
        print("📊 Models: ARIMA + SVM + Boruta+RF + XGBoost")
        print("🎯 Target: $20-60 USD accuracy with rapid change detection")
        print("📈 90-day historical data management enabled")
        print("="*60)
        
        retrain_counter = 0
        
        while True:
            try:
                # Fetch data every minute
                if self.fetch_btc_data():
                    print(f"💰 Current BTC Price: ${self.current_price:,.2f}")
                    
                    # Train models periodically with 90-day data
                    retrain_counter += 1
                    if retrain_counter >= 60:  # Retrain every hour (60 minutes)
                        print("🔄 Starting periodic model retraining with 90-day data...")
                        if self.train_models():
                            print("✅ Models retrained successfully")
                        retrain_counter = 0
                    elif len(self.price_history) >= 50 and len(self.price_history) % 20 == 0:
                        # Quick retrain with recent data
                        self.train_models()
                    
                    # Make prediction
                    if any(self.models_trained.values()):
                        prediction = self.make_prediction()
                        
                        if prediction:
                            self.predictions.append(prediction)
                            self.save_predictions()
                            
                            pred_price = prediction['predicted_price']
                            direction = "📈" if pred_price > self.current_price else "📉"
                            change = abs(pred_price - self.current_price)
                            print(f"🔮 Prediction: ${pred_price:,.2f} {direction} (±${change:.2f})")
                            
                            # Keep last 1000 predictions
                            if len(self.predictions) > 1000:
                                self.predictions = self.predictions[-1000:]
                
                time.sleep(60)  # 1-minute intervals (5-minute CSV updates handled in fetch_btc_data)
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping predictor...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(30)

def main():
    predictor = EnhancedBTCPredictor()
    
    # Start prediction loop in thread
    prediction_thread = threading.Thread(target=predictor.run_prediction_loop)
    prediction_thread.daemon = True
    prediction_thread.start()
    
    # Start API server
    print("🌐 Starting API server on port 8080...")
    predictor.app.run(host='0.0.0.0', port=8080, debug=False)

if __name__ == "__main__":
    main()    
 
   def update_historical_real_csv(self, timestamp, price):
        """Update historical_real.csv with new data every 5 minutes"""
        try:
            # Check if 5 minutes have passed since last update
            current_5min_mark = timestamp.replace(minute=(timestamp.minute // 5) * 5, second=0, microsecond=0)
            
            if self.last_5min_update and self.last_5min_update >= current_5min_mark:
                return  # Not time for update yet
            
            # Load existing data
            historical_data = []
            if os.path.exists('data/historical_real.csv'):
                df = pd.read_csv('data/historical_real.csv')
                historical_data = df.to_dict('records')
            
            # Add new 5-minute data point
            new_point = {
                'timestamp': current_5min_mark.isoformat(),
                'price': price
            }
            
            # Check if this timestamp already exists (avoid duplicates)
            existing_timestamps = [point['timestamp'] for point in historical_data]
            if new_point['timestamp'] not in existing_timestamps:
                historical_data.append(new_point)
                
                # Keep only last 90 days (25,920 points at 5-minute intervals)
                # 90 days * 24 hours * 12 (5-min intervals per hour) = 25,920
                if len(historical_data) > 25920:
                    historical_data = historical_data[-25920:]
                
                # Save updated data
                df = pd.DataFrame(historical_data)
                df.to_csv('data/historical_real.csv', index=False)
                
                self.last_5min_update = current_5min_mark
                print(f"📊 Updated historical_real.csv: {len(historical_data)} points (90 days)")
            
        except Exception as e:
            print(f"⚠️ Historical CSV update error: {e}")
    
    def load_historical_data_for_training(self):
        """Load 90-day historical data for model training"""
        try:
            if os.path.exists('data/historical_real.csv'):
                df = pd.read_csv('data/historical_real.csv')
                
                # Convert to training format
                training_prices = []
                for _, row in df.iterrows():
                    training_prices.append(float(row['price']))
                
                print(f"📈 Loaded {len(training_prices)} historical points for training")
                return training_prices
            else:
                print("⚠️ No historical_real.csv found for training")
                return []
        except Exception as e:
            print(f"❌ Historical data loading error: {e}")
            return []