#!/usr/bin/env python3
"""
Enhanced High-Precision BTC Predictor - Complete Version
Architecture: ARIMA + SVM + Boruta+RF + XGBoost + LSTM
90-day historical data + Dynamic weight adjustment
Target: $20-60 USD accuracy with rapid change detection
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error

# Advanced models
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

# Deep Learning
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow available - LSTM model enabled")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - LSTM model disabled")

class EnhancedBTCPredictor:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Data storage
        self.price_history = []
        self.predictions = []
        self.current_price = 0
        
        # Enhanced model ensemble with LSTM
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
            ),
            'lstm': None
        }
        
        # Dynamic weights
        if TENSORFLOW_AVAILABLE:
            self.model_weights = {
                'arima': 0.2, 'svm': 0.2, 'boruta_rf': 0.2, 'xgboost': 0.2, 'lstm': 0.2
            }
        else:
            self.model_weights = {
                'arima': 0.25, 'svm': 0.25, 'boruta_rf': 0.25, 'xgboost': 0.25, 'lstm': 0.0
            }
        
        # Performance tracking
        self.model_performance = {
            'arima': [], 'svm': [], 'boruta_rf': [], 'xgboost': [], 'lstm': []
        }
        
        # Scalers
        self.scalers = {
            'svm': StandardScaler(),
            'boruta_rf': RobustScaler(),
            'xgboost': StandardScaler(),
            'lstm': StandardScaler()
        }
        
        # Model training status
        self.models_trained = {
            'arima': False, 'svm': False, 'boruta_rf': False, 'xgboost': False, 'lstm': False
        }
        
        # Performance tracking
        self.directional_accuracy = 0.0
        self.price_direction_history = []
        self.target_accuracy = 45
        self.last_5min_update = None
        
        os.makedirs('data', exist_ok=True)
        self.initialize_90_day_data()
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask API routes"""
        
        @self.app.route('/api/health')
        def health():
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "3.0.0-enhanced-lstm",
                "models": "ARIMA+SVM+Boruta+RF+XGBoost+LSTM",
                "target_accuracy": "$20-60 USD"
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
                return jsonify(self.predictions[-1])
            return jsonify({"error": "No predictions available"})
        
        @self.app.route('/api/prediction-history')
        def prediction_history():
            return jsonify({
                "predictions": self.predictions[-50:],
                "total": len(self.predictions)
            })
        
        @self.app.route('/api/historical-data')
        def historical_data():
            try:
                if os.path.exists('data/historical_real.csv'):
                    df = pd.read_csv('data/historical_real.csv')
                    data_points = [{"timestamp": row['timestamp'], "price": float(row['price'])} 
                                 for _, row in df.iterrows()]
                    return jsonify({
                        "data": data_points,
                        "total": len(data_points),
                        "range": "90 days"
                    })
                return jsonify({"error": "Historical data not available"})
            except Exception as e:
                return jsonify({"error": str(e)})
    
    def initialize_90_day_data(self):
        """Initialize 90-day historical data"""
        try:
            if os.path.exists('data/historical_real.csv'):
                df = pd.read_csv('data/historical_real.csv')
                print(f"📊 Loaded {len(df)} historical price points")
            else:
                print("📈 Fetching 90 days of historical data...")
                self.fetch_90_days_historical_data()
        except Exception as e:
            print(f"⚠️ Historical data error: {e}")
    
    def fetch_90_days_historical_data(self):
        """Fetch 90 days of historical data"""
        try:
            print("🔄 Downloading 90 days of BTC data...")
            end_time = datetime.now()
            start_time = end_time - timedelta(days=90)
            all_data = []
            current_time = start_time
            
            while current_time < end_time:
                chunk_end = min(current_time + timedelta(days=1), end_time)
                params = {
                    "symbol": "BTCUSDT", "interval": "5m",
                    "startTime": int(current_time.timestamp() * 1000),
                    "endTime": int(chunk_end.timestamp() * 1000), "limit": 1000
                }
                
                response = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=30)
                if response.status_code == 200:
                    klines = response.json()
                    for kline in klines:
                        timestamp = datetime.fromtimestamp(kline[0] / 1000)
                        all_data.append({
                            'timestamp': timestamp.isoformat(),
                            'price': float(kline[4])
                        })
                    print(f"📊 Fetched {len(klines)} points for {current_time.strftime('%Y-%m-%d')}")
                
                current_time = chunk_end
                time.sleep(0.1)
            
            if all_data:
                df = pd.DataFrame(all_data)
                df.to_csv('data/historical_real.csv', index=False)
                print(f"✅ Saved {len(all_data)} historical points")
                return True
        except Exception as e:
            print(f"❌ 90-day fetch error: {e}")
            return False    
 
   def fetch_btc_data(self):
        """Fetch current data and update CSV every 5 minutes"""
        try:
            response = requests.get("https://api.binance.com/api/v3/ticker/price", 
                                  params={"symbol": "BTCUSDT"}, timeout=10)
            if response.status_code == 200:
                current_price = float(response.json()['price'])
                timestamp = datetime.now()
                
                self.price_history.append({'timestamp': timestamp, 'price': current_price})
                self.current_price = current_price
                
                if len(self.price_history) > 2000:
                    self.price_history = self.price_history[-2000:]
                
                self.update_historical_real_csv(timestamp, current_price)
                return True
        except Exception as e:
            print(f"❌ Data fetch error: {e}")
            return False
    
    def update_historical_real_csv(self, timestamp, price):
        """Update CSV every 5 minutes"""
        try:
            current_5min_mark = timestamp.replace(minute=(timestamp.minute // 5) * 5, second=0, microsecond=0)
            
            if self.last_5min_update and self.last_5min_update >= current_5min_mark:
                return
            
            historical_data = []
            if os.path.exists('data/historical_real.csv'):
                df = pd.read_csv('data/historical_real.csv')
                historical_data = df.to_dict('records')
            
            new_point = {'timestamp': current_5min_mark.isoformat(), 'price': price}
            existing_timestamps = [point['timestamp'] for point in historical_data]
            
            if new_point['timestamp'] not in existing_timestamps:
                historical_data.append(new_point)
                if len(historical_data) > 25920:  # 90 days
                    historical_data = historical_data[-25920:]
                
                df = pd.DataFrame(historical_data)
                df.to_csv('data/historical_real.csv', index=False)
                self.last_5min_update = current_5min_mark
                print(f"📊 Updated CSV: {len(historical_data)} points")
        except Exception as e:
            print(f"⚠️ CSV update error: {e}")
    
    def load_historical_data_for_training(self):
        """Load historical data for training"""
        try:
            if os.path.exists('data/historical_real.csv'):
                df = pd.read_csv('data/historical_real.csv')
                prices = [float(row['price']) for _, row in df.iterrows()]
                print(f"📈 Loaded {len(prices)} points for training")
                return prices
            return []
        except Exception as e:
            print(f"❌ Load error: {e}")
            return []
    
    def train_lstm_model(self, X, y):
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE or len(X) < 50:
            return False
        
        try:
            X_scaled = self.scalers['lstm'].fit_transform(X)
            X_lstm, y_lstm = [], []
            
            for i in range(10, len(X_scaled)):
                X_lstm.append(X_scaled[i-10:i])
                y_lstm.append(y[i])
            
            if len(X_lstm) < 20:
                return False
            
            X_lstm = np.array(X_lstm)
            y_lstm = np.array(y_lstm)
            
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)
            
            self.models['lstm'] = model
            self.models_trained['lstm'] = True
            print("✅ LSTM trained")
            return True
        except Exception as e:
            print(f"❌ LSTM error: {e}")
            return False
    
    def adjust_model_weights(self):
        """Dynamic weight adjustment"""
        try:
            model_scores = {}
            for model_name, errors in self.model_performance.items():
                if len(errors) >= 5:
                    avg_error = np.mean(errors[-20:])
                    model_scores[model_name] = 1 / (1 + avg_error)
                else:
                    model_scores[model_name] = 0.5
            
            total_score = sum(model_scores.values())
            if total_score > 0:
                for model_name in self.model_weights:
                    if model_name in model_scores:
                        new_weight = model_scores[model_name] / total_score
                        self.model_weights[model_name] = 0.7 * self.model_weights[model_name] + 0.3 * new_weight
                
                # Normalize weights
                total_weight = sum(self.model_weights.values())
                if total_weight > 0:
                    for model_name in self.model_weights:
                        self.model_weights[model_name] /= total_weight
                
                print(f"🎯 Weights: A:{self.model_weights['arima']:.2f} S:{self.model_weights['svm']:.2f} R:{self.model_weights['boruta_rf']:.2f} X:{self.model_weights['xgboost']:.2f} L:{self.model_weights['lstm']:.2f}")
        except Exception as e:
            print(f"⚠️ Weight error: {e}")
    
    def train_models(self):
        """Train all models"""
        try:
            historical_prices = self.load_historical_data_for_training()
            recent_prices = [p['price'] for p in self.price_history[-100:]]
            
            if len(historical_prices) > 1000:
                prices = historical_prices
                print(f"🎯 Training with {len(prices)} historical points")
            elif len(recent_prices) >= 50:
                prices = recent_prices
                print(f"🎯 Training with {len(prices)} recent points")
            else:
                return False
            
            # Train ARIMA
            try:
                model = ARIMA(prices, order=(2, 1, 2))
                self.models['arima'] = model.fit()
                self.models_trained['arima'] = True
                print("✅ ARIMA trained")
            except:
                print("⚠️ ARIMA failed")
            
            # Train ML models
            if len(prices) >= 20:
                X, y = [], []
                for i in range(10, len(prices) - 2):
                    X.append(prices[i-10:i])
                    y.append(prices[i+2])
                
                if len(X) >= 20:
                    X, y = np.array(X), np.array(y)
                    
                    # SVM
                    try:
                        X_svm = self.scalers['svm'].fit_transform(X)
                        self.models['svm'].fit(X_svm, y)
                        self.models_trained['svm'] = True
                        print("✅ SVM trained")
                    except:
                        print("⚠️ SVM failed")
                    
                    # RF
                    try:
                        X_rf = self.scalers['boruta_rf'].fit_transform(X)
                        self.models['boruta_rf'].fit(X_rf, y)
                        self.models_trained['boruta_rf'] = True
                        print("✅ RF trained")
                    except:
                        print("⚠️ RF failed")
                    
                    # XGBoost
                    try:
                        X_xgb = self.scalers['xgboost'].fit_transform(X)
                        self.models['xgboost'].fit(X_xgb, y)
                        self.models_trained['xgboost'] = True
                        print("✅ XGBoost trained")
                    except:
                        print("⚠️ XGBoost failed")
                    
                    # LSTM
                    if TENSORFLOW_AVAILABLE:
                        self.train_lstm_model(X, y)
            
            return any(self.models_trained.values())
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False    
    d
ef make_prediction(self):
        """Make ensemble prediction with LSTM"""
        try:
            if not any(self.models_trained.values()):
                return None
            
            predictions = {}
            
            # ARIMA
            if self.models_trained['arima']:
                try:
                    predictions['arima'] = self.models['arima'].forecast(steps=2)[1]
                except:
                    pass
            
            # ML models
            if len(self.price_history) >= 10:
                recent_prices = [p['price'] for p in self.price_history[-10:]]
                X_current = np.array([recent_prices])
                
                # SVM
                if self.models_trained['svm']:
                    try:
                        X_svm = self.scalers['svm'].transform(X_current)
                        predictions['svm'] = self.models['svm'].predict(X_svm)[0]
                    except:
                        pass
                
                # RF
                if self.models_trained['boruta_rf']:
                    try:
                        X_rf = self.scalers['boruta_rf'].transform(X_current)
                        predictions['boruta_rf'] = self.models['boruta_rf'].predict(X_rf)[0]
                    except:
                        pass
                
                # XGBoost
                if self.models_trained['xgboost']:
                    try:
                        X_xgb = self.scalers['xgboost'].transform(X_current)
                        predictions['xgboost'] = self.models['xgboost'].predict(X_xgb)[0]
                    except:
                        pass
                
                # LSTM
                if self.models_trained['lstm'] and TENSORFLOW_AVAILABLE:
                    try:
                        X_lstm_scaled = self.scalers['lstm'].transform(X_current)
                        X_lstm_reshaped = X_lstm_scaled.reshape(1, X_lstm_scaled.shape[1], 1)
                        predictions['lstm'] = self.models['lstm'].predict(X_lstm_reshaped, verbose=0)[0][0]
                    except:
                        pass
            
            # Ensemble with dynamic weights
            if predictions:
                weighted_prediction = sum(pred * self.model_weights.get(name, 0) 
                                        for name, pred in predictions.items())
                total_weight = sum(self.model_weights.get(name, 0) for name in predictions.keys())
                
                if total_weight > 0:
                    final_prediction = weighted_prediction / total_weight
                    return {
                        'timestamp': datetime.now().isoformat(),
                        'current_price': self.current_price,
                        'predicted_price': final_prediction,
                        'individual_predictions': predictions,
                        'model_weights': self.model_weights.copy(),
                        'model': 'Enhanced Ensemble with LSTM',
                        'confidence_lower': final_prediction * 0.98,
                        'confidence_upper': final_prediction * 1.02
                    }
            return None
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def update_model_performance(self, predictions_dict, actual_price):
        """Update performance and adjust weights"""
        try:
            for model_name, predicted_price in predictions_dict.items():
                error = abs(actual_price - predicted_price)
                self.model_performance[model_name].append(error)
                if len(self.model_performance[model_name]) > 50:
                    self.model_performance[model_name] = self.model_performance[model_name][-50:]
            self.adjust_model_weights()
        except Exception as e:
            print(f"⚠️ Performance error: {e}")
    
    def save_predictions(self):
        """Save predictions"""
        try:
            if self.predictions:
                pd.DataFrame(self.predictions).to_csv('data/predictions.csv', index=False)
        except Exception as e:
            print(f"⚠️ Save error: {e}")
    
    def run_prediction_loop(self):
        """Main loop"""
        print("🚀 Enhanced BTC Predictor with LSTM")
        print("📊 Models: ARIMA + SVM + Boruta+RF + XGBoost + LSTM")
        print("🎯 Target: $20-60 USD with dynamic weights")
        print("="*60)
        
        retrain_counter = 0
        
        while True:
            try:
                if self.fetch_btc_data():
                    print(f"💰 BTC: ${self.current_price:,.2f}")
                    
                    retrain_counter += 1
                    if retrain_counter >= 60:  # Retrain every hour
                        print("🔄 Retraining with 90-day data...")
                        if self.train_models():
                            print("✅ Models retrained")
                        retrain_counter = 0
                    elif len(self.price_history) >= 50 and len(self.price_history) % 20 == 0:
                        self.train_models()
                    
                    if any(self.models_trained.values()):
                        prediction = self.make_prediction()
                        if prediction:
                            self.predictions.append(prediction)
                            self.save_predictions()
                            
                            pred_price = prediction['predicted_price']
                            direction = "📈" if pred_price > self.current_price else "📉"
                            change = abs(pred_price - self.current_price)
                            print(f"🔮 Pred: ${pred_price:,.2f} {direction} (±${change:.2f})")
                            
                            # Performance tracking
                            if len(self.predictions) >= 2 and 'individual_predictions' in self.predictions[-2]:
                                self.update_model_performance(
                                    self.predictions[-2]['individual_predictions'], 
                                    self.current_price
                                )
                            
                            if len(self.predictions) > 1000:
                                self.predictions = self.predictions[-1000:]
                
                time.sleep(60)  # 1-minute intervals
            except KeyboardInterrupt:
                print("\n🛑 Stopping...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(30)

def main():
    predictor = EnhancedBTCPredictor()
    
    # Start prediction loop
    prediction_thread = threading.Thread(target=predictor.run_prediction_loop)
    prediction_thread.daemon = True
    prediction_thread.start()
    
    # Start API server
    print("🌐 API server starting on port 8080...")
    predictor.app.run(host='0.0.0.0', port=8080, debug=False)

if __name__ == "__main__":
    main()