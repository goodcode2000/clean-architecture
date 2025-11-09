#!/usr/bin/env python3
"""
Enhanced High-Precision BTC Predictor - Final Version
New Architecture: ARIMA + SVM + Boruta+RF + LightGBM
90-day historical data management with 5-minute updates
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
import lightgbm as lgb
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
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'lstm': None  # Will be built dynamically
        }
        
        # Dynamic weights (start equal, adjust based on performance)
        if TENSORFLOW_AVAILABLE:
            self.model_weights = {
                'arima': 0.2,
                'svm': 0.2,
                'boruta_rf': 0.2,
                'lightgbm': 0.2,
                'lstm': 0.2
            }
        else:
            self.model_weights = {
                'arima': 0.25,
                'svm': 0.25,
                'boruta_rf': 0.25,
                'lightgbm': 0.25,
                'lstm': 0.0
            }
        
        # Performance tracking for dynamic weight adjustment
        self.model_performance = {
            'arima': [],
            'svm': [],
            'boruta_rf': [],
            'lightgbm': [],
            'lstm': []
        }
        
        # Scalers
        self.scalers = {
            'svm': StandardScaler(),
            'boruta_rf': RobustScaler(),
            'lightgbm': StandardScaler(),
            'lstm': StandardScaler()
        }
        
        # Model training status
        self.models_trained = {
            'arima': False,
            'svm': False,
            'boruta_rf': False,
            'lightgbm': False,
            'lstm': False
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
                "models": "ARIMA+SVM+Boruta+RF+LightGBM+LSTM",
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
                    
                    # Train LightGBM
                    try:
                        X_lgb = self.scalers['lightgbm'].fit_transform(X)
                        self.models['lightgbm'].fit(X_lgb, y)
                        self.models_trained['lightgbm'] = True
                        print("✅ LightGBM trained")
                    except:
                        print("⚠️ LightGBM training failed")
                    
                    # Train LSTM
                    if TENSORFLOW_AVAILABLE:
                        try:
                            self.train_lstm_model(X, y)
                        except Exception as e:
                            print(f"⚠️ LSTM training failed: {e}")
            
            return any(self.models_trained.values())
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False    

    def make_prediction(self, prediction_timestamp=None):
        """Make ensemble prediction for 5 minutes ahead"""
        try:
            if not any(self.models_trained.values()):
                return None
            
            # Use provided timestamp or create clean timestamp (no seconds)
            if prediction_timestamp is None:
                prediction_timestamp = datetime.now().replace(second=0, microsecond=0)
            
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
                
                # LightGBM
                if self.models_trained['lightgbm']:
                    try:
                        X_lgb = self.scalers['lightgbm'].transform(X_current)
                        lgb_pred = self.models['lightgbm'].predict(X_lgb)[0]
                        predictions['lightgbm'] = lgb_pred
                    except:
                        pass
                
                # LSTM
                if self.models_trained['lstm'] and TENSORFLOW_AVAILABLE:
                    try:
                        # Prepare sequence for LSTM - need 10 sequences of 10 features each
                        if len(self.price_history) >= 20:
                            # Get last 20 prices to create 10 sequences
                            recent_prices = [p['price'] for p in self.price_history[-20:]]
                            
                            # Create 10 sequences of 10 prices each (sliding window)
                            sequences = []
                            for i in range(10, 20):
                                sequences.append(recent_prices[i-10:i])
                            
                            X_lstm_seq = np.array(sequences)  # Shape: (10, 10)
                            # Scale each sequence separately (10 features per sequence)
                            X_lstm_scaled = self.scalers['lstm'].transform(X_lstm_seq)
                            # Reshape to (1, 10, 10) for prediction
                            X_lstm_reshaped = X_lstm_scaled.reshape(1, 10, 10)
                            
                            lstm_pred = self.models['lstm'].predict(X_lstm_reshaped, verbose=0)[0][0]
                            predictions['lstm'] = lstm_pred
                    except Exception as e:
                        print(f"⚠️ LSTM prediction error: {e}")
            
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
                        'timestamp': prediction_timestamp.isoformat(),
                        'current_price': self.current_price,
                        'predicted_price': final_prediction,
                        'prediction_for_time': (prediction_timestamp + timedelta(minutes=5)).isoformat(),
                        'individual_predictions': predictions,
                        'model_weights': self.model_weights.copy(),
                        'model': 'Enhanced Ensemble with LightGBM+LSTM (5min ahead)',
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
        """Main prediction loop with 1-minute intervals, predicting 5 minutes ahead"""
        print("🚀 Starting Enhanced BTC Predictor...")
        print("📊 Models: ARIMA + SVM + Boruta+RF + LightGBM + LSTM")
        print("🎯 Target: $20-60 USD accuracy with rapid change detection")
        print("📈 90-day historical data management enabled")
        print("🔮 Prediction: 5 minutes ahead")
        print("="*60)
        
        # Wait until next exact minute mark before starting predictions
        now = datetime.now()
        seconds_to_wait = 60 - now.second
        if seconds_to_wait < 60:
            print(f"⏳ Waiting {seconds_to_wait} seconds until next minute mark...")
            time.sleep(seconds_to_wait)
        
        retrain_counter = 0
        
        while True:
            try:
                # Get current time and round to exact minute (remove seconds/microseconds)
                prediction_time = datetime.now().replace(second=0, microsecond=0)
                
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
                    
                    # Make prediction with clean timestamp
                    if any(self.models_trained.values()):
                        prediction = self.make_prediction(prediction_time)
                        
                        if prediction:
                            self.predictions.append(prediction)
                            self.save_predictions()
                            
                            pred_price = prediction['predicted_price']
                            pred_for_time = prediction['prediction_for_time']
                            direction = "📈" if pred_price > self.current_price else "📉"
                            change = abs(pred_price - self.current_price)
                            print(f"🔮 Prediction: ${pred_price:,.2f} {direction} (±${change:.2f}) for {pred_for_time}")
                            
                            # Update model performance for dynamic weight adjustment
                            if len(self.predictions) > 1 and 'individual_predictions' in prediction:
                                # Check if we have actual price from 2 minutes ago to evaluate performance
                                if len(self.predictions) >= 2:
                                    prev_prediction = self.predictions[-2]
                                    if 'individual_predictions' in prev_prediction:
                                        self.update_model_performance(
                                            prev_prediction['individual_predictions'], 
                                            self.current_price
                                        )
                            
                            # Keep last 1000 predictions
                            if len(self.predictions) > 1000:
                                self.predictions = self.predictions[-1000:]
                
                # Sleep until next exact minute mark
                now = datetime.now()
                seconds_until_next_minute = 60 - now.second
                time.sleep(seconds_until_next_minute)
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping predictor...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(30)
    
    def train_lstm_model(self, X, y):
        """Train LSTM model for sequential pattern learning"""
        try:
            if len(X) < 50:
                return False
            
            # Scale features for LSTM
            X_scaled = self.scalers['lstm'].fit_transform(X)
            
            # Reshape for LSTM (samples, timesteps, features)
            # Use last 10 timesteps for sequence
            X_lstm = []
            y_lstm = []
            
            for i in range(10, len(X_scaled)):
                X_lstm.append(X_scaled[i-10:i])
                y_lstm.append(y[i])
            
            if len(X_lstm) < 20:
                return False
            
            X_lstm = np.array(X_lstm)
            y_lstm = np.array(y_lstm)
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train model
            model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)
            
            self.models['lstm'] = model
            self.models_trained['lstm'] = True
            
            print("✅ LSTM model trained successfully")
            return True
            
        except Exception as e:
            print(f"❌ LSTM training error: {e}")
            return False
    
    def update_model_performance(self, predictions_dict, actual_price):
        """Update individual model performance for dynamic weight adjustment"""
        try:
            for model_name, predicted_price in predictions_dict.items():
                error = abs(actual_price - predicted_price)
                self.model_performance[model_name].append(error)
                
                # Keep only last 50 performance records
                if len(self.model_performance[model_name]) > 50:
                    self.model_performance[model_name] = self.model_performance[model_name][-50:]
            
            # Adjust weights based on performance
            self.adjust_model_weights()
            
        except Exception as e:
            print(f"⚠️ Performance update error: {e}")
    
    def detect_market_condition(self):
        """Detect current market condition: stable, volatile, trending up, or trending down"""
        try:
            if len(self.price_history) < 20:
                return 'stable', 0.0
            
            # Get recent prices
            recent_prices = [p['price'] for p in self.price_history[-20:]]
            
            # Calculate volatility (standard deviation)
            volatility = np.std(recent_prices)
            avg_price = np.mean(recent_prices)
            volatility_pct = (volatility / avg_price) * 100
            
            # Calculate trend (linear regression slope)
            x = np.arange(len(recent_prices))
            slope = np.polyfit(x, recent_prices, 1)[0]
            trend_pct = (slope / avg_price) * 100
            
            # Determine market condition
            if volatility_pct > 0.5:  # High volatility
                if abs(trend_pct) > 0.3:
                    condition = 'volatile_trending'
                else:
                    condition = 'volatile'
            elif abs(trend_pct) > 0.2:  # Clear trend
                if trend_pct > 0:
                    condition = 'trending_up'
                else:
                    condition = 'trending_down'
            else:
                condition = 'stable'
            
            return condition, volatility_pct
            
        except Exception as e:
            print(f"⚠️ Market condition detection error: {e}")
            return 'stable', 0.0
    
    def adjust_model_weights(self):
        """Dynamically adjust model weights based on recent performance AND market conditions"""
        try:
            # Detect current market condition
            market_condition, volatility = self.detect_market_condition()
            
            model_scores = {}
            
            for model_name, errors in self.model_performance.items():
                if len(errors) >= 5:  # Need at least 5 predictions
                    # Use recent errors with exponential weighting (more recent = more important)
                    recent_errors = errors[-20:]
                    weights = np.exp(np.linspace(-1, 0, len(recent_errors)))
                    weighted_error = np.average(recent_errors, weights=weights)
                    
                    # Base score (lower error = higher score)
                    base_score = 1 / (1 + weighted_error)
                    
                    # Apply market condition multipliers
                    multiplier = 1.0
                    
                    if market_condition == 'volatile' or market_condition == 'volatile_trending':
                        # Favor LightGBM and LSTM for volatile markets
                        if model_name == 'lightgbm':
                            multiplier = 1.3
                        elif model_name == 'lstm':
                            multiplier = 1.2
                        elif model_name == 'arima':
                            multiplier = 0.7  # ARIMA struggles with volatility
                    
                    elif market_condition == 'trending_up' or market_condition == 'trending_down':
                        # Favor ARIMA and LSTM for trending markets
                        if model_name == 'arima':
                            multiplier = 1.3
                        elif model_name == 'lstm':
                            multiplier = 1.2
                        elif model_name == 'svm':
                            multiplier = 1.1
                    
                    elif market_condition == 'stable':
                        # Favor SVM and RF for stable markets
                        if model_name == 'svm':
                            multiplier = 1.2
                        elif model_name == 'boruta_rf':
                            multiplier = 1.2
                        elif model_name == 'lightgbm':
                            multiplier = 0.9
                    
                    model_scores[model_name] = base_score * multiplier
                else:
                    # Default score for models without enough data
                    model_scores[model_name] = 0.5
            
            # Normalize scores to weights
            total_score = sum(model_scores.values())
            if total_score > 0:
                new_weights = {}
                for model_name in self.model_weights:
                    if model_name in model_scores:
                        new_weight = model_scores[model_name] / total_score
                        # Smooth weight adjustment (don't change too drastically)
                        # Use adaptive smoothing: faster adjustment in volatile markets
                        if market_condition == 'volatile' or market_condition == 'volatile_trending':
                            smooth_factor = 0.5  # Faster adaptation
                        else:
                            smooth_factor = 0.7  # Slower adaptation
                        
                        new_weights[model_name] = smooth_factor * self.model_weights[model_name] + (1 - smooth_factor) * new_weight
                    else:
                        new_weights[model_name] = self.model_weights[model_name]
                
                # Ensure weights sum to 1
                total_weight = sum(new_weights.values())
                if total_weight > 0:
                    for model_name in new_weights:
                        self.model_weights[model_name] = new_weights[model_name] / total_weight
                
                print(f"📊 Market: {market_condition} (vol: {volatility:.2f}%)")
                print(f"🎯 Dynamic weights: ARIMA:{self.model_weights['arima']:.2f} SVM:{self.model_weights['svm']:.2f} RF:{self.model_weights['boruta_rf']:.2f} LGB:{self.model_weights['lightgbm']:.2f} LSTM:{self.model_weights['lstm']:.2f}")
            
        except Exception as e:
            print(f"⚠️ Weight adjustment error: {e}")

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

