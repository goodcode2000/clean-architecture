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
from sklearn.model_selection import cross_val_score

# Advanced models
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available, LSTM model will be disabled")

# Feature selection
try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False
    print("⚠️ Boruta not available, using standard feature selection")

class EnhancedBTCPredictor:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Data storage
        self.price_history = []
        self.predictions = []
        self.current_price = 0
        self.feature_data = []
        
        # Enhanced model ensemble
        self.models = {
            'arima': None,  # ARIMA for time series patterns
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
            'lstm': None  # Will be built dynamically
        }
        
        # Dynamic weights (start equal, adjust based on performance)
        if TENSORFLOW_AVAILABLE:
            self.model_weights = {
                'arima': 0.2,
                'svm': 0.2,
                'boruta_rf': 0.2,
                'xgboost': 0.2,
                'lstm': 0.2
            }
        else:
            self.model_weights = {
                'arima': 0.25,
                'svm': 0.25,
                'boruta_rf': 0.25,
                'xgboost': 0.25,
                'lstm': 0.0
            }
        
        # Performance tracking for weight adjustment
        self.model_performance = {
            'arima': [],
            'svm': [],
            'boruta_rf': [],
            'xgboost': [],
            'lstm': []
        }
        
        # Scalers for different models
        self.scalers = {
            'svm': StandardScaler(),
            'boruta_rf': RobustScaler(),
            'xgboost': StandardScaler(),
            'lstm': MinMaxScaler(feature_range=(0, 1))
        }
        
        # Feature selection
        self.boruta_selector = None
        self.selected_features = None
        
        # Model training status
        self.models_trained = {
            'arima': False,
            'svm': False,
            'boruta_rf': False,
            'xgboost': False,
            'lstm': False
        }
        
        # Enhanced market analysis
        self.market_state = 'stable'
        self.directional_accuracy = 0.0
        self.price_direction_history = []
        
        # Performance targets
        self.target_accuracy = 30  # $30 USD max error
        self.target_directional_accuracy = 0.75  # 75% directional accuracy
        
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask API routes"""
        
        @self.app.route('/api/health')
        def health():
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "3.0.0-enhanced",
                "models": "ARIMA+SVM+Boruta+RF+XGBoost+LSTM",
                "target_accuracy": "$15-30 USD",
                "directional_accuracy": f"{self.directional_accuracy:.1%}"
            })
        
        @self.app.route('/api/current-price')
        def current_price():
            return jsonify({
                "price": self.current_price,
                "timestamp": datetime.now().isoformat(),
                "market_state": self.market_state,
                "model_weights": self.model_weights
            })
        
        @self.app.route('/api/latest-prediction')
        def latest_prediction():
            if self.predictions:
                latest = self.predictions[-1].copy()
                latest['model_ensemble'] = "ARIMA+SVM+Boruta+RF+XGBoost+LSTM"
                latest['dynamic_weights'] = self.model_weights
                latest['directional_accuracy'] = f"{self.directional_accuracy:.1%}"
                return jsonify(latest)
            return jsonify({"error": "No predictions available"})
        
        @self.app.route('/api/prediction-history')
        def prediction_history():
            recent_preds = self.predictions[-50:]
            accurate_preds = [p for p in recent_preds if 'error' in p and p['error'] <= self.target_accuracy]
            
            return jsonify({
                "predictions": recent_preds,
                "total": len(self.predictions),
                "accurate_predictions": len(accurate_preds),
                "accuracy_rate": f"{(len(accurate_preds)/len(recent_preds)*100):.1f}%" if recent_preds else "0%",
                "directional_accuracy": f"{self.directional_accuracy:.1%}",
                "model_weights": self.model_weights
            })
        
        @self.app.route('/api/model-performance')
        def model_performance():
            return jsonify({
                "individual_performance": self.model_performance,
                "current_weights": self.model_weights,
                "models_trained": self.models_trained,
                "selected_features": len(self.selected_features) if self.selected_features is not None else 0
            })
    
    def fetch_btc_data(self):
        """Enhanced data fetching with more features"""
        try:
            # Get current price
            url = "https://api.binance.com/api/v3/ticker/price"
            response = requests.get(url, params={"symbol": "BTCUSDT"}, timeout=10)
            if response.status_code == 200:
                price_data = response.json()
                current_price = float(price_data['price'])
                
                # Get 24hr statistics for additional features
                stats_url = "https://api.binance.com/api/v3/ticker/24hr"
                stats_response = requests.get(stats_url, params={"symbol": "BTCUSDT"}, timeout=10)
                stats = stats_response.json() if stats_response.status_code == 200 else {}
                
                # Get recent klines for technical indicators
                klines_url = "https://api.binance.com/api/v3/klines"
                klines_response = requests.get(klines_url, params={
                    "symbol": "BTCUSDT",
                    "interval": "1m",
                    "limit": 100
                }, timeout=10)
                klines = klines_response.json() if klines_response.status_code == 200 else []
                
                # Create enhanced feature set
                features = self.create_enhanced_features(current_price, stats, klines)
                
                # Store data
                timestamp = datetime.now()
                data_point = {
                    'timestamp': timestamp,
                    'price': current_price,
                    'features': features
                }
                
                self.price_history.append(data_point)
                self.current_price = current_price
                
                # Keep last 1000 points for efficiency
                if len(self.price_history) > 1000:
                    self.price_history = self.price_history[-1000:]
                
                return True
                
        except Exception as e:
            print(f"❌ Data fetch error: {e}")
            return False
    
    def create_enhanced_features(self, current_price, stats, klines):
        """Create comprehensive feature set for ML models"""
        features = {}
        
        try:
            # Basic price features
            features['price'] = current_price
            features['log_price'] = np.log(current_price) if current_price > 0 else 0
            
            # 24hr statistics features
            if stats:
                features['volume'] = float(stats.get('volume', 0))
                features['high_24h'] = float(stats.get('highPrice', current_price))
                features['low_24h'] = float(stats.get('lowPrice', current_price))
                features['price_change_24h'] = float(stats.get('priceChange', 0))
                features['price_change_pct_24h'] = float(stats.get('priceChangePercent', 0))
                features['weighted_avg_price'] = float(stats.get('weightedAvgPrice', current_price))
                
            # Technical indicators from klines
            if len(klines) >= 20:
                closes = [float(k[4]) for k in klines]  # Close prices
                volumes = [float(k[5]) for k in klines]  # Volumes
                
                # Moving averages
                features['sma_5'] = np.mean(closes[-5:])
                features['sma_10'] = np.mean(closes[-10:])
                features['sma_20'] = np.mean(closes[-20:])
                
                # Exponential moving averages
                features['ema_5'] = self.calculate_ema(closes, 5)
                features['ema_10'] = self.calculate_ema(closes, 10)
                
                # Volatility measures
                features['volatility_5'] = np.std(closes[-5:])
                features['volatility_10'] = np.std(closes[-10:])
                features['volatility_20'] = np.std(closes[-20:])
                
                # Price momentum
                features['momentum_5'] = closes[-1] - closes[-6] if len(closes) > 5 else 0
                features['momentum_10'] = closes[-1] - closes[-11] if len(closes) > 10 else 0
                
                # RSI (Relative Strength Index)
                features['rsi'] = self.calculate_rsi(closes)
                
                # MACD
                macd_line, signal_line = self.calculate_macd(closes)
                features['macd'] = macd_line
                features['macd_signal'] = signal_line
                features['macd_histogram'] = macd_line - signal_line
                
                # Bollinger Bands
                bb_upper, bb_lower, bb_middle = self.calculate_bollinger_bands(closes)
                features['bb_upper'] = bb_upper
                features['bb_lower'] = bb_lower
                features['bb_middle'] = bb_middle
                features['bb_position'] = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
                
                # Volume features
                features['volume_sma_5'] = np.mean(volumes[-5:])
                features['volume_ratio'] = volumes[-1] / np.mean(volumes[-10:]) if len(volumes) >= 10 else 1
                
            # Time-based features
            now = datetime.now()
            features['hour'] = now.hour
            features['day_of_week'] = now.weekday()
            features['is_weekend'] = 1 if now.weekday() >= 5 else 0
            
            # Historical price features
            if len(self.price_history) >= 10:
                recent_prices = [p['price'] for p in self.price_history[-10:]]
                features['price_trend_5'] = np.polyfit(range(5), recent_prices[-5:], 1)[0] if len(recent_prices) >= 5 else 0
                features['price_acceleration'] = np.polyfit(range(len(recent_prices)), recent_prices, 2)[0] if len(recent_prices) >= 3 else 0
                
        except Exception as e:
            print(f"⚠️ Feature creation error: {e}")
            
        return features
    
    def calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        if len(prices) < slow:
            return 0, 0
        
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        
        # For simplicity, using the current MACD as signal (in real implementation, you'd calculate EMA of MACD)
        signal_line = macd_line * 0.9  # Simplified signal line
        
        return macd_line, signal_line
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            sma = np.mean(prices)
            std = np.std(prices)
        else:
            sma = np.mean(prices[-period:])
            std = np.std(prices[-period:])
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, lower_band, sma
    
    def prepare_training_data(self):
        """Prepare data for model training with feature selection"""
        if len(self.price_history) < 50:
            return None, None, None
        
        try:
            # Extract features and targets
            feature_list = []
            target_list = []
            
            for i in range(len(self.price_history) - 5):  # Leave 5 for prediction horizon
                current_features = self.price_history[i]['features']
                future_price = self.price_history[i + 2]['price']  # 2-minute ahead prediction
                
                # Convert features dict to list
                feature_vector = []
                feature_names = []
                for key, value in current_features.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        feature_vector.append(value)
                        feature_names.append(key)
                
                if len(feature_vector) > 0:
                    feature_list.append(feature_vector)
                    target_list.append(future_price)
            
            if len(feature_list) < 20:
                return None, None, None
            
            X = np.array(feature_list)
            y = np.array(target_list)
            
            # Feature selection with Boruta (if available)
            if BORUTA_AVAILABLE and self.boruta_selector is None and X.shape[1] > 5:
                try:
                    rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
                    self.boruta_selector = BorutaPy(rf_selector, n_estimators='auto', random_state=42)
                    self.boruta_selector.fit(X, y)
                    self.selected_features = self.boruta_selector.support_
                    print(f"🎯 Boruta selected {np.sum(self.selected_features)} features out of {len(self.selected_features)}")
                except Exception as e:
                    print(f"⚠️ Boruta feature selection failed: {e}")
                    self.selected_features = np.ones(X.shape[1], dtype=bool)
            elif self.selected_features is None:
                self.selected_features = np.ones(X.shape[1], dtype=bool)
            
            # Apply feature selection
            if self.selected_features is not None and len(self.selected_features) == X.shape[1]:
                X = X[:, self.selected_features]
            
            return X, y, feature_names
            
        except Exception as e:
            print(f"❌ Data preparation error: {e}")
            return None, None, None
    
    def train_arima_model(self):
        """Train ARIMA model for time series prediction"""
        try:
            if len(self.price_history) < 50:
                return False
            
            # Extract price series
            prices = [p['price'] for p in self.price_history[-100:]]  # Last 100 points
            
            # Fit ARIMA model
            model = ARIMA(prices, order=(2, 1, 2))  # ARIMA(2,1,2) - adjust as needed
            fitted_model = model.fit()
            self.models['arima'] = fitted_model
            self.models_trained['arima'] = True
            
            print("✅ ARIMA model trained successfully")
            return True
            
        except Exception as e:
            print(f"❌ ARIMA training error: {e}")
            return False
    
    def train_svm_model(self, X, y):
        """Train SVM model"""
        try:
            if X is None or len(X) < 20:
                return False
            
            # Scale features
            X_scaled = self.scalers['svm'].fit_transform(X)
            
            # Train SVM
            self.models['svm'].fit(X_scaled, y)
            self.models_trained['svm'] = True
            
            print("✅ SVM model trained successfully")
            return True
            
        except Exception as e:
            print(f"❌ SVM training error: {e}")
            return False
    
    def train_boruta_rf_model(self, X, y):
        """Train Boruta + Random Forest model"""
        try:
            if X is None or len(X) < 20:
                return False
            
            # Scale features
            X_scaled = self.scalers['boruta_rf'].fit_transform(X)
            
            # Train Random Forest with Boruta-selected features
            self.models['boruta_rf'].fit(X_scaled, y)
            self.models_trained['boruta_rf'] = True
            
            print("✅ Boruta+RF model trained successfully")
            return True
            
        except Exception as e:
            print(f"❌ Boruta+RF training error: {e}")
            return False
    
    def train_xgboost_model(self, X, y):
        """Train XGBoost model for directional accuracy"""
        try:
            if X is None or len(X) < 20:
                return False
            
            # Scale features
            X_scaled = self.scalers['xgboost'].fit_transform(X)
            
            # Train XGBoost
            self.models['xgboost'].fit(X_scaled, y)
            self.models_trained['xgboost'] = True
            
            print("✅ XGBoost model trained successfully")
            return True
            
        except Exception as e:
            print(f"❌ XGBoost training error: {e}")
            return False
    
    def train_lstm_model(self, X, y):
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            return False
        
        try:
            if X is None or len(X) < 50:
                return False
            
            # Scale features
            X_scaled = self.scalers['lstm'].fit_transform(X)
            
            # Reshape for LSTM (samples, timesteps, features)
            # Use last 10 timesteps for sequence
            if len(X_scaled) < 10:
                return False
            
            X_lstm = []
            y_lstm = []
            
            for i in range(10, len(X_scaled)):
                X_lstm.append(X_scaled[i-10:i])
                y_lstm.append(y[i])
            
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
    
    def train_all_models(self):
        """Train all models in the ensemble"""
        print("🚀 Training enhanced model ensemble...")
        
        # Prepare training data
        X, y, feature_names = self.prepare_training_data()
        
        if X is None:
            print("❌ Insufficient data for training")
            return False
        
        print(f"📊 Training data: {len(X)} samples, {X.shape[1]} features")
        
        # Train individual models
        self.train_arima_model()
        self.train_svm_model(X, y)
        self.train_boruta_rf_model(X, y)
        self.train_xgboost_model(X, y)
        
        if TENSORFLOW_AVAILABLE:
            self.train_lstm_model(X, y)
        
        trained_count = sum(self.models_trained.values())
        print(f"✅ Trained {trained_count} models successfully")
        
        return trained_count > 0
    
    def make_ensemble_prediction(self):
        """Make prediction using ensemble of models with dynamic weights"""
        try:
            if not any(self.models_trained.values()):
                return None
            
            predictions = {}
            
            # ARIMA prediction
            if self.models_trained['arima'] and self.models['arima'] is not None:
                try:
                    arima_pred = self.models['arima'].forecast(steps=2)[1]  # 2-minute ahead
                    predictions['arima'] = arima_pred
                except Exception as e:
                    print(f"⚠️ ARIMA prediction error: {e}")
            
            # ML model predictions
            if len(self.price_history) > 0:
                current_features = self.price_history[-1]['features']
                feature_vector = []
                
                for key, value in current_features.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        feature_vector.append(value)
                
                if len(feature_vector) > 0:
                    X_current = np.array([feature_vector])
                    
                    # Apply feature selection
                    if self.selected_features is not None and len(self.selected_features) == X_current.shape[1]:
                        X_current = X_current[:, self.selected_features]
                    
                    # SVM prediction
                    if self.models_trained['svm']:
                        try:
                            X_svm = self.scalers['svm'].transform(X_current)
                            svm_pred = self.models['svm'].predict(X_svm)[0]
                            predictions['svm'] = svm_pred
                        except Exception as e:
                            print(f"⚠️ SVM prediction error: {e}")
                    
                    # Boruta+RF prediction
                    if self.models_trained['boruta_rf']:
                        try:
                            X_rf = self.scalers['boruta_rf'].transform(X_current)
                            rf_pred = self.models['boruta_rf'].predict(X_rf)[0]
                            predictions['boruta_rf'] = rf_pred
                        except Exception as e:
                            print(f"⚠️ Boruta+RF prediction error: {e}")
                    
                    # XGBoost prediction
                    if self.models_trained['xgboost']:
                        try:
                            X_xgb = self.scalers['xgboost'].transform(X_current)
                            xgb_pred = self.models['xgboost'].predict(X_xgb)[0]
                            predictions['xgboost'] = xgb_pred
                        except Exception as e:
                            print(f"⚠️ XGBoost prediction error: {e}")
                    
                    # LSTM prediction
                    if self.models_trained['lstm'] and TENSORFLOW_AVAILABLE:
                        try:
                            # Prepare sequence for LSTM
                            if len(self.price_history) >= 10:
                                recent_features = []
                                for i in range(-10, 0):
                                    hist_features = self.price_history[i]['features']
                                    hist_vector = []
                                    for key, value in hist_features.items():
                                        if isinstance(value, (int, float)) and not np.isnan(value):
                                            hist_vector.append(value)
                                    recent_features.append(hist_vector)
                                
                                if len(recent_features) == 10 and all(len(f) == len(recent_features[0]) for f in recent_features):
                                    X_lstm = np.array([recent_features])
                                    X_lstm_scaled = self.scalers['lstm'].transform(X_lstm.reshape(-1, X_lstm.shape[-1]))
                                    X_lstm_scaled = X_lstm_scaled.reshape(X_lstm.shape)
                                    
                                    lstm_pred = self.models['lstm'].predict(X_lstm_scaled, verbose=0)[0][0]
                                    predictions['lstm'] = lstm_pred
                        except Exception as e:
                            print(f"⚠️ LSTM prediction error: {e}")
            
            # Combine predictions using dynamic weights
            if predictions:
                weighted_prediction = 0
                total_weight = 0
                
                for model_name, pred_value in predictions.items():
                    weight = self.model_weights.get(model_name, 0)
                    weighted_prediction += pred_value * weight
                    total_weight += weight
                
                if total_weight > 0:
                    final_prediction = weighted_prediction / total_weight
                    
                    # Create prediction record
                    prediction_record = {
                        'timestamp': datetime.now().isoformat(),
                        'current_price': self.current_price,
                        'predicted_price': final_prediction,
                        'individual_predictions': predictions,
                        'model_weights': self.model_weights.copy(),
                        'confidence_lower': final_prediction * 0.98,
                        'confidence_upper': final_prediction * 1.02,
                        'model': 'Enhanced Ensemble',
                        'directional_prediction': 'up' if final_prediction > self.current_price else 'down'
                    }
                    
                    return prediction_record
            
            return None
            
        except Exception as e:
            print(f"❌ Ensemble prediction error: {e}")
            return None
    
    def update_model_performance(self, prediction_record):
        """Update individual model performance and adjust weights"""
        try:
            if len(self.predictions) < 2:
                return
            
            # Get actual price after 2 minutes (if available)
            current_time = datetime.now()
            for i in range(len(self.predictions) - 1, -1, -1):
                pred = self.predictions[i]
                pred_time = datetime.fromisoformat(pred['timestamp'])
                time_diff = (current_time - pred_time).total_seconds() / 60
                
                if 1.5 <= time_diff <= 2.5:  # Around 2 minutes
                    actual_price = self.current_price
                    predicted_price = pred['predicted_price']
                    error = abs(actual_price - predicted_price)
                    
                    # Update individual model performance
                    if 'individual_predictions' in pred:
                        for model_name, model_pred in pred['individual_predictions'].items():
                            model_error = abs(actual_price - model_pred)
                            self.model_performance[model_name].append(model_error)
                            
                            # Keep only last 50 performance records
                            if len(self.model_performance[model_name]) > 50:
                                self.model_performance[model_name] = self.model_performance[model_name][-50:]
                    
                    # Update directional accuracy
                    predicted_direction = pred.get('directional_prediction', 'neutral')
                    actual_direction = 'up' if actual_price > pred['current_price'] else 'down'
                    
                    if predicted_direction == actual_direction:
                        self.price_direction_history.append(1)
                    else:
                        self.price_direction_history.append(0)
                    
                    # Keep only last 100 directional predictions
                    if len(self.price_direction_history) > 100:
                        self.price_direction_history = self.price_direction_history[-100:]
                    
                    # Calculate directional accuracy
                    if len(self.price_direction_history) > 0:
                        self.directional_accuracy = np.mean(self.price_direction_history)
                    
                    # Update prediction record with actual error
                    pred['error'] = error
                    pred['actual_price'] = actual_price
                    
                    break
            
            # Adjust model weights based on performance
            self.adjust_model_weights()
            
        except Exception as e:
            print(f"⚠️ Performance update error: {e}")
    
    def adjust_model_weights(self):
        """Dynamically adjust model weights based on recent performance"""
        try:
            # Calculate average errors for each model
            model_scores = {}
            
            for model_name, errors in self.model_performance.items():
                if len(errors) >= 5:  # Need at least 5 predictions
                    avg_error = np.mean(errors[-20:])  # Last 20 predictions
                    # Convert error to score (lower error = higher score)
                    model_scores[model_name] = 1 / (1 + avg_error)
                else:
                    # Default score for models without enough data
                    model_scores[model_name] = 0.5
            
            # Normalize scores to weights
            total_score = sum(model_scores.values())
            if total_score > 0:
                for model_name in self.model_weights:
                    if model_name in model_scores:
                        new_weight = model_scores[model_name] / total_score
                        # Smooth weight adjustment (don't change too drastically)
                        self.model_weights[model_name] = 0.7 * self.model_weights[model_name] + 0.3 * new_weight
                    
                # Ensure weights sum to 1
                total_weight = sum(self.model_weights.values())
                if total_weight > 0:
                    for model_name in self.model_weights:
                        self.model_weights[model_name] /= total_weight
            
            print(f"🎯 Updated model weights: {self.model_weights}")
            
        except Exception as e:
            print(f"⚠️ Weight adjustment error: {e}")
    
    def save_predictions(self):
        """Save predictions to CSV file"""
        try:
            if len(self.predictions) > 0:
                df = pd.DataFrame(self.predictions)
                df.to_csv('data/predictions.csv', index=False)
                print(f"💾 Saved {len(self.predictions)} predictions to CSV")
        except Exception as e:
            print(f"⚠️ Save predictions error: {e}")
    
    def run_prediction_loop(self):
        """Main prediction loop"""
        print("🚀 Starting Enhanced BTC Predictor...")
        print("📊 Models: ARIMA + SVM + Boruta+RF + XGBoost + LSTM")
        print("🎯 Target: $15-30 USD accuracy with directional focus")
        print("="*60)
        
        while True:
            try:
                # Fetch new data
                if self.fetch_btc_data():
                    print(f"💰 Current BTC Price: ${self.current_price:,.2f}")
                    
                    # Train models if we have enough data
                    if len(self.price_history) >= 50 and len(self.price_history) % 20 == 0:
                        self.train_all_models()
                    
                    # Make prediction
                    if any(self.models_trained.values()):
                        prediction = self.make_ensemble_prediction()
                        
                        if prediction:
                            self.predictions.append(prediction)
                            
                            # Update performance and adjust weights
                            self.update_model_performance(prediction)
                            
                            # Save predictions
                            self.save_predictions()
                            
                            # Display prediction
                            pred_price = prediction['predicted_price']
                            direction = prediction['directional_prediction']
                            print(f"🔮 Prediction: ${pred_price:,.2f} ({direction}) | "
                                  f"Directional Accuracy: {self.directional_accuracy:.1%}")
                            
                            # Keep only last 1000 predictions
                            if len(self.predictions) > 1000:
                                self.predictions = self.predictions[-1000:]
                
                # Wait before next iteration
                time.sleep(60)  # 1-minute intervals
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping predictor...")
                break
            except Exception as e:
                print(f"❌ Main loop error: {e}")
                time.sleep(30)  # Wait 30 seconds on error

def main():
    predictor = EnhancedBTCPredictor()
    
    # Start prediction loop in a separate thread
    prediction_thread = threading.Thread(target=predictor.run_prediction_loop)
    prediction_thread.daemon = True
    prediction_thread.start()
    
    # Start Flask API server
    print("🌐 Starting API server on port 8080...")
    predictor.app.run(host='0.0.0.0', port=8080, debug=False)

if __name__ == "__main__":
    main()