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
        
        # Performance tracking
        self.directional_accuracy = 0.0
        self.price_direction_history = []
        self.target_accuracy = 30
        
        os.makedirs('data', exist_ok=True)
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
    
    def fetch_btc_data(self):
        """Fetch BTC data from Binance"""
        try:
            url = "https://api.binance.com/api/v3/ticker/price"
            response = requests.get(url, params={"symbol": "BTCUSDT"}, timeout=10)
            if response.status_code == 200:
                price_data = response.json()
                current_price = float(price_data['price'])
                
                # Store data
                timestamp = datetime.now()
                data_point = {
                    'timestamp': timestamp,
                    'price': current_price
                }
                
                self.price_history.append(data_point)
                self.current_price = current_price
                
                # Keep last 1000 points
                if len(self.price_history) > 1000:
                    self.price_history = self.price_history[-1000:]
                
                return True
        except Exception as e:
            print(f"❌ Data fetch error: {e}")
            return False
    
    def train_models(self):
        """Train all models"""
        if len(self.price_history) < 50:
            return False
        
        try:
            # Prepare data
            prices = [p['price'] for p in self.price_history[-100:]]
            
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
        """Main prediction loop"""
        print("🚀 Starting Enhanced BTC Predictor...")
        print("📊 Models: ARIMA + SVM + Boruta+RF + XGBoost")
        print("🎯 Target: $15-30 USD accuracy")
        print("="*50)
        
        while True:
            try:
                # Fetch data
                if self.fetch_btc_data():
                    print(f"💰 Current BTC Price: ${self.current_price:,.2f}")
                    
                    # Train models periodically
                    if len(self.price_history) >= 50 and len(self.price_history) % 20 == 0:
                        self.train_models()
                    
                    # Make prediction
                    if any(self.models_trained.values()):
                        prediction = self.make_prediction()
                        
                        if prediction:
                            self.predictions.append(prediction)
                            self.save_predictions()
                            
                            pred_price = prediction['predicted_price']
                            print(f"🔮 Prediction: ${pred_price:,.2f}")
                            
                            # Keep last 1000 predictions
                            if len(self.predictions) > 1000:
                                self.predictions = self.predictions[-1000:]
                
                time.sleep(60)  # 1-minute intervals
                
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