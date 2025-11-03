"""
Main application for BTC price prediction
Runs on Ubuntu VPS with 16GB GPU
Updates predictions every 5 minutes
"""
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from config import HISTORY_DAYS, INTERVAL_MINUTES, DATA_DIR, MODEL_DIR, PREDICTIONS_FILE
from data_fetcher import BTCDataFetcher
from feature_engineering import FeatureEngineer
from train_ensemble import EnsembleTrainer
from predictor import EnsemblePredictor


class BTCPricePredictorApp:
    def __init__(self):
        """Initialize the application"""
        self.fetcher = BTCDataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.trainer = EnsembleTrainer()
        self.predictor = EnsemblePredictor()
        
        # Initialize directories
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Training tracking
        self.last_training_time = None
        self.last_data_update = None
        self.data_cache = None
        
        print("=" * 70)
        print("BTC Price Prediction System - App 1 (Ubuntu VPS)")
        print("=" * 70)
    
    def initial_training(self):
        """Perform initial model training"""
        print("\n[INITIAL TRAINING]")
        print("Fetching historical data...")
        
        try:
            data = self.fetcher.get_historical_data(HISTORY_DAYS)
            self.data_cache = data
            
            print(f"Data shape: {data.shape}")
            print(f"Date range: {data.index[0]} to {data.index[-1]}")
            
            print("\nTraining ensemble models...")
            self.trainer.train_models(data)
            
            self.last_training_time = datetime.now()
            self.last_data_update = datetime.now()
            
            print("\nInitial training completed!")
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_data(self):
        """Update data with latest information"""
        try:
            # Get latest data point
            latest_point = self.fetcher.get_latest_data_point()
            latest_time = latest_point.name
            
            # Check if we have this data point already
            if self.data_cache is not None and len(self.data_cache) > 0:
                if latest_time <= self.data_cache.index[-1]:
                    # No new data
                    return False
            
            # Append new data point
            if self.data_cache is None:
                # Fetch full history if cache is empty
                self.data_cache = self.fetcher.get_historical_data(HISTORY_DAYS)
            else:
                # Append new point
                self.data_cache = pd.concat([self.data_cache, pd.DataFrame([latest_point])])
                self.data_cache = self.data_cache.sort_index()
                self.data_cache = self.data_cache.drop_duplicates()
                
                # Keep only last HISTORY_DAYS
                cutoff = datetime.now() - timedelta(days=HISTORY_DAYS)
                self.data_cache = self.data_cache[self.data_cache.index >= cutoff]
            
            self.last_data_update = datetime.now()
            return True
            
        except Exception as e:
            print(f"Data update error: {e}")
            return False
    
    def make_prediction(self):
        """Make prediction for next 5 minutes"""
        try:
            # Update data first
            self.update_data()
            
            if self.data_cache is None or len(self.data_cache) == 0:
                print("No data available for prediction")
                return None
            
            # Create features
            data_with_features = self.feature_engineer.create_features(self.data_cache.copy())
            
            if len(data_with_features) == 0:
                print("No features available")
                return None
            
            # Make prediction
            predictions = self.predictor.predict(data_with_features)
            
            return predictions
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def should_retrain(self):
        """Check if models should be retrained"""
        if self.last_training_time is None:
            return True
        
        hours_since_training = (datetime.now() - self.last_training_time).total_seconds() / 3600
        return hours_since_training >= 24  # Retrain every 24 hours
    
    def retrain_models(self):
        """Retrain models with recent data"""
        print("\n[RETRAINING MODELS]")
        print(f"Last training: {self.last_training_time}")
        
        try:
            # Update full dataset
            print("Updating dataset...")
            self.data_cache = self.fetcher.get_historical_data(HISTORY_DAYS)
            
            print("Retraining ensemble...")
            self.trainer.train_models(self.data_cache)
            
            # Reload models in predictor
            self.predictor.load_models()
            
            self.last_training_time = datetime.now()
            print("Retraining completed!")
            return True
            
        except Exception as e:
            print(f"Retraining error: {e}")
            return False
    
    def display_prediction(self, current_price, predictions):
        """Display prediction results in terminal"""
        print("\n" + "=" * 70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 70)
        print(f"{'Current Real Price:':<30} ${current_price:>10.2f}")
        print("-" * 70)
        
        if predictions:
            ensemble_pred = predictions.get('ensemble', current_price)
            offset_stats = self.predictor.get_offset_statistics()
            
            print(f"{'Predicted Price (5 min):':<30} ${ensemble_pred:>10.2f}")
            print(f"{'Expected Change:':<30} ${ensemble_pred - current_price:>10.2f} ({predictions.get('price_change_pct', 0):>5.2f}%)")
            
            if predictions.get('rapid_change', False):
                print(f"{'⚠️  RAPID CHANGE PREDICTED':<30}")
            
            print("-" * 70)
            print("Individual Model Predictions:")
            for model_name in ['ets', 'garch', 'lightgbm', 'tcn', 'cnn']:
                if model_name in predictions:
                    pred = predictions[model_name]
                    print(f"  {model_name.upper():<15} ${pred:>10.2f}")
            
            if offset_stats:
                print("-" * 70)
                print(f"Offset Statistics (from {offset_stats['count']} predictions):")
                print(f"  Mean Offset:   ${offset_stats['mean_offset']:.2f}")
                print(f"  Median Offset: ${offset_stats['median_offset']:.2f}")
                print(f"  Std Offset:    ${offset_stats['std_offset']:.2f}")
        else:
            print("Prediction unavailable")
        
        print("=" * 70)
    
    def run(self):
        """Main application loop"""
        # Initial training
        if not os.path.exists(os.path.join(MODEL_DIR, 'lightgbm_model.pkl')):
            if not self.initial_training():
                print("Failed to complete initial training. Exiting.")
                return
        
        # Load existing models if available
        print("\nLoading existing models...")
        self.predictor.load_models()
        
        # First prediction - show only real price
        print("\n" + "=" * 70)
        print("Initializing...")
        print("=" * 70)
        try:
            current_price = self.fetcher.get_current_price()
            print(f"\n{'Current Real Price:':<30} ${current_price:>10.2f}")
            print("Waiting for first prediction cycle...")
        except Exception as e:
            print(f"Error fetching current price: {e}")
        
        # Main loop
        cycle_count = 0
        while True:
            try:
                cycle_count += 1
                print(f"\n[Cycle {cycle_count}]")
                
                # Check if retraining is needed
                if self.should_retrain():
                    self.retrain_models()
                
                # Get current price
                current_price = self.fetcher.get_current_price()
                
                # Make prediction
                predictions = self.make_prediction()
                
                if predictions:
                    ensemble_pred = predictions.get('ensemble', current_price)
                    
                    # Save prediction
                    self.predictor.save_prediction(current_price, ensemble_pred, predictions)
                    
                    # Display results
                    self.display_prediction(current_price, predictions)
                
                # Wait for next cycle (5 minutes)
                print(f"\nNext update in {INTERVAL_MINUTES} minutes...")
                time.sleep(INTERVAL_MINUTES * 60)
                
            except KeyboardInterrupt:
                print("\n\nApplication stopped by user")
                break
            except Exception as e:
                print(f"\nError in main loop: {e}")
                import traceback
                traceback.print_exc()
                print("Retrying in 1 minute...")
                time.sleep(60)


if __name__ == "__main__":
    app = BTCPricePredictorApp()
    app.run()

