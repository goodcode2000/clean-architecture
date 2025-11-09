"""Ensemble model combining ETS, SVR, Random Forest, XGBoost, and LSTM for TAO prediction."""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, List, Optional
from loguru import logger
import joblib
import os
import sys
from datetime import datetime
import threading
import time
from collections import deque

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from models.ets_model import ETSPredictor
from models.svr_model import SVRPredictor
from models.random_forest_model import RandomForestPredictor
from models.lstm_model import LSTMPredictor
from models.kalman_model import KalmanPredictor
from services.feature_engineering import FeatureEngineer
from services.preprocessing import DataPreprocessor

class EnsemblePredictor:
    """Ensemble model with dynamic weight adjustment for TAO price prediction."""
    
    def __init__(self):
        # Initialize individual models (removed LightGBM for faster predictions)
        self.models = {
            'ets': ETSPredictor(),
            'svr': SVRPredictor(),
            'kalman': KalmanPredictor(),
            'random_forest': RandomForestPredictor(),
            'xgboost': RandomForestPredictor(),  # Using RF as XGBoost placeholder for now
            'lstm': LSTMPredictor()
        }
        
        # Ensemble weights from config
        self.weights = Config.ENSEMBLE_WEIGHTS.copy()
        self.initial_weights = Config.ENSEMBLE_WEIGHTS.copy()
        
        # Dynamic weight adjustment
        self.enable_dynamic_weights = Config.ENABLE_DYNAMIC_WEIGHTS
        self.weight_adjustment_window = Config.WEIGHT_ADJUSTMENT_WINDOW
        self.weight_learning_rate = Config.WEIGHT_LEARNING_RATE
        self.xgboost_min_weight = Config.XGBOOST_MIN_WEIGHT
        
        # Track model performance for dynamic weights
        self.model_errors = {model: deque(maxlen=self.weight_adjustment_window) for model in self.models.keys()}
        self.model_accuracies = {model: deque(maxlen=self.weight_adjustment_window) for model in self.models.keys()}
        
        # Feature engineering and preprocessing
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = DataPreprocessor()
        
        # Training state
        self.is_trained = False
        self.training_scores = {}
        self.model_contributions = {}
        self.last_training_data = None
        
        # Volatility detection
        self.volatility_threshold = 0.05  # 5% price change threshold
        self.rapid_movement_detected = False
        
    def prepare_data_for_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare data for different model types.
        
        Args:
            df: Raw price data
            
        Returns:
            Dictionary with prepared data for each model type
        """
        try:
            logger.info("Preparing data for ensemble models...")
            
            # Create features
            features_df = self.feature_engineer.create_all_features(df)
            
            if len(features_df) == 0:
                logger.error("Feature engineering failed - no features created")
                return {}
            
            logger.info(f"Features created successfully: {len(features_df)} rows, {len(features_df.columns)} columns")
            
            # Clean data
            clean_df = self.preprocessor.clean_data_for_training(features_df)
            
            if len(clean_df) == 0:
                logger.error("Feature engineering failed - all data removed during cleaning")
                return {}
            
            logger.info(f"Data cleaning successful: {len(clean_df)} rows remaining")
            
            # Prepare data for different model types
            prepared_data = {
                'raw_df': df,
                'features_df': features_df,
                'clean_df': clean_df,
                'ets_data': clean_df,  # ETS uses clean data directly
                'sklearn_data': clean_df,  # SVR, RF, LightGBM use clean data
                'lstm_data': clean_df  # LSTM uses clean data with sequences
            }
            
            logger.info(f"Data prepared: {len(clean_df)} samples, {len(clean_df.columns)} features")
            return prepared_data
            
        except Exception as e:
            logger.error(f"Failed to prepare ensemble data: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return {}
    
    def train_individual_models(self, prepared_data: Dict[str, Any]) -> Dict[str, bool]:
        """
        Train all individual models.
        
        Args:
            prepared_data: Prepared data for models
            
        Returns:
            Dictionary with training success status for each model
        """
        training_results = {}
        
        try:
            # Verify we have enough data
            if len(prepared_data['clean_df']) < 100:
                logger.error(f"Insufficient data for training: only {len(prepared_data['clean_df'])} samples")
                return {name: False for name in self.models.keys()}
            
            logger.info(f"Training models with {len(prepared_data['clean_df'])} samples")
            
            # Train ETS model
            try:
                logger.info("Training ETS model...")
                training_results['ets'] = self.models['ets'].train(prepared_data['ets_data'])
                if training_results['ets']:
                    logger.info("✅ ETS model trained successfully")
                else:
                    logger.warning("❌ ETS model training failed")
            except Exception as e:
                logger.error(f"❌ ETS model training error: {e}")
                training_results['ets'] = False
            
            # Train SVR model
            try:
                logger.info("Training SVR model...")
                training_results['svr'] = self.models['svr'].train(prepared_data['sklearn_data'], optimize_params=False)
                if training_results['svr']:
                    logger.info("✅ SVR model trained successfully")
                else:
                    logger.warning("❌ SVR model training failed")
            except Exception as e:
                logger.error(f"❌ SVR model training error: {e}")
                training_results['svr'] = False
            
            # Train Kalman model
            try:
                logger.info("Training Kalman model...")
                training_results['kalman'] = self.models['kalman'].train(prepared_data['sklearn_data'])
                if training_results['kalman']:
                    logger.info("✅ Kalman model trained successfully")
                else:
                    logger.warning("❌ Kalman model training failed")
            except Exception as e:
                logger.error(f"❌ Kalman model training error: {e}")
                training_results['kalman'] = False
            
            # Train Random Forest model
            try:
                logger.info("Training Random Forest model...")
                training_results['random_forest'] = self.models['random_forest'].train(prepared_data['sklearn_data'], optimize_params=False)
                if training_results['random_forest']:
                    logger.info("✅ Random Forest model trained successfully")
                else:
                    logger.warning("❌ Random Forest model training failed")
            except Exception as e:
                logger.error(f"❌ Random Forest model training error: {e}")
                training_results['random_forest'] = False
            
            # Train XGBoost model (using Random Forest as placeholder)
            try:
                logger.info("Training XGBoost model...")
                training_results['xgboost'] = self.models['xgboost'].train(prepared_data['sklearn_data'], optimize_params=False)
                if training_results['xgboost']:
                    logger.info("✅ XGBoost model trained successfully")
                else:
                    logger.warning("❌ XGBoost model training failed")
            except Exception as e:
                logger.error(f"❌ XGBoost model training error: {e}")
                training_results['xgboost'] = False
            
            # Train LSTM model
            try:
                logger.info("Training LSTM model...")
                training_results['lstm'] = self.models['lstm'].train(prepared_data['lstm_data'])
                if training_results['lstm']:
                    logger.info("✅ LSTM model trained successfully")
                else:
                    logger.warning("❌ LSTM model training failed")
            except Exception as e:
            
            # Log training results
            successful_models = [name for name, success in training_results.items() if success]
            failed_models = [name for name, success in training_results.items() if not success]
            
            logger.info(f"Successfully trained models: {successful_models}")
            if failed_models:
                logger.warning(f"Failed to train models: {failed_models}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Failed to train individual models: {e}")
            return {name: False for name in self.models.keys()}
    
    def train(self, df: pd.DataFrame) -> bool:
        """
        Train the ensemble model.
        
        Args:
            df: DataFrame with historical price data
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            logger.info("Training ensemble model...")
            
            # Prepare data
            prepared_data = self.prepare_data_for_models(df)
            
            if not prepared_data:
                logger.error("Data preparation failed")
                return False
            
            # Store training data
            self.last_training_data = prepared_data['clean_df'].copy()
            
            # Train individual models
            training_results = self.train_individual_models(prepared_data)
            
            # Check if at least 3 models trained successfully
            successful_count = sum(training_results.values())
            
            if successful_count < 3:
                logger.error(f"Only {successful_count} models trained successfully. Need at least 3.")
                return False
            
            # Adjust weights for failed models
            self.adjust_weights_for_failed_models(training_results)
            
            # Calculate training scores
            self.calculate_training_scores(prepared_data['clean_df'])
            
            self.is_trained = True
            
            logger.info(f"Ensemble model trained successfully with {successful_count}/5 models")
            logger.info(f"Adjusted weights: {self.weights}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            return False
    
    def adjust_weights_dynamically(self):
        """
        Dynamically adjust model weights based on recent performance.
        Better performing models get higher weights.
        """
        try:
            if not self.enable_dynamic_weights:
                return
            
            # Calculate performance scores for each model
            performance_scores = {}
            
            for model_name in self.models.keys():
                if len(self.model_errors[model_name]) < 5:
                    # Not enough data yet, use initial weight
                    performance_scores[model_name] = self.initial_weights.get(model_name, 0.1)
                    continue
                
                # Calculate inverse error (lower error = higher score)
                avg_error = np.mean(list(self.model_errors[model_name]))
                if avg_error > 0:
                    performance_scores[model_name] = 1.0 / (1.0 + avg_error)
                else:
                    performance_scores[model_name] = 1.0
            
            # Normalize scores to weights
            total_score = sum(performance_scores.values())
            if total_score > 0:
                new_weights = {model: score / total_score for model, score in performance_scores.items()}
                
                # Ensure XGBoost maintains minimum weight
                if 'xgboost' in new_weights and new_weights['xgboost'] < self.xgboost_min_weight:
                    # Redistribute weights to ensure XGBoost gets minimum
                    deficit = self.xgboost_min_weight - new_weights['xgboost']
                    new_weights['xgboost'] = self.xgboost_min_weight
                    
                    # Reduce other weights proportionally
                    other_models = [m for m in new_weights.keys() if m != 'xgboost']
                    other_total = sum(new_weights[m] for m in other_models)
                    
                    if other_total > 0:
                        reduction_factor = (1.0 - self.xgboost_min_weight) / other_total
                        for model in other_models:
                            new_weights[model] *= reduction_factor
                
                # Smooth transition using learning rate
                for model_name in self.weights.keys():
                    if model_name in new_weights:
                        old_weight = self.weights[model_name]
                        new_weight = new_weights[model_name]
                        self.weights[model_name] = old_weight + self.weight_learning_rate * (new_weight - old_weight)
                
                logger.info(f"Dynamic weights updated: {[(k, f'{v:.3f}') for k, v in self.weights.items()]}")
            
        except Exception as e:
            logger.error(f"Failed to adjust weights dynamically: {e}")
    
    def update_model_performance(self, model_name: str, predicted: float, actual: float):
        """
        Update performance tracking for a model.
        
        Args:
            model_name: Name of the model
            predicted: Predicted price
            actual: Actual price
        """
        try:
            error = abs(predicted - actual)
            accuracy = 1.0 - min(error / actual, 1.0) if actual > 0 else 0.0
            
            self.model_errors[model_name].append(error)
            self.model_accuracies[model_name].append(accuracy)
            
        except Exception as e:
            logger.error(f"Failed to update model performance: {e}")
    
    def adjust_weights_for_failed_models(self, training_results: Dict[str, bool]):
        """
        Adjust ensemble weights for models that failed to train.
        
        Args:
            training_results: Dictionary with training success status
        """
        try:
            # Set weights to 0 for failed models
            failed_models = [name for name, success in training_results.items() if not success]
            
            for model_name in failed_models:
                self.weights[model_name] = 0.0
            
            # Ensure XGBoost maintains minimum weight if it trained successfully
            if 'xgboost' in training_results and training_results['xgboost']:
                if self.weights['xgboost'] < self.xgboost_min_weight:
                    self.weights['xgboost'] = self.xgboost_min_weight
            
            # Renormalize weights
            total_weight = sum(self.weights.values())
            
            if total_weight > 0:
                for model_name in self.weights:
                    self.weights[model_name] /= total_weight
            
            logger.info(f"Weights adjusted for failed models: {failed_models}")
            
        except Exception as e:
            logger.error(f"Failed to adjust weights: {e}")
    
    def predict(self, df: pd.DataFrame) -> Tuple[float, Tuple[float, float]]:
        """
        Make ensemble prediction.
        
        Args:
            df: DataFrame with current data for prediction
            
        Returns:
            Tuple of (prediction, confidence_interval)
        """
        try:
            if not self.is_trained:
                logger.error("Ensemble model not trained")
                return 0.0, (0.0, 0.0)
            
            # Prepare data
            prepared_data = self.prepare_data_for_models(df)
            
            if not prepared_data:
                logger.error("Data preparation failed for prediction")
                return 0.0, (0.0, 0.0)
            
            # Get predictions from individual models
            predictions = {}
            confidence_intervals = {}
            
            for model_name, model in self.models.items():
                if self.weights[model_name] > 0:  # Only predict with trained models
                    try:
                        if model_name == 'ets':
                            pred, conf = model.predict()
                        else:
                            pred, conf = model.predict(prepared_data['sklearn_data'])
                        
                        predictions[model_name] = pred
                        confidence_intervals[model_name] = conf
                        
                    except Exception as e:
                        logger.warning(f"Prediction failed for {model_name}: {e}")
                        predictions[model_name] = 0.0
                        confidence_intervals[model_name] = (0.0, 0.0)
            
            if not predictions:
                logger.error("No model predictions available")
                return 0.0, (0.0, 0.0)
            
            # Calculate weighted ensemble prediction
            ensemble_prediction = 0.0
            total_weight = 0.0
            
            for model_name, pred in predictions.items():
                weight = self.weights[model_name]
                ensemble_prediction += weight * pred
                total_weight += weight
            
            if total_weight > 0:
                ensemble_prediction /= total_weight
            
            # Calculate ensemble confidence interval
            lower_bounds = []
            upper_bounds = []
            
            for model_name, (lower, upper) in confidence_intervals.items():
                if self.weights[model_name] > 0:
                    lower_bounds.append(lower)
                    upper_bounds.append(upper)
            
            if lower_bounds and upper_bounds:
                ensemble_lower = np.mean(lower_bounds)
                ensemble_upper = np.mean(upper_bounds)
            else:
                # Fallback confidence interval
                margin = abs(ensemble_prediction) * 0.05
                ensemble_lower = ensemble_prediction - margin
                ensemble_upper = ensemble_prediction + margin
            
            # Store model contributions for analysis
            self.model_contributions = predictions.copy()
            
            # Detect rapid price movements
            self.detect_rapid_movements(df, ensemble_prediction)
            
            logger.debug(f"Ensemble prediction: {ensemble_prediction:.2f} [{ensemble_lower:.2f}, {ensemble_upper:.2f}]")
            logger.debug(f"Model contributions: {[(k, f'{v:.2f}') for k, v in predictions.items()]}")
            
            return ensemble_prediction, (ensemble_lower, ensemble_upper)
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return 0.0, (0.0, 0.0)
    
    def detect_rapid_movements(self, df: pd.DataFrame, predicted_price: float):
        """
        Detect rapid rise/fall in BTC price.
        
        Args:
            df: Current price data
            predicted_price: Predicted price
        """
        try:
            if len(df) < 2:
                return
            
            current_price = df['close'].iloc[-1]
            previous_price = df['close'].iloc[-2]
            
            # Calculate price change percentage
            price_change = abs(predicted_price - current_price) / current_price
            recent_change = abs(current_price - previous_price) / previous_price
            
            # Detect rapid movement
            if price_change > self.volatility_threshold or recent_change > self.volatility_threshold:
                self.rapid_movement_detected = True
                movement_type = "rise" if predicted_price > current_price else "fall"
                logger.warning(f"Rapid {movement_type} detected: {price_change*100:.2f}% predicted change")
            else:
                self.rapid_movement_detected = False
                
        except Exception as e:
            logger.error(f"Failed to detect rapid movements: {e}")
    
    def calculate_training_scores(self, df: pd.DataFrame):
        """
        Calculate training scores for the ensemble.
        
        Args:
            df: Training data
        """
        try:
            # Get individual model scores
            for model_name, model in self.models.items():
                if hasattr(model, 'training_score') and model.training_score:
                    self.training_scores[model_name] = model.training_score
            
            logger.debug(f"Training scores collected: {list(self.training_scores.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to calculate training scores: {e}")
    
    def update_models(self, new_data: pd.DataFrame) -> bool:
        """
        Update ensemble models with new data.
        
        Args:
            new_data: New price data
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if not self.is_trained:
                logger.warning("Ensemble not trained, performing full training")
                return self.train(new_data)
            
            logger.info("Updating ensemble models with new data...")
            
            # Prepare data
            prepared_data = self.prepare_data_for_models(new_data)
            
            if not prepared_data:
                logger.error("Data preparation failed for update")
                return False
            
            # Update individual models
            update_results = {}
            
            for model_name, model in self.models.items():
                if self.weights[model_name] > 0:  # Only update trained models
                    try:
                        if model_name == 'ets':
                            update_results[model_name] = model.update_model(prepared_data['ets_data'])
                        else:
                            update_results[model_name] = model.update_model(prepared_data['sklearn_data'])
                    except Exception as e:
                        logger.warning(f"Update failed for {model_name}: {e}")
                        update_results[model_name] = False
            
            successful_updates = sum(update_results.values())
            logger.info(f"Successfully updated {successful_updates} models")
            
            return successful_updates > 0
            
        except Exception as e:
            logger.error(f"Ensemble update failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the ensemble model.
        
        Returns:
            Dictionary with ensemble information
        """
        info = {
            'model_type': 'Ensemble',
            'is_trained': self.is_trained,
            'weights': self.weights.copy(),
            'individual_models': {},
            'rapid_movement_detected': self.rapid_movement_detected,
            'last_contributions': self.model_contributions.copy()
        }
        
        # Get info from individual models
        for model_name, model in self.models.items():
            if hasattr(model, 'get_model_info'):
                info['individual_models'][model_name] = model.get_model_info()
        
        return info
    
    def save_ensemble(self, filepath: str) -> bool:
        """
        Save the entire ensemble to file.
        
        Args:
            filepath: Base filepath for saving
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if not self.is_trained:
                logger.error("Cannot save untrained ensemble")
                return False
            
            # Save individual models
            models_dir = os.path.dirname(filepath)
            saved_models = {}
            
            for model_name, model in self.models.items():
                if self.weights[model_name] > 0:
                    model_path = os.path.join(models_dir, f"{model_name}_model.joblib")
                    if model.save_model(model_path):
                        saved_models[model_name] = model_path
            
            # Save ensemble metadata
            ensemble_data = {
                'weights': self.weights,
                'is_trained': self.is_trained,
                'training_scores': self.training_scores,
                'saved_models': saved_models,
                'volatility_threshold': self.volatility_threshold
            }
            
            joblib.dump(ensemble_data, filepath)
            logger.info(f"Ensemble saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save ensemble: {e}")
            return False
    
    def load_ensemble(self, filepath: str) -> bool:
        """
        Load the entire ensemble from file.
        
        Args:
            filepath: Base filepath for loading
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Ensemble file not found: {filepath}")
                return False
            
            # Load ensemble metadata
            ensemble_data = joblib.load(filepath)
            
            self.weights = ensemble_data['weights']
            self.is_trained = ensemble_data['is_trained']
            self.training_scores = ensemble_data.get('training_scores', {})
            self.volatility_threshold = ensemble_data.get('volatility_threshold', 0.05)
            
            # Load individual models
            saved_models = ensemble_data.get('saved_models', {})
            
            for model_name, model_path in saved_models.items():
                if model_name in self.models:
                    if not self.models[model_name].load_model(model_path):
                        logger.warning(f"Failed to load {model_name} model")
                        self.weights[model_name] = 0.0
            
            logger.info(f"Ensemble loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")
            return False