"""Ensemble model combining ETS, SVR, Random Forest, LightGBM, and LSTM for BTC prediction."""
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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from models.ets_model import ETSPredictor
from models.svr_model import SVRPredictor
from models.random_forest_model import RandomForestPredictor
from models.lightgbm_model import LightGBMPredictor
from models.lstm_model import LSTMPredictor
from services.feature_engineering import FeatureEngineer
from services.preprocessing import DataPreprocessor

class EnsemblePredictor:
    """Ensemble model combining multiple ML algorithms for BTC price prediction."""
    
    def __init__(self):
        # Initialize individual models
        self.models = {
            'ets': ETSPredictor(),
            'svr': SVRPredictor(),
            'random_forest': RandomForestPredictor(),
            'lightgbm': LightGBMPredictor(),
            'lstm': LSTMPredictor()
        }
        
        # Ensemble weights from config
        self.weights = Config.ENSEMBLE_WEIGHTS.copy()
        
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
                logger.error("Feature engineering failed")
                return {}
            
            # Clean data
            clean_df = self.preprocessor.clean_data_for_training(features_df)
            
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
            # Train ETS model
            logger.info("Training ETS model...")
            training_results['ets'] = self.models['ets'].train(prepared_data['ets_data'])
            
            # Train SVR model
            logger.info("Training SVR model...")
            training_results['svr'] = self.models['svr'].train(prepared_data['sklearn_data'], optimize_params=False)
            
            # Train Random Forest model
            logger.info("Training Random Forest model...")
            training_results['random_forest'] = self.models['random_forest'].train(prepared_data['sklearn_data'], optimize_params=False)
            
            # Train LightGBM model
            logger.info("Training LightGBM model...")
            training_results['lightgbm'] = self.models['lightgbm'].train(prepared_data['sklearn_data'], optimize_params=False)
            
            # Train LSTM model
            logger.info("Training LSTM model...")
            training_results['lstm'] = self.models['lstm'].train(prepared_data['lstm_data'])
            
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