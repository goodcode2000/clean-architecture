"""Offset correction system for reducing prediction bias and improving accuracy."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import os
import sys
from datetime import datetime, timedelta
from collections import deque
import statistics

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class OffsetCorrectionSystem:
    """Analyzes prediction errors and applies corrections to reduce systematic bias."""
    
    def __init__(self):
        self.error_window_days = Config.ERROR_ANALYSIS_WINDOW_DAYS
        self.min_predictions = Config.MIN_PREDICTIONS_FOR_CORRECTION
        
        # Error tracking
        self.prediction_errors = {
            'ensemble': deque(maxlen=1000),
            'garch': deque(maxlen=1000),
            'svr': deque(maxlen=1000),
            'random_forest': deque(maxlen=1000),
            'lightgbm': deque(maxlen=1000),
            'lstm': deque(maxlen=1000)
        }
        
        # Bias corrections
        self.bias_corrections = {
            'ensemble': 0.0,
            'garch': 0.0,
            'svr': 0.0,
            'random_forest': 0.0,
            'lightgbm': 0.0,
            'lstm': 0.0
        }
        
        # Adaptive learning rates
        self.learning_rates = {
            'ensemble': 0.1,
            'garch': 0.05,
            'svr': 0.1,
            'random_forest': 0.1,
            'lightgbm': 0.1,
            'lstm': 0.05
        }
        
        # Market volatility tracking
        self.volatility_history = deque(maxlen=100)
        self.current_volatility_regime = 'normal'  # 'low', 'normal', 'high'
        
        # Performance metrics
        self.accuracy_metrics = {}
        self.correction_history = []
        
    def add_prediction_error(self, model_name: str, predicted: float, actual: float, 
                           timestamp: datetime = None) -> None:
        """
        Add a prediction error for analysis.
        
        Args:
            model_name: Name of the model ('ensemble', 'ets', etc.)
            predicted: Predicted value
            actual: Actual observed value
            timestamp: When the prediction was made
        """
        try:
            if model_name not in self.prediction_errors:
                logger.warning(f"Unknown model name: {model_name}")
                return
            
            # Calculate error (predicted - actual)
            error = predicted - actual
            error_percentage = (error / actual) * 100 if actual != 0 else 0
            
            # Store error with metadata
            error_data = {
                'error': error,
                'error_percentage': error_percentage,
                'predicted': predicted,
                'actual': actual,
                'timestamp': timestamp or datetime.now(),
                'abs_error': abs(error),
                'abs_error_percentage': abs(error_percentage)
            }
            
            self.prediction_errors[model_name].append(error_data)
            
            # Update volatility tracking
            if len(self.volatility_history) > 0:
                price_change = abs(actual - self.volatility_history[-1]) / self.volatility_history[-1]
                self.volatility_history.append(actual)
                self.update_volatility_regime(price_change)
            else:
                self.volatility_history.append(actual)
            
            logger.debug(f"Added error for {model_name}: {error:.2f} ({error_percentage:.2f}%)")
            
        except Exception as e:
            logger.error(f"Failed to add prediction error: {e}")
    
    def update_volatility_regime(self, price_change: float) -> None:
        """
        Update the current market volatility regime.
        
        Args:
            price_change: Recent price change percentage
        """
        try:
            # Calculate recent volatility
            if len(self.volatility_history) >= 20:
                recent_prices = list(self.volatility_history)[-20:]
                price_changes = [abs(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                               for i in range(1, len(recent_prices))]
                avg_volatility = np.mean(price_changes)
                
                # Classify volatility regime
                if avg_volatility < 0.01:  # Less than 1% average change
                    new_regime = 'low'
                elif avg_volatility > 0.05:  # More than 5% average change
                    new_regime = 'high'
                else:
                    new_regime = 'normal'
                
                if new_regime != self.current_volatility_regime:
                    logger.info(f"Volatility regime changed: {self.current_volatility_regime} -> {new_regime}")
                    self.current_volatility_regime = new_regime
                    self.adjust_learning_rates_for_volatility()
            
        except Exception as e:
            logger.error(f"Failed to update volatility regime: {e}")
    
    def adjust_learning_rates_for_volatility(self) -> None:
        """Adjust learning rates based on market volatility."""
        try:
            base_rates = {
                'ensemble': 0.1,
                'garch': 0.05,
                'svr': 0.1,
                'random_forest': 0.1,
                'lightgbm': 0.1,
                'lstm': 0.05
            }
            
            # Adjust rates based on volatility
            if self.current_volatility_regime == 'low':
                # Lower learning rates in stable markets
                multiplier = 0.5
            elif self.current_volatility_regime == 'high':
                # Higher learning rates in volatile markets
                multiplier = 1.5
            else:
                multiplier = 1.0
            
            for model_name in self.learning_rates:
                self.learning_rates[model_name] = base_rates[model_name] * multiplier
            
            logger.debug(f"Learning rates adjusted for {self.current_volatility_regime} volatility")
            
        except Exception as e:
            logger.error(f"Failed to adjust learning rates: {e}")
    
    def calculate_bias_correction(self, model_name: str) -> float:
        """
        Calculate bias correction for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Bias correction value
        """
        try:
            if model_name not in self.prediction_errors:
                return 0.0
            
            errors = self.prediction_errors[model_name]
            
            if len(errors) < self.min_predictions:
                logger.debug(f"Insufficient data for {model_name} bias correction: {len(errors)} < {self.min_predictions}")
                return 0.0
            
            # Get recent errors within the analysis window
            cutoff_time = datetime.now() - timedelta(days=self.error_window_days)
            recent_errors = [e for e in errors if e['timestamp'] >= cutoff_time]
            
            if len(recent_errors) < self.min_predictions:
                logger.debug(f"Insufficient recent data for {model_name} bias correction")
                return 0.0
            
            # Calculate systematic bias
            error_values = [e['error'] for e in recent_errors]
            
            # Use median for robustness against outliers
            systematic_bias = statistics.median(error_values)
            
            # Apply exponential smoothing to current correction
            learning_rate = self.learning_rates[model_name]
            current_correction = self.bias_corrections[model_name]
            
            # Update correction with adaptive learning
            new_correction = current_correction + learning_rate * (-systematic_bias)
            
            logger.debug(f"Bias correction for {model_name}: {current_correction:.4f} -> {new_correction:.4f}")
            
            return new_correction
            
        except Exception as e:
            logger.error(f"Failed to calculate bias correction for {model_name}: {e}")
            return 0.0
    
    def update_all_corrections(self) -> Dict[str, float]:
        """
        Update bias corrections for all models.
        
        Returns:
            Dictionary with updated corrections
        """
        try:
            logger.info("Updating bias corrections for all models...")
            
            updated_corrections = {}
            
            for model_name in self.bias_corrections:
                new_correction = self.calculate_bias_correction(model_name)
                self.bias_corrections[model_name] = new_correction
                updated_corrections[model_name] = new_correction
            
            # Log correction update
            correction_summary = {
                'timestamp': datetime.now(),
                'corrections': updated_corrections.copy(),
                'volatility_regime': self.current_volatility_regime,
                'learning_rates': self.learning_rates.copy()
            }
            
            self.correction_history.append(correction_summary)
            
            # Keep only recent correction history
            if len(self.correction_history) > 100:
                self.correction_history = self.correction_history[-100:]
            
            logger.info(f"Bias corrections updated: {[(k, f'{v:.4f}') for k, v in updated_corrections.items()]}")
            
            return updated_corrections
            
        except Exception as e:
            logger.error(f"Failed to update corrections: {e}")
            return {}
    
    def apply_correction(self, model_name: str, prediction: float) -> float:
        """
        Apply bias correction to a prediction.
        
        Args:
            model_name: Name of the model
            prediction: Original prediction
            
        Returns:
            Corrected prediction
        """
        try:
            if model_name not in self.bias_corrections:
                logger.warning(f"No correction available for model: {model_name}")
                return prediction
            
            correction = self.bias_corrections[model_name]
            corrected_prediction = prediction + correction
            
            logger.debug(f"Applied correction to {model_name}: {prediction:.2f} -> {corrected_prediction:.2f}")
            
            return corrected_prediction
            
        except Exception as e:
            logger.error(f"Failed to apply correction: {e}")
            return prediction
    
    def calculate_accuracy_metrics(self, model_name: str = None) -> Dict[str, Any]:
        """
        Calculate accuracy metrics for models.
        
        Args:
            model_name: Specific model name, or None for all models
            
        Returns:
            Dictionary with accuracy metrics
        """
        try:
            if model_name:
                models_to_analyze = [model_name]
            else:
                models_to_analyze = list(self.prediction_errors.keys())
            
            metrics = {}
            
            for name in models_to_analyze:
                if name not in self.prediction_errors:
                    continue
                
                errors = self.prediction_errors[name]
                
                if len(errors) == 0:
                    continue
                
                # Get recent errors
                cutoff_time = datetime.now() - timedelta(days=self.error_window_days)
                recent_errors = [e for e in errors if e['timestamp'] >= cutoff_time]
                
                if len(recent_errors) == 0:
                    continue
                
                # Calculate metrics
                error_values = [e['error'] for e in recent_errors]
                abs_errors = [e['abs_error'] for e in recent_errors]
                abs_error_percentages = [e['abs_error_percentage'] for e in recent_errors]
                
                model_metrics = {
                    'n_predictions': len(recent_errors),
                    'mean_error': np.mean(error_values),
                    'median_error': np.median(error_values),
                    'std_error': np.std(error_values),
                    'mae': np.mean(abs_errors),
                    'median_ae': np.median(abs_errors),
                    'mape': np.mean(abs_error_percentages),
                    'median_ape': np.median(abs_error_percentages),
                    'rmse': np.sqrt(np.mean([e**2 for e in error_values])),
                    'bias_correction': self.bias_corrections[name],
                    'learning_rate': self.learning_rates[name]
                }
                
                # Directional accuracy
                if len(recent_errors) > 1:
                    correct_directions = 0
                    for i in range(1, len(recent_errors)):
                        actual_direction = np.sign(recent_errors[i]['actual'] - recent_errors[i-1]['actual'])
                        pred_direction = np.sign(recent_errors[i]['predicted'] - recent_errors[i-1]['predicted'])
                        if actual_direction == pred_direction:
                            correct_directions += 1
                    
                    model_metrics['directional_accuracy'] = correct_directions / (len(recent_errors) - 1) * 100
                else:
                    model_metrics['directional_accuracy'] = 0.0
                
                metrics[name] = model_metrics
            
            self.accuracy_metrics = metrics
            
            logger.info(f"Accuracy metrics calculated for {len(metrics)} models")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate accuracy metrics: {e}")
            return {}
    
    def get_correction_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze the effectiveness of bias corrections.
        
        Returns:
            Dictionary with correction effectiveness metrics
        """
        try:
            effectiveness = {}
            
            for model_name in self.prediction_errors:
                errors = self.prediction_errors[model_name]
                
                if len(errors) < 20:  # Need sufficient data
                    continue
                
                # Split errors into before/after correction periods
                mid_point = len(errors) // 2
                early_errors = list(errors)[:mid_point]
                recent_errors = list(errors)[mid_point:]
                
                if len(early_errors) == 0 or len(recent_errors) == 0:
                    continue
                
                # Calculate improvement metrics
                early_mae = np.mean([e['abs_error'] for e in early_errors])
                recent_mae = np.mean([e['abs_error'] for e in recent_errors])
                
                early_bias = np.mean([e['error'] for e in early_errors])
                recent_bias = np.mean([e['error'] for e in recent_errors])
                
                mae_improvement = (early_mae - recent_mae) / early_mae * 100 if early_mae != 0 else 0
                bias_reduction = (abs(early_bias) - abs(recent_bias)) / abs(early_bias) * 100 if early_bias != 0 else 0
                
                effectiveness[model_name] = {
                    'mae_improvement_percent': mae_improvement,
                    'bias_reduction_percent': bias_reduction,
                    'early_mae': early_mae,
                    'recent_mae': recent_mae,
                    'early_bias': early_bias,
                    'recent_bias': recent_bias,
                    'current_correction': self.bias_corrections[model_name]
                }
            
            logger.info(f"Correction effectiveness analyzed for {len(effectiveness)} models")
            return effectiveness
            
        except Exception as e:
            logger.error(f"Failed to analyze correction effectiveness: {e}")
            return {}
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the offset correction system.
        
        Returns:
            Dictionary with system status
        """
        try:
            status = {
                'current_corrections': self.bias_corrections.copy(),
                'learning_rates': self.learning_rates.copy(),
                'volatility_regime': self.current_volatility_regime,
                'error_counts': {name: len(errors) for name, errors in self.prediction_errors.items()},
                'min_predictions_threshold': self.min_predictions,
                'analysis_window_days': self.error_window_days,
                'correction_history_length': len(self.correction_history)
            }
            
            # Add recent accuracy metrics
            recent_metrics = self.calculate_accuracy_metrics()
            if recent_metrics:
                status['accuracy_metrics'] = recent_metrics
            
            # Add correction effectiveness
            effectiveness = self.get_correction_effectiveness()
            if effectiveness:
                status['correction_effectiveness'] = effectiveness
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    def reset_corrections(self, model_name: str = None) -> None:
        """
        Reset bias corrections for specified model or all models.
        
        Args:
            model_name: Specific model to reset, or None for all models
        """
        try:
            if model_name:
                if model_name in self.bias_corrections:
                    self.bias_corrections[model_name] = 0.0
                    logger.info(f"Reset correction for {model_name}")
            else:
                for name in self.bias_corrections:
                    self.bias_corrections[name] = 0.0
                logger.info("Reset all bias corrections")
                
        except Exception as e:
            logger.error(f"Failed to reset corrections: {e}")
    
    def save_correction_data(self, filepath: str) -> bool:
        """
        Save correction system data to file.
        
        Args:
            filepath: Path to save data
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            import joblib
            
            correction_data = {
                'bias_corrections': self.bias_corrections,
                'learning_rates': self.learning_rates,
                'current_volatility_regime': self.current_volatility_regime,
                'correction_history': self.correction_history,
                'accuracy_metrics': self.accuracy_metrics,
                'error_window_days': self.error_window_days,
                'min_predictions': self.min_predictions
            }
            
            joblib.dump(correction_data, filepath)
            logger.info(f"Correction data saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save correction data: {e}")
            return False
    
    def load_correction_data(self, filepath: str) -> bool:
        """
        Load correction system data from file.
        
        Args:
            filepath: Path to load data from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            import joblib
            
            if not os.path.exists(filepath):
                logger.warning(f"Correction data file not found: {filepath}")
                return False
            
            correction_data = joblib.load(filepath)
            
            self.bias_corrections = correction_data.get('bias_corrections', {})
            self.learning_rates = correction_data.get('learning_rates', {})
            self.current_volatility_regime = correction_data.get('current_volatility_regime', 'normal')
            self.correction_history = correction_data.get('correction_history', [])
            self.accuracy_metrics = correction_data.get('accuracy_metrics', {})
            self.error_window_days = correction_data.get('error_window_days', self.error_window_days)
            self.min_predictions = correction_data.get('min_predictions', self.min_predictions)
            
            logger.info(f"Correction data loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load correction data: {e}")
            return False