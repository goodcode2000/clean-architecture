"""
Attention Fusion Meta-Learner for Ensemble Model Combination
Combines ETS, GARCH, LightGBM, LSTM, and CNN models with online error weighting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import asyncio
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .base_model import BaseModel
from .ets_model import ETSPredictionModel
from .garch_model import GARCHPredictionModel
from .lightgbm_model import LightGBMPredictionModel
from .lstm_model import LSTMPredictionModel
from .cnn_model import CNNPredictionModel
from ..feature_engineering import FeatureSet
from ..data_models import Prediction

logger = logging.getLogger(__name__)

class AttentionFusionMetaLearner(BaseModel):
    """Meta-learner that combines ensemble models using attention fusion and online error weighting"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            # Meta-learner configuration
            'meta_model_type': 'ridge',  # 'ridge', 'random_forest', 'neural'
            'attention_mechanism': True,
            'online_weighting': True,
            'error_weighted_boosting': True,
            
            # Model weights initialization
            'initial_weights': {
                'ETS': 0.15,
                'GARCH': 0.15,
                'LightGBM': 0.25,
                'LSTM': 0.25,
                'CNN': 0.20
            },
            
            # Online learning parameters
            'learning_rate': 0.01,
            'weight_decay': 0.95,
            'min_weight': 0.05,
            'max_weight': 0.50,
            'adaptation_window': 50,
            
            # Error tracking
            'error_memory': 100,  # Number of recent errors to remember
            'error_threshold': 100.0,  # USD threshold for error weighting
            
            # Attention parameters
            'attention_temperature': 1.0,
            'attention_dropout': 0.1,
            
            # Rapid change detection
            'rapid_change_threshold': 50.0,  # USD threshold for rapid changes
            'rapid_change_boost': 1.5,  # Boost factor for rapid change models
            
            # Training parameters
            'min_predictions': 20,  # Minimum predictions before meta-learning
            'retrain_frequency': 24,  # Hours between meta-model retraining
        }
        
        if config:
            default_config.update(config)
        
        super().__init__("EnsembleMetaLearner", default_config)
        
        # Initialize base models
        self.base_models = {
            'ETS': ETSPredictionModel(),
            'GARCH': GARCHPredictionModel(),
            'LightGBM': LightGBMPredictionModel(),
            'LSTM': LSTMPredictionModel(),
            'CNN': CNNPredictionModel()
        }
        
        # Meta-learner attributes
        self.meta_model = None
        self.model_weights = self.config['initial_weights'].copy()
        self.error_history = {model_name: [] for model_name in self.base_models.keys()}
        self.prediction_history = []
        self.attention_weights = None
        self.performance_metrics = {}
        
    async def train(self, features: List[FeatureSet], targets: List[float]) -> bool:
        """Train all base models and meta-learner"""
        try:
            logger.info(f"Training ensemble meta-learner with {len(targets)} data points")
            
            # Train all base models
            training_results = {}
            for model_name, model in self.base_models.items():
                logger.info(f"Training {model_name} model...")
                success = await model.train(features, targets)
                training_results[model_name] = success
                
                if success:
                    logger.info(f"✓ {model_name} model trained successfully")
                else:
                    logger.warning(f"✗ {model_name} model training failed")
            
            # Check if at least some models trained successfully
            successful_models = sum(training_results.values())
            if successful_models < 2:
                logger.error("Insufficient models trained successfully for ensemble")
                return False
            
            # Generate predictions from all models for meta-learning
            base_predictions = await self._generate_base_predictions(features)
            
            if len(base_predictions) < self.config['min_predictions']:
                logger.warning("Insufficient predictions for meta-learning")
                # Use simple averaging as fallback
                self.is_trained = True
                return True
            
            # Train meta-model
            await self._train_meta_model(base_predictions, targets)
            
            # Calculate initial performance metrics
            self._calculate_performance_metrics(base_predictions, targets)
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            logger.info(f"Ensemble meta-learner trained with {successful_models}/{len(self.base_models)} base models")
            return True
            
        except Exception as e:
            logger.error(f"Ensemble meta-learner training failed: {e}")
            return False
    
    async def predict(self, features: FeatureSet) -> Tuple[float, float]:
        """Make ensemble prediction using attention fusion and online weighting"""
        try:
            if not self.is_trained:
                logger.warning("Ensemble meta-learner not trained")
                return 0.0, 0.0
            
            # Get predictions from all base models
            base_predictions = {}
            base_confidences = {}
            
            for model_name, model in self.base_models.items():
                if model.is_trained:
                    pred, conf = await model.predict(features)
                    base_predictions[model_name] = pred
                    base_confidences[model_name] = conf
                else:
                    base_predictions[model_name] = 0.0
                    base_confidences[model_name] = 0.0
            
            # Apply attention fusion
            if self.config['attention_mechanism']:
                ensemble_prediction, ensemble_confidence = self._attention_fusion_predict(
                    base_predictions, base_confidences, features
                )
            else:
                ensemble_prediction, ensemble_confidence = self._weighted_average_predict(
                    base_predictions, base_confidences
                )
            
            # Detect rapid changes and adjust prediction
            rapid_change_detected = self._detect_rapid_change(features)
            if rapid_change_detected:
                ensemble_prediction, ensemble_confidence = self._adjust_for_rapid_change(
                    ensemble_prediction, ensemble_confidence, base_predictions, base_confidences
                )
            
            # Store prediction for online learning
            prediction_record = {
                'timestamp': datetime.now(),
                'ensemble_prediction': ensemble_prediction,
                'ensemble_confidence': ensemble_confidence,
                'base_predictions': base_predictions.copy(),
                'base_confidences': base_confidences.copy(),
                'model_weights': self.model_weights.copy(),
                'rapid_change': rapid_change_detected
            }
            
            self.prediction_history.append(prediction_record)
            
            # Limit history size
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            logger.debug(f"Ensemble prediction: {ensemble_prediction:.2f}, confidence: {ensemble_confidence:.3f}")
            return ensemble_prediction, ensemble_confidence
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return 0.0, 0.0
    
    async def _generate_base_predictions(self, features: List[FeatureSet]) -> List[Dict[str, float]]:
        """Generate predictions from all base models for meta-learning"""
        try:
            base_predictions = []
            
            # Use a subset of features for cross-validation style prediction
            for i in range(len(features) - 50, len(features)):  # Last 50 points
                if i < 0:
                    continue
                
                predictions = {}
                for model_name, model in self.base_models.items():
                    if model.is_trained:
                        pred, _ = await model.predict(features[i])
                        predictions[model_name] = pred
                    else:
                        predictions[model_name] = 0.0
                
                base_predictions.append(predictions)
            
            return base_predictions
            
        except Exception as e:
            logger.error(f"Base predictions generation failed: {e}")
            return []
    
    async def _train_meta_model(self, base_predictions: List[Dict[str, float]], targets: List[float]):
        """Train meta-model to combine base predictions"""
        try:
            if len(base_predictions) == 0:
                return
            
            # Prepare training data for meta-model
            X_meta = []
            y_meta = targets[-len(base_predictions):]  # Align with predictions
            
            for pred_dict in base_predictions:
                pred_vector = [pred_dict.get(model_name, 0.0) for model_name in self.base_models.keys()]
                X_meta.append(pred_vector)
            
            X_meta = np.array(X_meta)
            y_meta = np.array(y_meta)
            
            # Train meta-model
            if self.config['meta_model_type'] == 'ridge':
                self.meta_model = Ridge(alpha=1.0, random_state=42)
            elif self.config['meta_model_type'] == 'random_forest':
                self.meta_model = RandomForestRegressor(n_estimators=50, random_state=42)
            else:
                # Default to ridge
                self.meta_model = Ridge(alpha=1.0, random_state=42)
            
            self.meta_model.fit(X_meta, y_meta)
            
            # Update feature importance (meta-model coefficients)
            if hasattr(self.meta_model, 'coef_'):
                coefficients = self.meta_model.coef_
                self.feature_importance = {}
                for i, model_name in enumerate(self.base_models.keys()):
                    self.feature_importance[model_name] = abs(coefficients[i])
            
            logger.info("Meta-model trained successfully")
            
        except Exception as e:
            logger.warning(f"Meta-model training failed: {e}")
            self.meta_model = None
    
    def _attention_fusion_predict(self, base_predictions: Dict[str, float], 
                                base_confidences: Dict[str, float], 
                                features: FeatureSet) -> Tuple[float, float]:
        """Make prediction using attention fusion mechanism"""
        try:
            # Calculate attention weights based on model performance and confidence
            attention_scores = {}
            
            for model_name in base_predictions.keys():
                if model_name not in self.base_models or not self.base_models[model_name].is_trained:
                    attention_scores[model_name] = 0.0
                    continue
                
                # Base score from model confidence
                confidence_score = base_confidences.get(model_name, 0.0)
                
                # Historical performance score
                if model_name in self.error_history and len(self.error_history[model_name]) > 0:
                    recent_errors = self.error_history[model_name][-10:]  # Last 10 errors
                    avg_error = np.mean(recent_errors)
                    performance_score = max(0.1, 1.0 - min(avg_error / 100.0, 0.9))  # Normalize to 0.1-1.0
                else:
                    performance_score = 0.5
                
                # Current model weight
                weight_score = self.model_weights.get(model_name, 0.2)
                
                # Combine scores
                attention_scores[model_name] = confidence_score * performance_score * weight_score
            
            # Apply softmax to get attention weights
            attention_values = list(attention_scores.values())
            if sum(attention_values) > 0:
                attention_exp = np.exp(np.array(attention_values) / self.config['attention_temperature'])
                attention_weights = attention_exp / np.sum(attention_exp)
            else:
                attention_weights = np.ones(len(attention_values)) / len(attention_values)
            
            # Store attention weights
            self.attention_weights = {}
            for i, model_name in enumerate(attention_scores.keys()):
                self.attention_weights[model_name] = attention_weights[i]
            
            # Calculate weighted prediction
            weighted_prediction = 0.0
            weighted_confidence = 0.0
            
            for i, (model_name, prediction) in enumerate(base_predictions.items()):
                weight = attention_weights[i]
                weighted_prediction += weight * prediction
                weighted_confidence += weight * base_confidences.get(model_name, 0.0)
            
            return weighted_prediction, weighted_confidence
            
        except Exception as e:
            logger.error(f"Attention fusion failed: {e}")
            return self._weighted_average_predict(base_predictions, base_confidences)
    
    def _weighted_average_predict(self, base_predictions: Dict[str, float], 
                                base_confidences: Dict[str, float]) -> Tuple[float, float]:
        """Make prediction using weighted average"""
        try:
            total_weight = 0.0
            weighted_prediction = 0.0
            weighted_confidence = 0.0
            
            for model_name, prediction in base_predictions.items():
                if model_name in self.model_weights and self.base_models[model_name].is_trained:
                    weight = self.model_weights[model_name]
                    total_weight += weight
                    weighted_prediction += weight * prediction
                    weighted_confidence += weight * base_confidences.get(model_name, 0.0)
            
            if total_weight > 0:
                weighted_prediction /= total_weight
                weighted_confidence /= total_weight
            else:
                # Fallback to simple average
                valid_predictions = [p for p in base_predictions.values() if p > 0]
                valid_confidences = [base_confidences.get(name, 0.0) for name, p in base_predictions.items() if p > 0]
                
                if valid_predictions:
                    weighted_prediction = np.mean(valid_predictions)
                    weighted_confidence = np.mean(valid_confidences)
                else:
                    weighted_prediction = 0.0
                    weighted_confidence = 0.0
            
            return weighted_prediction, weighted_confidence
            
        except Exception as e:
            logger.error(f"Weighted average prediction failed: {e}")
            return 0.0, 0.0
    
    def _detect_rapid_change(self, features: FeatureSet) -> bool:
        """Detect if rapid price change is likely"""
        try:
            # Check recent volatility
            volatility = features.volatility_features.get('volatility_5d', 0.0)
            if volatility > 0.5:  # High volatility threshold
                return True
            
            # Check momentum indicators
            momentum = features.micro_features.get('momentum_5', 0.0)
            if abs(momentum) > 0.02:  # 2% momentum threshold
                return True
            
            # Check recent jump detection
            recent_jump = features.volatility_features.get('recent_jump', 0.0)
            if recent_jump > 0:
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Rapid change detection failed: {e}")
            return False
    
    def _adjust_for_rapid_change(self, prediction: float, confidence: float,
                               base_predictions: Dict[str, float], 
                               base_confidences: Dict[str, float]) -> Tuple[float, float]:
        """Adjust prediction for rapid change scenarios"""
        try:
            # Boost models that are better at rapid change detection
            rapid_change_models = ['GARCH', 'CNN', 'LightGBM']  # Models good at volatility/patterns
            
            adjusted_prediction = 0.0
            adjusted_confidence = 0.0
            total_weight = 0.0
            
            for model_name, pred in base_predictions.items():
                if model_name in rapid_change_models:
                    weight = self.model_weights.get(model_name, 0.2) * self.config['rapid_change_boost']
                else:
                    weight = self.model_weights.get(model_name, 0.2)
                
                total_weight += weight
                adjusted_prediction += weight * pred
                adjusted_confidence += weight * base_confidences.get(model_name, 0.0)
            
            if total_weight > 0:
                adjusted_prediction /= total_weight
                adjusted_confidence /= total_weight
                
                # Slightly reduce confidence due to rapid change uncertainty
                adjusted_confidence *= 0.9
            else:
                adjusted_prediction = prediction
                adjusted_confidence = confidence
            
            logger.debug("Adjusted prediction for rapid change scenario")
            return adjusted_prediction, adjusted_confidence
            
        except Exception as e:
            logger.warning(f"Rapid change adjustment failed: {e}")
            return prediction, confidence
    
    async def update_with_actual(self, actual_price: float, prediction_timestamp: datetime):
        """Update model weights based on actual price (online learning)"""
        try:
            if not self.config['online_weighting']:
                return
            
            # Find corresponding prediction
            prediction_record = None
            for record in reversed(self.prediction_history):
                if abs((record['timestamp'] - prediction_timestamp).total_seconds()) < 300:  # Within 5 minutes
                    prediction_record = record
                    break
            
            if not prediction_record:
                return
            
            # Calculate errors for each model
            ensemble_error = abs(actual_price - prediction_record['ensemble_prediction'])
            
            for model_name, predicted_price in prediction_record['base_predictions'].items():
                if predicted_price > 0:  # Valid prediction
                    error = abs(actual_price - predicted_price)
                    
                    # Store error
                    if model_name not in self.error_history:
                        self.error_history[model_name] = []
                    
                    self.error_history[model_name].append(error)
                    
                    # Limit error history
                    if len(self.error_history[model_name]) > self.config['error_memory']:
                        self.error_history[model_name] = self.error_history[model_name][-self.config['error_memory']:]
            
            # Update model weights based on recent performance
            self._update_model_weights()
            
            logger.debug(f"Updated model weights based on actual price: ${actual_price:.2f}")
            
        except Exception as e:
            logger.warning(f"Online weight update failed: {e}")
    
    def _update_model_weights(self):
        """Update model weights based on recent performance"""
        try:
            new_weights = {}
            
            for model_name in self.base_models.keys():
                if model_name in self.error_history and len(self.error_history[model_name]) > 0:
                    # Calculate recent performance
                    recent_errors = self.error_history[model_name][-self.config['adaptation_window']:]
                    avg_error = np.mean(recent_errors)
                    
                    # Convert error to weight (lower error = higher weight)
                    performance_score = max(0.1, 1.0 - min(avg_error / self.config['error_threshold'], 0.9))
                    
                    # Update weight with learning rate
                    current_weight = self.model_weights.get(model_name, 0.2)
                    new_weight = current_weight * (1 - self.config['learning_rate']) + \
                               performance_score * self.config['learning_rate']
                    
                    # Apply weight decay
                    new_weight *= self.config['weight_decay']
                    
                    # Clamp weight
                    new_weight = max(self.config['min_weight'], 
                                   min(self.config['max_weight'], new_weight))
                    
                    new_weights[model_name] = new_weight
                else:
                    # Keep current weight if no error history
                    new_weights[model_name] = self.model_weights.get(model_name, 0.2)
            
            # Normalize weights
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                for model_name in new_weights:
                    new_weights[model_name] /= total_weight
            
            self.model_weights = new_weights
            
        except Exception as e:
            logger.warning(f"Model weight update failed: {e}")
    
    def _calculate_performance_metrics(self, base_predictions: List[Dict[str, float]], targets: List[float]):
        """Calculate performance metrics for all models"""
        try:
            self.performance_metrics = {}
            
            for model_name in self.base_models.keys():
                predictions = [pred.get(model_name, 0.0) for pred in base_predictions]
                
                if len(predictions) > 0 and len(targets) >= len(predictions):
                    aligned_targets = targets[-len(predictions):]
                    
                    mae = mean_absolute_error(aligned_targets, predictions)
                    rmse = np.sqrt(mean_squared_error(aligned_targets, predictions))
                    mape = np.mean(np.abs((np.array(aligned_targets) - np.array(predictions)) / 
                                        (np.array(aligned_targets) + 1e-10))) * 100
                    
                    self.performance_metrics[model_name] = {
                        'mae': mae,
                        'rmse': rmse,
                        'mape': mape,
                        'predictions_count': len(predictions)
                    }
            
        except Exception as e:
            logger.warning(f"Performance metrics calculation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ensemble meta-learner information"""
        info = {
            'model_name': self.model_name,
            'model_type': 'Attention Fusion Meta-Learner',
            'is_trained': self.is_trained,
            'config': self.config.copy(),
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'model_weights': self.model_weights.copy(),
            'attention_weights': self.attention_weights.copy() if self.attention_weights else {},
            'performance_metrics': self.performance_metrics.copy()
        }
        
        # Base model information
        base_model_info = {}
        for model_name, model in self.base_models.items():
            base_model_info[model_name] = {
                'is_trained': model.is_trained,
                'last_training_time': model.last_training_time.isoformat() if model.last_training_time else None
            }
        
        info['base_models'] = base_model_info
        
        # Error history summary
        error_summary = {}
        for model_name, errors in self.error_history.items():
            if errors:
                error_summary[model_name] = {
                    'recent_errors': len(errors),
                    'avg_error': np.mean(errors[-10:]) if len(errors) >= 10 else np.mean(errors),
                    'error_trend': 'improving' if len(errors) >= 2 and errors[-1] < errors[-2] else 'stable'
                }
        
        info['error_summary'] = error_summary
        
        return info
    
    def get_ensemble_analysis(self) -> Dict[str, Any]:
        """Get detailed ensemble analysis"""
        analysis = {
            'model_contributions': self.model_weights.copy(),
            'attention_weights': self.attention_weights.copy() if self.attention_weights else {},
            'performance_ranking': {},
            'prediction_agreement': {},
            'recent_performance': {}
        }
        
        # Performance ranking
        if self.performance_metrics:
            mae_ranking = sorted(self.performance_metrics.items(), key=lambda x: x[1]['mae'])
            analysis['performance_ranking'] = {
                'by_mae': [model_name for model_name, _ in mae_ranking],
                'best_model': mae_ranking[0][0] if mae_ranking else None
            }
        
        # Recent prediction analysis
        if len(self.prediction_history) >= 10:
            recent_predictions = self.prediction_history[-10:]
            
            # Calculate prediction variance (agreement)
            for i, record in enumerate(recent_predictions):
                base_preds = list(record['base_predictions'].values())
                if len(base_preds) > 1:
                    variance = np.var(base_preds)
                    analysis['prediction_agreement'][i] = {
                        'variance': variance,
                        'agreement_level': 'high' if variance < 100 else 'medium' if variance < 500 else 'low'
                    }
        
        return analysis