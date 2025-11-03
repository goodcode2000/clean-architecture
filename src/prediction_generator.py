"""
Prediction Generator with Rapid Change Detection
Orchestrates ensemble models and generates final predictions
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from .models.ensemble_meta_learner import AttentionFusionMetaLearner
from .feature_engineering import FeatureEngineer, FeatureSet
from .feature_cache import FeaturePipeline
from .data_models import PriceData, Prediction, ModelPerformance
from .data_storage import DataStorage
from .config import Config

logger = logging.getLogger(__name__)

class PredictionGenerator:
    """Main prediction generator that orchestrates ensemble models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.ensemble_model = AttentionFusionMetaLearner()
        self.feature_engineer = FeatureEngineer()
        self.feature_pipeline = FeaturePipeline(config, self.feature_engineer)
        self.data_storage = DataStorage(config)
        
        # Prediction state
        self.is_initialized = False
        self.last_prediction_time = None
        self.prediction_count = 0
        self.rapid_change_threshold = 50.0  # USD
        
        # Performance tracking
        self.recent_predictions = []
        self.performance_history = []
        self.current_accuracy = 0.0
        
    async def initialize(self):
        """Initialize the prediction generator"""
        try:
            logger.info("Initializing prediction generator...")
            
            # Initialize data storage
            await self.data_storage.initialize()
            
            # Load historical data for model training
            historical_data = await self.data_storage.get_historical_data(days=90)
            
            if len(historical_data) < 100:
                logger.warning("Insufficient historical data for model training")
                return False
            
            # Engineer features
            logger.info("Engineering features for model training...")
            feature_sets = await self.feature_pipeline.process_features(historical_data)
            
            if not feature_sets:
                logger.error("Feature engineering failed")
                return False
            
            # Extract targets (prices)
            targets = [pd.close_price for pd in historical_data]
            
            # Train ensemble model
            logger.info("Training ensemble model...")
            training_success = await self.ensemble_model.train(feature_sets, targets)
            
            if not training_success:
                logger.error("Ensemble model training failed")
                return False
            
            self.is_initialized = True
            logger.info("Prediction generator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Prediction generator initialization failed: {e}")
            return False
    
    async def generate_prediction(self, current_data: Optional[PriceData] = None) -> Optional[Prediction]:
        """Generate a new price prediction"""
        try:
            if not self.is_initialized:
                logger.warning("Prediction generator not initialized")
                return None
            
            # Get current data if not provided
            if current_data is None:
                recent_data = await self.data_storage.get_recent_price_data(hours=2)
                if not recent_data:
                    logger.warning("No recent data available for prediction")
                    return None
                current_data = recent_data[-1]
            
            # Get recent data for feature engineering
            recent_data = await self.data_storage.get_recent_price_data(hours=5)
            if len(recent_data) < 10:
                logger.warning("Insufficient recent data for feature engineering")
                return None
            
            # Engineer features
            feature_sets = await self.feature_pipeline.process_features(recent_data)
            if not feature_sets:
                logger.warning("Feature engineering failed for prediction")
                return None
            
            current_features = feature_sets[-1]
            
            # Generate ensemble prediction
            predicted_price, confidence = await self.ensemble_model.predict(current_features)
            
            # Detect rapid change
            rapid_change_info = await self._detect_rapid_change(current_features, predicted_price, current_data.close_price)
            
            # Create prediction object
            prediction = Prediction(
                timestamp=datetime.now(),
                current_price=current_data.close_price,
                predicted_price=predicted_price,
                confidence_score=confidence,
                model_contributions=self._get_model_contributions(),
                features_used=list(current_features.to_dict().keys()),
                prediction_horizon=5  # 5 minutes
            )
            
            # Add rapid change information
            prediction.rapid_change_detected = rapid_change_info['detected']
            prediction.rapid_change_magnitude = rapid_change_info['magnitude']
            prediction.rapid_change_direction = rapid_change_info['direction']
            
            # Store prediction
            await self.data_storage.store_prediction(prediction)
            
            # Update tracking
            self.recent_predictions.append(prediction)
            if len(self.recent_predictions) > 100:
                self.recent_predictions = self.recent_predictions[-100:]
            
            self.last_prediction_time = datetime.now()
            self.prediction_count += 1
            
            logger.info(f"Generated prediction #{self.prediction_count}: ${predicted_price:.2f} (confidence: {confidence:.3f})")
            
            if rapid_change_info['detected']:
                logger.warning(f"Rapid change detected: {rapid_change_info['direction']} ${rapid_change_info['magnitude']:.2f}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            return None
    
    async def _detect_rapid_change(self, features: FeatureSet, predicted_price: float, current_price: float) -> Dict[str, Any]:
        """Detect potential rapid price changes"""
        try:
            rapid_change_info = {
                'detected': False,
                'magnitude': 0.0,
                'direction': 'stable',
                'confidence': 0.0,
                'indicators': []
            }
            
            # Calculate predicted change magnitude
            price_change = abs(predicted_price - current_price)
            
            # Check if predicted change exceeds threshold
            if price_change > self.rapid_change_threshold:
                rapid_change_info['detected'] = True
                rapid_change_info['magnitude'] = price_change
                rapid_change_info['direction'] = 'up' if predicted_price > current_price else 'down'
                rapid_change_info['indicators'].append('prediction_magnitude')
            
            # Check volatility indicators
            feature_dict = features.to_dict()
            
            # High volatility indicator
            volatility_5d = feature_dict.get('volatility_5d', 0.0)
            if volatility_5d > 0.5:  # High volatility threshold
                rapid_change_info['detected'] = True
                rapid_change_info['indicators'].append('high_volatility')
            
            # Recent jump detection
            recent_jump = feature_dict.get('recent_jump', 0.0)
            if recent_jump > 0:
                rapid_change_info['detected'] = True
                rapid_change_info['indicators'].append('recent_jump')
            
            # Momentum indicators
            momentum_5 = feature_dict.get('momentum_5', 0.0)
            if abs(momentum_5) > 0.02:  # 2% momentum threshold
                rapid_change_info['detected'] = True
                rapid_change_info['indicators'].append('high_momentum')
            
            # RSI extreme levels
            rsi = feature_dict.get('rsi', 50.0)
            if rsi > 80 or rsi < 20:
                rapid_change_info['detected'] = True
                rapid_change_info['indicators'].append('rsi_extreme')
            
            # Bollinger Band position
            bb_position = feature_dict.get('bb_position', 0.5)
            if bb_position > 0.95 or bb_position < 0.05:
                rapid_change_info['detected'] = True
                rapid_change_info['indicators'].append('bb_extreme')
            
            # Calculate confidence based on number of indicators
            num_indicators = len(rapid_change_info['indicators'])
            rapid_change_info['confidence'] = min(1.0, num_indicators / 3.0)
            
            return rapid_change_info
            
        except Exception as e:
            logger.error(f"Rapid change detection failed: {e}")
            return {'detected': False, 'magnitude': 0.0, 'direction': 'stable', 'confidence': 0.0, 'indicators': []}
    
    def _get_model_contributions(self) -> Dict[str, float]:
        """Get current model contributions from ensemble"""
        try:
            if hasattr(self.ensemble_model, 'model_weights'):
                return self.ensemble_model.model_weights.copy()
            else:
                return {}
        except Exception as e:
            logger.warning(f"Failed to get model contributions: {e}")
            return {}
    
    async def update_with_actual_price(self, actual_price: float, prediction_timestamp: datetime):
        """Update models with actual price for online learning"""
        try:
            # Update ensemble model
            await self.ensemble_model.update_with_actual(actual_price, prediction_timestamp)
            
            # Find corresponding prediction for accuracy calculation
            matching_prediction = None
            for pred in reversed(self.recent_predictions):
                time_diff = abs((pred.timestamp - prediction_timestamp).total_seconds())
                if time_diff < 300:  # Within 5 minutes
                    matching_prediction = pred
                    break
            
            if matching_prediction:
                # Calculate accuracy
                error = abs(actual_price - matching_prediction.predicted_price)
                accuracy = max(0.0, 1.0 - error / actual_price)
                
                # Update performance tracking
                performance_record = {
                    'timestamp': datetime.now(),
                    'predicted_price': matching_prediction.predicted_price,
                    'actual_price': actual_price,
                    'error': error,
                    'accuracy': accuracy,
                    'rapid_change_detected': getattr(matching_prediction, 'rapid_change_detected', False)
                }
                
                self.performance_history.append(performance_record)
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-1000:]
                
                # Update current accuracy (rolling average)
                recent_accuracies = [p['accuracy'] for p in self.performance_history[-20:]]
                self.current_accuracy = np.mean(recent_accuracies)
                
                logger.info(f"Updated with actual price: ${actual_price:.2f}, Error: ${error:.2f}, Accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Actual price update failed: {e}")
    
    async def retrain_models(self):
        """Retrain ensemble models with recent data"""
        try:
            logger.info("Starting model retraining...")
            
            # Get recent data for retraining (7-day rolling window)
            recent_data = await self.data_storage.get_historical_data(days=7)
            
            if len(recent_data) < 100:
                logger.warning("Insufficient data for retraining")
                return False
            
            # Engineer features
            feature_sets = await self.feature_pipeline.process_features(recent_data)
            if not feature_sets:
                logger.error("Feature engineering failed for retraining")
                return False
            
            # Extract targets
            targets = [pd.close_price for pd in recent_data]
            
            # Retrain ensemble model
            training_success = await self.ensemble_model.train(feature_sets, targets)
            
            if training_success:
                logger.info("Model retraining completed successfully")
                
                # Store performance metrics
                model_info = self.ensemble_model.get_model_info()
                if 'latest_metrics' in model_info:
                    performance = ModelPerformance(
                        model_name="Ensemble",
                        timestamp=datetime.now(),
                        mae=model_info['latest_metrics'].get('mae', 0.0),
                        rmse=model_info['latest_metrics'].get('rmse', 0.0),
                        accuracy_within_threshold=self.current_accuracy,
                        rapid_change_detection_rate=self._calculate_rapid_change_detection_rate()
                    )
                    
                    await self.data_storage.store_performance(performance)
                
                return True
            else:
                logger.error("Model retraining failed")
                return False
                
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            return False
    
    def _calculate_rapid_change_detection_rate(self) -> float:
        """Calculate rapid change detection success rate"""
        try:
            if not self.performance_history:
                return 0.0
            
            rapid_change_cases = [p for p in self.performance_history if p.get('rapid_change_detected', False)]
            
            if not rapid_change_cases:
                return 0.0
            
            # Calculate success rate for rapid change predictions
            successful_detections = 0
            for case in rapid_change_cases:
                actual_change = abs(case['actual_price'] - case['predicted_price'])
                if actual_change > self.rapid_change_threshold:
                    successful_detections += 1
            
            return successful_detections / len(rapid_change_cases)
            
        except Exception as e:
            logger.warning(f"Rapid change detection rate calculation failed: {e}")
            return 0.0
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction generator statistics"""
        try:
            stats = {
                'is_initialized': self.is_initialized,
                'prediction_count': self.prediction_count,
                'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
                'current_accuracy': self.current_accuracy,
                'rapid_change_threshold': self.rapid_change_threshold
            }
            
            # Recent performance
            if self.performance_history:
                recent_performance = self.performance_history[-10:]
                stats['recent_performance'] = {
                    'avg_error': np.mean([p['error'] for p in recent_performance]),
                    'avg_accuracy': np.mean([p['accuracy'] for p in recent_performance]),
                    'rapid_changes_detected': sum(1 for p in recent_performance if p.get('rapid_change_detected', False))
                }
            
            # Model information
            if self.ensemble_model.is_trained:
                ensemble_info = self.ensemble_model.get_model_info()
                stats['ensemble_info'] = {
                    'model_weights': ensemble_info.get('model_weights', {}),
                    'attention_weights': ensemble_info.get('attention_weights', {}),
                    'base_models_trained': sum(1 for model_info in ensemble_info.get('base_models', {}).values() 
                                             if model_info.get('is_trained', False))
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get prediction stats: {e}")
            return {'error': str(e)}
    
    def get_recent_predictions(self, count: int = 20) -> List[Dict[str, Any]]:
        """Get recent predictions"""
        try:
            recent = self.recent_predictions[-count:] if count <= len(self.recent_predictions) else self.recent_predictions
            
            predictions_data = []
            for pred in recent:
                pred_data = pred.to_dict()
                
                # Add additional fields if available
                if hasattr(pred, 'rapid_change_detected'):
                    pred_data['rapid_change_detected'] = pred.rapid_change_detected
                    pred_data['rapid_change_magnitude'] = getattr(pred, 'rapid_change_magnitude', 0.0)
                    pred_data['rapid_change_direction'] = getattr(pred, 'rapid_change_direction', 'stable')
                
                predictions_data.append(pred_data)
            
            return predictions_data
            
        except Exception as e:
            logger.error(f"Failed to get recent predictions: {e}")
            return []
    
    async def cleanup_cache(self):
        """Clean up feature cache"""
        try:
            await self.feature_pipeline.cleanup_cache()
            logger.info("Feature cache cleaned up")
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get detailed model diagnostics"""
        try:
            diagnostics = {
                'prediction_generator': {
                    'initialized': self.is_initialized,
                    'prediction_count': self.prediction_count,
                    'current_accuracy': self.current_accuracy,
                    'performance_history_size': len(self.performance_history)
                }
            }
            
            # Ensemble model diagnostics
            if self.ensemble_model.is_trained:
                ensemble_analysis = self.ensemble_model.get_ensemble_analysis()
                diagnostics['ensemble'] = ensemble_analysis
                
                # Individual model diagnostics
                diagnostics['base_models'] = {}
                for model_name, model in self.ensemble_model.base_models.items():
                    if model.is_trained:
                        model_info = model.get_model_info()
                        diagnostics['base_models'][model_name] = {
                            'is_trained': model.is_trained,
                            'last_training_time': model_info.get('last_training_time'),
                            'latest_metrics': model_info.get('latest_metrics', {})
                        }
            
            # Feature pipeline diagnostics
            cache_stats = self.feature_pipeline.get_cache_stats()
            diagnostics['feature_cache'] = cache_stats
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Model diagnostics failed: {e}")
            return {'error': str(e)}