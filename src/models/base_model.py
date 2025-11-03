"""
Base model interface for ensemble models
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging
from datetime import datetime
from ..data_models import PriceData, Prediction
from ..feature_engineering import FeatureSet

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all prediction models"""
    
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        self.model_name = model_name
        self.config = config or {}
        self.is_trained = False
        self.model = None
        self.feature_importance = {}
        self.training_history = []
        self.last_training_time = None
        
    @abstractmethod
    async def train(self, features: List[FeatureSet], targets: List[float]) -> bool:
        """Train the model with features and targets"""
        pass
    
    @abstractmethod
    async def predict(self, features: FeatureSet) -> Tuple[float, float]:
        """Make prediction and return (prediction, confidence)"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and parameters"""
        pass
    
    def save_model(self, filepath: str) -> bool:
        """Save model to file"""
        try:
            import pickle
            model_data = {
                'model': self.model,
                'model_name': self.model_name,
                'config': self.config,
                'is_trained': self.is_trained,
                'feature_importance': self.feature_importance,
                'last_training_time': self.last_training_time
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model {self.model_name} saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model {self.model_name}: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load model from file"""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.config = model_data['config']
            self.is_trained = model_data['is_trained']
            self.feature_importance = model_data.get('feature_importance', {})
            self.last_training_time = model_data.get('last_training_time')
            
            logger.info(f"Model {self.model_name} loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance.copy()
    
    def update_training_history(self, metrics: Dict[str, float]):
        """Update training history with metrics"""
        history_entry = {
            'timestamp': datetime.now(),
            'metrics': metrics
        }
        self.training_history.append(history_entry)
        
        # Keep only last 100 entries
        if len(self.training_history) > 100:
            self.training_history = self.training_history[-100:]
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history"""
        return self.training_history.copy()
    
    def validate_features(self, features: FeatureSet) -> bool:
        """Validate feature set for prediction"""
        try:
            feature_dict = features.to_dict()
            
            # Check for required features (basic validation)
            required_features = ['price', 'volume']
            for feature in required_features:
                if feature not in feature_dict:
                    logger.warning(f"Missing required feature: {feature}")
                    return False
            
            # Check for NaN or infinite values
            for key, value in feature_dict.items():
                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                    logger.warning(f"Invalid feature value: {key} = {value}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Feature validation failed: {e}")
            return False
    
    def preprocess_features(self, features: FeatureSet) -> np.ndarray:
        """Preprocess features for model input"""
        try:
            feature_dict = features.to_dict()
            
            # Convert to numpy array
            feature_values = []
            feature_names = sorted(feature_dict.keys())  # Ensure consistent ordering
            
            for name in feature_names:
                value = feature_dict[name]
                
                # Handle missing values
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                
                feature_values.append(value)
            
            return np.array(feature_values)
            
        except Exception as e:
            logger.error(f"Feature preprocessing failed: {e}")
            return np.array([])
    
    def calculate_prediction_confidence(self, prediction: float, features: FeatureSet) -> float:
        """Calculate confidence score for prediction"""
        try:
            # Base confidence calculation (can be overridden by specific models)
            confidence = 0.5  # Default confidence
            
            # Adjust based on feature quality
            feature_dict = features.to_dict()
            
            # Higher confidence if we have more features
            feature_count = len([v for v in feature_dict.values() if not (np.isnan(v) or np.isinf(v))])
            confidence += min(0.3, feature_count / 100.0)
            
            # Adjust based on volatility (lower confidence in high volatility)
            if 'volatility_5d' in feature_dict:
                vol = feature_dict['volatility_5d']
                if vol > 0.5:  # High volatility
                    confidence -= 0.2
                elif vol < 0.2:  # Low volatility
                    confidence += 0.1
            
            # Ensure confidence is between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5  # Default confidence