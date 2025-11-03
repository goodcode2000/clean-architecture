"""
Data models for the BTC Prediction Engine
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid

@dataclass
class PriceData:
    """Model for BTC price data"""
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV storage"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'open_price': self.open_price,
            'high_price': self.high_price,
            'low_price': self.low_price,
            'close_price': self.close_price,
            'volume': self.volume,
            **self.technical_indicators
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PriceData':
        """Create from dictionary"""
        # Extract technical indicators
        excluded_keys = {'timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume'}
        technical_indicators = {k: v for k, v in data.items() if k not in excluded_keys}
        
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            open_price=float(data['open_price']),
            high_price=float(data['high_price']),
            low_price=float(data['low_price']),
            close_price=float(data['close_price']),
            volume=float(data['volume']),
            technical_indicators=technical_indicators
        )

@dataclass
class Prediction:
    """Model for price predictions"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    current_price: float = 0.0
    predicted_price: float = 0.0
    confidence_score: float = 0.0
    model_contributions: Dict[str, float] = field(default_factory=dict)
    features_used: List[str] = field(default_factory=list)
    prediction_horizon: int = 5  # minutes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV storage"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'current_price': self.current_price,
            'predicted_price': self.predicted_price,
            'confidence_score': self.confidence_score,
            'model_contributions': str(self.model_contributions),
            'features_used': ','.join(self.features_used),
            'prediction_horizon': self.prediction_horizon
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Prediction':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            current_price=float(data['current_price']),
            predicted_price=float(data['predicted_price']),
            confidence_score=float(data['confidence_score']),
            model_contributions=eval(data['model_contributions']) if data['model_contributions'] else {},
            features_used=data['features_used'].split(',') if data['features_used'] else [],
            prediction_horizon=int(data['prediction_horizon'])
        )
    
    @property
    def prediction_offset(self) -> float:
        """Calculate the absolute offset between predicted and actual price"""
        return abs(self.predicted_price - self.current_price)

@dataclass
class ModelPerformance:
    """Model for tracking model performance metrics"""
    model_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    mae: float = 0.0  # Mean Absolute Error
    rmse: float = 0.0  # Root Mean Square Error
    accuracy_within_threshold: float = 0.0  # Percentage within 20-50 USD
    rapid_change_detection_rate: float = 0.0  # Success rate for >50 USD changes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV storage"""
        return {
            'model_name': self.model_name,
            'timestamp': self.timestamp.isoformat(),
            'mae': self.mae,
            'rmse': self.rmse,
            'accuracy_within_threshold': self.accuracy_within_threshold,
            'rapid_change_detection_rate': self.rapid_change_detection_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelPerformance':
        """Create from dictionary"""
        return cls(
            model_name=data['model_name'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            mae=float(data['mae']),
            rmse=float(data['rmse']),
            accuracy_within_threshold=float(data['accuracy_within_threshold']),
            rapid_change_detection_rate=float(data['rapid_change_detection_rate'])
        )

@dataclass
class MarketData:
    """Extended market data for feature engineering"""
    price_data: PriceData
    order_book_depth: Dict[str, float] = field(default_factory=dict)
    volume_delta: float = 0.0
    funding_rate: float = 0.0
    realized_volatility: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = self.price_data.to_dict()
        data.update({
            'order_book_depth': str(self.order_book_depth),
            'volume_delta': self.volume_delta,
            'funding_rate': self.funding_rate,
            'realized_volatility': self.realized_volatility
        })
        return data