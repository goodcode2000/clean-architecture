"""Dynamic weighting system for ensemble model adaptation."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

@dataclass
class ModelPerformance:
    """Data structure for tracking individual model performance."""
    model_name: str
    predictions: List[float]
    actuals: List[float]
    timestamps: List[datetime]
    errors: List[float]
    volatility_scores: List[float]
    
    def calculate_accuracy(self) -> float:
        """Calculate overall accuracy (1 - normalized MAE)."""
        if not self.errors:
            return 0.0
        
        mae = np.mean(np.abs(self.errors))
        mean_actual = np.mean(self.actuals) if self.actuals else 1.0
        normalized_mae = mae / mean_actual if mean_actual > 0 else mae
        
        # Convert to accuracy score (higher is better)
        accuracy = max(0.0, 1.0 - normalized_mae)
        return accuracy
    
    def calculate_volatility_performance(self) -> float:
        """Calculate performance during volatile periods."""
        if not self.volatility_scores or not self.errors:
            return 0.0
        
        # Weight errors by volatility (higher volatility = more important)
        weighted_errors = []
        for error, vol_score in zip(self.errors, self.volatility_scores):
            if vol_score > 0.05:  # Only consider volatile periods
                weighted_errors.append(abs(error))
        
        if not weighted_errors:
            return self.calculate_accuracy()  # Fallback to overall accuracy
        
        volatility_mae = np.mean(weighted_errors)
        mean_actual = np.mean(self.actuals) if self.actuals else 1.0
        normalized_mae = volatility_mae / mean_actual if mean_actual > 0 else volatility_mae
        
        # Convert to performance score
        performance = max(0.0, 1.0 - normalized_mae)
        return performance
    
    def get_recent_performance(self, window: int) -> float:
        """Get performance over recent window."""
        if len(self.errors) < window:
            return self.calculate_accuracy()
        
        recent_errors = self.errors[-window:]
        recent_actuals = self.actuals[-window:]
        
        mae = np.mean(np.abs(recent_errors))
        mean_actual = np.mean(recent_actuals) if recent_actuals else 1.0
        normalized_mae = mae / mean_actual if mean_actual > 0 else mae
        
        performance = max(0.0, 1.0 - normalized_mae)
        return performance

@dataclass
class WeightCalculation:
    """Data structure for weight calculation results."""
    model_weights: Dict[str, float]
    performance_scores: Dict[str, float]
    market_regime: str
    calculation_timestamp: datetime
    
    def normalize_weights(self) -> Dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total_weight = sum(self.model_weights.values())
        if total_weight <= 0:
            # Equal weights if all are zero
            n_models = len(self.model_weights)
            return {name: 1.0 / n_models for name in self.model_weights.keys()}
        
        return {name: weight / total_weight for name, weight in self.model_weights.items()}
    
    def apply_constraints(self, min_weight: float, max_weight: float) -> Dict[str, float]:
        """Apply minimum and maximum weight constraints."""
        constrained_weights = {}
        
        for name, weight in self.model_weights.items():
            constrained_weights[name] = max(min_weight, min(max_weight, weight))
        
        # Renormalize after applying constraints
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {name: weight / total_weight 
                                 for name, weight in constrained_weights.items()}
        
        return constrained_weights

class DynamicWeightManager:
    """Dynamic weight management system for ensemble model adaptation."""
    
    def __init__(self, window_size: int = 50, min_weight: float = 0.05, max_weight: float = 0.5):
        self.window_size = window_size
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Performance tracking
        self.model_performances: Dict[str, ModelPerformance] = {}
        self.current_weights: Dict[str, float] = {}
        self.weight_history: List[WeightCalculation] = []
        
        # Market regime detection
        self.volatility_threshold = 0.05  # 5% price change threshold
        self.current_regime = "stable"
        self.regime_history: List[Tuple[datetime, str]] = []
        
        # Performance decay parameters
        self.decay_factor = 0.95  # Exponential decay for older predictions
        self.adaptation_speed = 0.1  # How quickly to adapt weights (0-1)
        
        # Initialize with default model names
        self.initialize_models(['arima', 'svr', 'random_forest', 'xgboost', 'lstm'])
        
    def initialize_models(self, model_names: List[str]):
        """Initialize performance tracking for models."""
        for model_name in model_names:
            self.model_performances[model_name] = ModelPerformance(
                model_name=model_name,
                predictions=[],
                actuals=[],
                timestamps=[],
                errors=[],
                volatility_scores=[]
            )
            # Initialize with equal weights
            self.current_weights[model_name] = 1.0 / len(model_names)
        
        logger.info(f"Initialized dynamic weighting for models: {model_names}")
    
    def update_performance(self, model_name: str, prediction: float, actual: float, 
                         volatility: float, timestamp: Optional[datetime] = None):
        """
        Update performance tracking for a model.
        
        Args:
            model_name: Name of the model
            prediction: Predicted value
            actual: Actual value
            volatility: Market volatility score
            timestamp: Timestamp of prediction (defaults to now)
        """
        try:
            if model_name not in self.model_performances:
                logger.warning(f"Model {model_name} not initialized, adding it")
                self.model_performances[model_name] = ModelPerformance(
                    model_name=model_name,
                    predictions=[],
                    actuals=[],
                    timestamps=[],
                    errors=[],
                    volatility_scores=[]
                )
            
            if timestamp is None:
                timestamp = datetime.now()
            
            # Calculate error
            error = prediction - actual
            
            # Update performance data
            performance = self.model_performances[model_name]
            performance.predictions.append(prediction)
            performance.actuals.append(actual)
            performance.timestamps.append(timestamp)
            performance.errors.append(error)
            performance.volatility_scores.append(volatility)
            
            # Maintain window size
            if len(performance.predictions) > self.window_size:
                performance.predictions = performance.predictions[-self.window_size:]
                performance.actuals = performance.actuals[-self.window_size:]
                performance.timestamps = performance.timestamps[-self.window_size:]
                performance.errors = performance.errors[-self.window_size:]
                performance.volatility_scores = performance.volatility_scores[-self.window_size:]
            
            logger.debug(f"Updated performance for {model_name}: error={error:.2f}, volatility={volatility:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to update performance for {model_name}: {e}")
    
    def detect_market_regime(self, recent_prices: List[float]) -> str:
        """
        Detect current market regime based on recent price movements.
        
        Args:
            recent_prices: List of recent prices
            
        Returns:
            Market regime: 'stable', 'volatile', 'trending_up', 'trending_down'
        """
        try:
            if len(recent_prices) < 10:
                return "stable"
            
            # Calculate price changes
            price_changes = []
            for i in range(1, len(recent_prices)):
                change = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                price_changes.append(abs(change))
            
            # Calculate volatility metrics
            avg_volatility = np.mean(price_changes)
            max_volatility = np.max(price_changes)
            
            # Calculate trend
            trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            # Determine regime
            if max_volatility > self.volatility_threshold * 2:  # High volatility
                regime = "volatile"
            elif avg_volatility > self.volatility_threshold:  # Moderate volatility
                if trend > 0.02:  # 2% upward trend
                    regime = "trending_up"
                elif trend < -0.02:  # 2% downward trend
                    regime = "trending_down"
                else:
                    regime = "volatile"
            else:
                regime = "stable"
            
            # Update regime history
            self.current_regime = regime
            self.regime_history.append((datetime.now(), regime))
            
            # Maintain regime history size
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            logger.debug(f"Market regime detected: {regime} (volatility: {avg_volatility:.4f}, trend: {trend:.4f})")
            return regime
            
        except Exception as e:
            logger.error(f"Market regime detection failed: {e}")
            return "stable"
    
    def calculate_weights(self) -> Dict[str, float]:
        """
        Calculate dynamic weights based on recent model performance and market regime.
        
        Returns:
            Dictionary of model weights
        """
        try:
            logger.debug("Calculating dynamic weights...")
            
            # Get performance scores for each model
            performance_scores = {}
            
            for model_name, performance in self.model_performances.items():
                if len(performance.predictions) < 5:  # Need minimum data
                    performance_scores[model_name] = 0.5  # Neutral score
                    continue
                
                # Calculate different performance metrics
                overall_accuracy = performance.calculate_accuracy()
                volatility_performance = performance.calculate_volatility_performance()
                recent_performance = performance.get_recent_performance(window=min(20, len(performance.predictions)))
                
                # Weight the metrics based on market regime
                if self.current_regime == "volatile":
                    # Prioritize volatility performance during volatile periods
                    score = (0.2 * overall_accuracy + 
                            0.5 * volatility_performance + 
                            0.3 * recent_performance)
                elif self.current_regime in ["trending_up", "trending_down"]:
                    # Prioritize recent performance during trending markets
                    score = (0.3 * overall_accuracy + 
                            0.2 * volatility_performance + 
                            0.5 * recent_performance)
                else:  # stable market
                    # Balanced approach for stable markets
                    score = (0.4 * overall_accuracy + 
                            0.3 * volatility_performance + 
                            0.3 * recent_performance)
                
                performance_scores[model_name] = score
            
            # Convert performance scores to weights
            raw_weights = {}
            total_score = sum(performance_scores.values())
            
            if total_score <= 0:
                # Equal weights if no performance data
                n_models = len(performance_scores)
                raw_weights = {name: 1.0 / n_models for name in performance_scores.keys()}
            else:
                # Performance-based weights
                for model_name, score in performance_scores.items():
                    raw_weights[model_name] = score / total_score
            
            # Apply adaptive smoothing (gradual weight changes)
            if self.current_weights:
                smoothed_weights = {}
                for model_name in raw_weights.keys():
                    old_weight = self.current_weights.get(model_name, 0.0)
                    new_weight = raw_weights[model_name]
                    
                    # Exponential smoothing
                    smoothed_weight = (1 - self.adaptation_speed) * old_weight + self.adaptation_speed * new_weight
                    smoothed_weights[model_name] = smoothed_weight
                
                raw_weights = smoothed_weights
            
            # Create weight calculation object
            weight_calc = WeightCalculation(
                model_weights=raw_weights.copy(),
                performance_scores=performance_scores.copy(),
                market_regime=self.current_regime,
                calculation_timestamp=datetime.now()
            )
            
            # Apply constraints and normalize
            final_weights = weight_calc.apply_constraints(self.min_weight, self.max_weight)
            final_weights = WeightCalculation(
                model_weights=final_weights,
                performance_scores=performance_scores,
                market_regime=self.current_regime,
                calculation_timestamp=datetime.now()
            ).normalize_weights()
            
            # Update current weights
            self.current_weights = final_weights.copy()
            
            # Store in history
            weight_calc.model_weights = final_weights
            self.weight_history.append(weight_calc)
            
            # Maintain history size
            if len(self.weight_history) > 100:
                self.weight_history = self.weight_history[-100:]
            
            logger.info(f"Dynamic weights calculated for {self.current_regime} market:")
            for model_name, weight in final_weights.items():
                score = performance_scores.get(model_name, 0.0)
                logger.info(f"  {model_name}: {weight:.3f} (performance: {score:.3f})")
            
            return final_weights
            
        except Exception as e:
            logger.error(f"Weight calculation failed: {e}")
            # Return equal weights as fallback
            n_models = len(self.model_performances)
            return {name: 1.0 / n_models for name in self.model_performances.keys()}
    
    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive performance metrics for all models.
        
        Returns:
            Dictionary with performance metrics for each model
        """
        try:
            metrics = {}
            
            for model_name, performance in self.model_performances.items():
                if len(performance.predictions) == 0:
                    metrics[model_name] = {
                        'overall_accuracy': 0.0,
                        'volatility_performance': 0.0,
                        'recent_performance': 0.0,
                        'n_predictions': 0,
                        'current_weight': self.current_weights.get(model_name, 0.0)
                    }
                    continue
                
                model_metrics = {
                    'overall_accuracy': performance.calculate_accuracy(),
                    'volatility_performance': performance.calculate_volatility_performance(),
                    'recent_performance': performance.get_recent_performance(window=20),
                    'n_predictions': len(performance.predictions),
                    'current_weight': self.current_weights.get(model_name, 0.0),
                    'mean_error': np.mean(performance.errors) if performance.errors else 0.0,
                    'mae': np.mean(np.abs(performance.errors)) if performance.errors else 0.0,
                    'rmse': np.sqrt(np.mean(np.square(performance.errors))) if performance.errors else 0.0
                }
                
                # Directional accuracy
                if len(performance.predictions) > 1:
                    actual_direction = np.sign(np.diff(performance.actuals))
                    pred_direction = np.sign(np.diff(performance.predictions))
                    directional_accuracy = np.mean(actual_direction == pred_direction)
                    model_metrics['directional_accuracy'] = directional_accuracy
                else:
                    model_metrics['directional_accuracy'] = 0.0
                
                metrics[model_name] = model_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    def get_weight_history(self, n_recent: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent weight calculation history.
        
        Args:
            n_recent: Number of recent calculations to return
            
        Returns:
            List of weight calculation dictionaries
        """
        try:
            recent_history = self.weight_history[-n_recent:] if self.weight_history else []
            
            history_data = []
            for calc in recent_history:
                history_data.append({
                    'timestamp': calc.calculation_timestamp.isoformat(),
                    'market_regime': calc.market_regime,
                    'weights': calc.model_weights.copy(),
                    'performance_scores': calc.performance_scores.copy()
                })
            
            return history_data
            
        except Exception as e:
            logger.error(f"Failed to get weight history: {e}")
            return []
    
    def save_state(self, filepath: str) -> bool:
        """
        Save the current state of the weight manager.
        
        Args:
            filepath: Path to save the state
            
        Returns:
            True if successful, False otherwise
        """
        try:
            state_data = {
                'window_size': self.window_size,
                'min_weight': self.min_weight,
                'max_weight': self.max_weight,
                'current_weights': self.current_weights,
                'current_regime': self.current_regime,
                'volatility_threshold': self.volatility_threshold,
                'decay_factor': self.decay_factor,
                'adaptation_speed': self.adaptation_speed,
                'model_performances': {},
                'weight_history': [],
                'regime_history': []
            }
            
            # Serialize model performances
            for model_name, performance in self.model_performances.items():
                state_data['model_performances'][model_name] = {
                    'model_name': performance.model_name,
                    'predictions': performance.predictions,
                    'actuals': performance.actuals,
                    'timestamps': [ts.isoformat() for ts in performance.timestamps],
                    'errors': performance.errors,
                    'volatility_scores': performance.volatility_scores
                }
            
            # Serialize weight history
            for calc in self.weight_history:
                state_data['weight_history'].append({
                    'model_weights': calc.model_weights,
                    'performance_scores': calc.performance_scores,
                    'market_regime': calc.market_regime,
                    'calculation_timestamp': calc.calculation_timestamp.isoformat()
                })
            
            # Serialize regime history
            for timestamp, regime in self.regime_history:
                state_data['regime_history'].append({
                    'timestamp': timestamp.isoformat(),
                    'regime': regime
                })
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.info(f"Dynamic weight manager state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save weight manager state: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Load the state of the weight manager.
        
        Args:
            filepath: Path to load the state from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Weight manager state file not found: {filepath}")
                return False
            
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            # Load basic parameters
            self.window_size = state_data.get('window_size', 50)
            self.min_weight = state_data.get('min_weight', 0.05)
            self.max_weight = state_data.get('max_weight', 0.5)
            self.current_weights = state_data.get('current_weights', {})
            self.current_regime = state_data.get('current_regime', 'stable')
            self.volatility_threshold = state_data.get('volatility_threshold', 0.05)
            self.decay_factor = state_data.get('decay_factor', 0.95)
            self.adaptation_speed = state_data.get('adaptation_speed', 0.1)
            
            # Load model performances
            self.model_performances = {}
            for model_name, perf_data in state_data.get('model_performances', {}).items():
                timestamps = [datetime.fromisoformat(ts) for ts in perf_data['timestamps']]
                
                self.model_performances[model_name] = ModelPerformance(
                    model_name=perf_data['model_name'],
                    predictions=perf_data['predictions'],
                    actuals=perf_data['actuals'],
                    timestamps=timestamps,
                    errors=perf_data['errors'],
                    volatility_scores=perf_data['volatility_scores']
                )
            
            # Load weight history
            self.weight_history = []
            for calc_data in state_data.get('weight_history', []):
                timestamp = datetime.fromisoformat(calc_data['calculation_timestamp'])
                
                weight_calc = WeightCalculation(
                    model_weights=calc_data['model_weights'],
                    performance_scores=calc_data['performance_scores'],
                    market_regime=calc_data['market_regime'],
                    calculation_timestamp=timestamp
                )
                self.weight_history.append(weight_calc)
            
            # Load regime history
            self.regime_history = []
            for regime_data in state_data.get('regime_history', []):
                timestamp = datetime.fromisoformat(regime_data['timestamp'])
                self.regime_history.append((timestamp, regime_data['regime']))
            
            logger.info(f"Dynamic weight manager state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load weight manager state: {e}")
            return False
    
    def reset_performance(self, model_name: Optional[str] = None):
        """
        Reset performance tracking for a specific model or all models.
        
        Args:
            model_name: Name of model to reset, or None to reset all
        """
        try:
            if model_name:
                if model_name in self.model_performances:
                    self.model_performances[model_name] = ModelPerformance(
                        model_name=model_name,
                        predictions=[],
                        actuals=[],
                        timestamps=[],
                        errors=[],
                        volatility_scores=[]
                    )
                    logger.info(f"Reset performance tracking for {model_name}")
            else:
                # Reset all models
                for name in self.model_performances.keys():
                    self.model_performances[name] = ModelPerformance(
                        model_name=name,
                        predictions=[],
                        actuals=[],
                        timestamps=[],
                        errors=[],
                        volatility_scores=[]
                    )
                
                # Reset weights to equal
                n_models = len(self.model_performances)
                self.current_weights = {name: 1.0 / n_models for name in self.model_performances.keys()}
                
                logger.info("Reset performance tracking for all models")
                
        except Exception as e:
            logger.error(f"Failed to reset performance tracking: {e}")
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about market regime changes.
        
        Returns:
            Dictionary with regime statistics
        """
        try:
            if not self.regime_history:
                return {'total_regimes': 0, 'current_regime': self.current_regime}
            
            # Count regime occurrences
            regime_counts = {}
            for _, regime in self.regime_history:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # Calculate regime durations
            regime_durations = []
            for i in range(1, len(self.regime_history)):
                prev_time, prev_regime = self.regime_history[i-1]
                curr_time, curr_regime = self.regime_history[i]
                
                if prev_regime != curr_regime:
                    duration = (curr_time - prev_time).total_seconds() / 60  # minutes
                    regime_durations.append(duration)
            
            stats = {
                'total_regimes': len(self.regime_history),
                'current_regime': self.current_regime,
                'regime_counts': regime_counts,
                'avg_regime_duration_minutes': np.mean(regime_durations) if regime_durations else 0.0,
                'regime_changes': len(regime_durations)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get regime statistics: {e}")
            return {'error': str(e)}