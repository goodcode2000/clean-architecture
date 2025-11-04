"""Dynamic model weight adjustment based on performance and market conditions."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger
from config.config import Config

class DynamicWeightAdjuster:
    """Adjusts model weights based on performance and market conditions."""
    
    def __init__(self, base_weights: Dict[str, float],
                 window_size: int = 100,
                 min_weight: float = 0.05,
                 max_weight: float = 0.40):
        """
        Initialize the weight adjuster.
        
        Args:
            base_weights: Initial model weights
            window_size: Number of predictions to consider for adjustment
            min_weight: Minimum weight for any model
            max_weight: Maximum weight for any model
        """
        self.base_weights = base_weights.copy()
        self.current_weights = base_weights.copy()
        self.window_size = window_size
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Performance tracking
        self.prediction_errors = {model: [] for model in base_weights.keys()}
        self.market_conditions = []
        
    def update_errors(self, model_predictions: Dict[str, float],
                     actual_price: float,
                     market_volatility: float) -> None:
        """
        Update prediction errors for each model.
        
        Args:
            model_predictions: Dictionary of model predictions
            actual_price: Actual price that occurred
            market_volatility: Current market volatility
        """
        try:
            # Calculate absolute percentage errors
            for model, pred in model_predictions.items():
                error = abs(pred - actual_price) / actual_price
                self.prediction_errors[model].append(error)
                
                # Keep only recent errors
                if len(self.prediction_errors[model]) > self.window_size:
                    self.prediction_errors[model].pop(0)
            
            # Track market conditions
            self.market_conditions.append(market_volatility)
            if len(self.market_conditions) > self.window_size:
                self.market_conditions.pop(0)
                
            logger.debug(f"Updated errors for {len(model_predictions)} models")
            
        except Exception as e:
            logger.error(f"Failed to update errors: {e}")
    
    def calculate_performance_scores(self) -> Dict[str, float]:
        """Calculate performance scores for each model."""
        try:
            scores = {}
            
            for model, errors in self.prediction_errors.items():
                if not errors:  # No errors recorded yet
                    scores[model] = 1.0  # Default score
                    continue
                
                # Convert to numpy array for calculations
                errors = np.array(errors)
                
                # Calculate exponentially weighted average error
                # Recent errors matter more
                weights = np.exp(np.linspace(-1, 0, len(errors)))
                weights /= weights.sum()
                weighted_error = (errors * weights).sum()
                
                # Convert error to score (lower error = higher score)
                scores[model] = 1.0 / (1.0 + weighted_error)
            
            return scores
            
        except Exception as e:
            logger.error(f"Failed to calculate performance scores: {e}")
            return {model: 1.0 for model in self.prediction_errors.keys()}
    
    def assess_market_conditions(self) -> Dict[str, float]:
        """
        Assess current market conditions and determine model suitability.
        Returns weights based on market conditions.
        """
        try:
            if not self.market_conditions:
                return {model: 1.0 for model in self.base_weights.keys()}
            
            # Calculate recent volatility trend
            recent_vol = np.array(self.market_conditions[-20:])
            vol_trend = np.polyfit(np.arange(len(recent_vol)), recent_vol, 1)[0]
            
            # Current volatility level
            current_vol = self.market_conditions[-1]
            historical_vol = np.mean(self.market_conditions)
            
            # Determine market regime
            high_vol = current_vol > historical_vol * 1.5
            rising_vol = vol_trend > 0
            
            # Adjust weights based on market conditions
            condition_weights = {}
            
            for model in self.base_weights.keys():
                weight = 1.0
                
                # GARCH performs better in high volatility
                if model == 'garch':
                    weight *= 1.5 if high_vol else 0.8
                
                # LSTM performs better in trending markets
                elif model == 'lstm':
                    weight *= 1.3 if rising_vol else 0.9
                
                # SVR performs better in range-bound markets
                elif model == 'svr':
                    weight *= 1.2 if not high_vol else 0.8
                
                # RF and LightGBM are more stable across conditions
                else:
                    weight *= 1.0
                
                condition_weights[model] = weight
            
            return condition_weights
            
        except Exception as e:
            logger.error(f"Failed to assess market conditions: {e}")
            return {model: 1.0 for model in self.base_weights.keys()}
    
    def adjust_weights(self) -> Dict[str, float]:
        """
        Adjust model weights based on performance and market conditions.
        Returns updated weights.
        """
        try:
            # Get performance scores and market condition weights
            performance_scores = self.calculate_performance_scores()
            condition_weights = self.assess_market_conditions()
            
            # Combine scores and conditions
            combined_scores = {}
            for model in self.base_weights.keys():
                combined_scores[model] = (
                    performance_scores[model] *
                    condition_weights[model] *
                    self.base_weights[model]
                )
            
            # Normalize to get new weights
            total_score = sum(combined_scores.values())
            if total_score > 0:
                new_weights = {
                    model: score / total_score
                    for model, score in combined_scores.items()
                }
            else:
                new_weights = self.base_weights.copy()
            
            # Apply min/max constraints
            constrained_weights = self._apply_weight_constraints(new_weights)
            
            # Update current weights
            self.current_weights = constrained_weights
            
            logger.info("Model weights adjusted based on performance and market conditions")
            logger.debug(f"New weights: {constrained_weights}")
            
            return constrained_weights
            
        except Exception as e:
            logger.error(f"Failed to adjust weights: {e}")
            return self.base_weights.copy()
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply minimum and maximum weight constraints."""
        try:
            # First pass: Apply min/max constraints
            constrained = {}
            remaining_weight = 1.0
            
            for model, weight in weights.items():
                if weight < self.min_weight:
                    constrained[model] = self.min_weight
                    remaining_weight -= self.min_weight
                elif weight > self.max_weight:
                    constrained[model] = self.max_weight
                    remaining_weight -= self.max_weight
                else:
                    constrained[model] = weight
            
            # Second pass: Normalize remaining weights
            if remaining_weight > 0:
                unconstrained = {
                    m: w for m, w in weights.items()
                    if m not in constrained
                }
                
                if unconstrained:
                    total_unconstrained = sum(unconstrained.values())
                    if total_unconstrained > 0:
                        for model, weight in unconstrained.items():
                            constrained[model] = (weight / total_unconstrained) * remaining_weight
            
            # Final normalization
            total = sum(constrained.values())
            return {m: w/total for m, w in constrained.items()}
            
        except Exception as e:
            logger.error(f"Failed to apply weight constraints: {e}")
            return weights
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.current_weights.copy()
    
    def reset_weights(self) -> None:
        """Reset weights to base values."""
        self.current_weights = self.base_weights.copy()
        self.prediction_errors = {model: [] for model in self.base_weights.keys()}
        self.market_conditions = []