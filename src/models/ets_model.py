"""
ETS (Error, Trend, Seasonality) Model for BTC Price Prediction
Captures level, trend, and seasonal components for smooth overall behavior
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import seasonal_decompose
from .base_model import BaseModel
from ..feature_engineering import FeatureSet

logger = logging.getLogger(__name__)

class ETSPredictionModel(BaseModel):
    """ETS model for capturing level, trend, and seasonality in BTC prices"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'error': 'add',  # additive error
            'trend': 'add',  # additive trend
            'seasonal': None,  # no seasonality for crypto (can be changed)
            'seasonal_periods': 24,  # 24 periods (2 hours for 5-min data)
            'damped_trend': True,  # damped trend
            'smoothing_level': None,  # auto-optimize
            'smoothing_trend': None,  # auto-optimize
            'smoothing_seasonal': None,  # auto-optimize
            'use_boxcox': False,  # Box-Cox transformation
            'min_periods': 50,  # minimum periods for training
            'forecast_horizon': 1  # 1 step ahead (5 minutes)
        }
        
        if config:
            default_config.update(config)
        
        super().__init__("ETS", default_config)
        
        # ETS-specific attributes
        self.ets_model = None
        self.fitted_model = None
        self.decomposition = None
        self.level_component = None
        self.trend_component = None
        self.seasonal_component = None
        
    async def train(self, features: List[FeatureSet], targets: List[float]) -> bool:
        """Train ETS model with price data"""
        try:
            if len(targets) < self.config['min_periods']:
                logger.warning(f"Insufficient data for ETS training: {len(targets)} < {self.config['min_periods']}")
                return False
            
            logger.info(f"Training ETS model with {len(targets)} data points")
            
            # Convert to pandas Series with datetime index
            timestamps = [fs.timestamp for fs in features]
            price_series = pd.Series(targets, index=pd.DatetimeIndex(timestamps))
            
            # Remove any NaN or infinite values
            price_series = price_series.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(price_series) < self.config['min_periods']:
                logger.warning("Insufficient valid data after cleaning")
                return False
            
            # Perform seasonal decomposition for analysis
            try:
                if len(price_series) >= self.config['seasonal_periods'] * 2:
                    self.decomposition = seasonal_decompose(
                        price_series, 
                        model='additive',
                        period=self.config['seasonal_periods']
                    )
                    
                    self.level_component = self.decomposition.trend
                    self.seasonal_component = self.decomposition.seasonal
                    
                    logger.info("Seasonal decomposition completed")
            except Exception as e:
                logger.warning(f"Seasonal decomposition failed: {e}")
                self.decomposition = None
            
            # Fit ETS model
            try:
                self.ets_model = ETSModel(
                    price_series,
                    error=self.config['error'],
                    trend=self.config['trend'],
                    seasonal=self.config['seasonal'],
                    seasonal_periods=self.config['seasonal_periods'] if self.config['seasonal'] else None,
                    damped_trend=self.config['damped_trend']
                )
                
                # Fit the model
                self.fitted_model = self.ets_model.fit(
                    smoothing_level=self.config['smoothing_level'],
                    smoothing_trend=self.config['smoothing_trend'],
                    smoothing_seasonal=self.config['smoothing_seasonal'],
                    use_boxcox=self.config['use_boxcox'],
                    remove_bias=True,
                    method='L-BFGS-B'
                )
                
                # Calculate model performance metrics
                fitted_values = self.fitted_model.fittedvalues
                residuals = price_series - fitted_values
                
                mae = np.mean(np.abs(residuals))
                rmse = np.sqrt(np.mean(residuals**2))
                mape = np.mean(np.abs(residuals / price_series)) * 100
                
                # Update training history
                metrics = {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'aic': self.fitted_model.aic,
                    'bic': self.fitted_model.bic,
                    'data_points': len(price_series)
                }
                
                self.update_training_history(metrics)
                
                # Extract feature importance (ETS components)
                self.feature_importance = {
                    'level': abs(self.fitted_model.params.get('smoothing_level', 0)),
                    'trend': abs(self.fitted_model.params.get('smoothing_trend', 0)),
                    'seasonal': abs(self.fitted_model.params.get('smoothing_seasonal', 0)),
                    'damping': abs(self.fitted_model.params.get('damping_trend', 0))
                }
                
                self.is_trained = True
                self.last_training_time = datetime.now()
                
                logger.info(f"ETS model trained successfully - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
                return True
                
            except Exception as e:
                logger.error(f"ETS model fitting failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"ETS training failed: {e}")
            return False
    
    async def predict(self, features: FeatureSet) -> Tuple[float, float]:
        """Make prediction using ETS model"""
        try:
            if not self.is_trained or not self.fitted_model:
                logger.warning("ETS model not trained")
                return 0.0, 0.0
            
            # Make forecast
            forecast = self.fitted_model.forecast(steps=self.config['forecast_horizon'])
            
            if isinstance(forecast, pd.Series):
                prediction = float(forecast.iloc[0])
            else:
                prediction = float(forecast)
            
            # Calculate confidence based on prediction interval
            try:
                # Get prediction intervals
                forecast_ci = self.fitted_model.get_prediction(
                    start=len(self.fitted_model.fittedvalues),
                    end=len(self.fitted_model.fittedvalues) + self.config['forecast_horizon'] - 1
                ).conf_int()
                
                if not forecast_ci.empty:
                    lower_bound = forecast_ci.iloc[0, 0]
                    upper_bound = forecast_ci.iloc[0, 1]
                    
                    # Calculate confidence based on interval width
                    interval_width = upper_bound - lower_bound
                    relative_width = interval_width / abs(prediction) if prediction != 0 else 1.0
                    
                    # Convert to confidence score (narrower interval = higher confidence)
                    confidence = max(0.1, min(0.9, 1.0 - min(relative_width, 1.0)))
                else:
                    confidence = 0.5
                    
            except Exception as e:
                logger.warning(f"Confidence calculation failed: {e}")
                confidence = self.calculate_prediction_confidence(prediction, features)
            
            # Ensure prediction is positive (prices can't be negative)
            prediction = max(prediction, 0.01)
            
            logger.debug(f"ETS prediction: {prediction:.2f}, confidence: {confidence:.3f}")
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"ETS prediction failed: {e}")
            return 0.0, 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ETS model information"""
        info = {
            'model_name': self.model_name,
            'model_type': 'ETS (Error, Trend, Seasonality)',
            'is_trained': self.is_trained,
            'config': self.config.copy(),
            'feature_importance': self.feature_importance.copy(),
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None
        }
        
        if self.fitted_model:
            info.update({
                'model_params': dict(self.fitted_model.params),
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'log_likelihood': self.fitted_model.llf
            })
        
        if self.training_history:
            latest_metrics = self.training_history[-1]['metrics']
            info['latest_metrics'] = latest_metrics
        
        return info
    
    def get_components(self) -> Dict[str, Any]:
        """Get ETS components (level, trend, seasonal)"""
        components = {}
        
        if self.fitted_model:
            # Get state components
            states = self.fitted_model.states
            
            if states is not None:
                components['level'] = states.iloc[-1, 0] if len(states.columns) > 0 else None
                components['trend'] = states.iloc[-1, 1] if len(states.columns) > 1 else None
                
                if len(states.columns) > 2:  # Has seasonal component
                    seasonal_cols = states.columns[2:]
                    components['seasonal'] = states.iloc[-1, seasonal_cols].tolist()
        
        if self.decomposition:
            components['decomposition'] = {
                'trend': self.decomposition.trend.iloc[-10:].tolist() if self.decomposition.trend is not None else None,
                'seasonal': self.decomposition.seasonal.iloc[-10:].tolist() if self.decomposition.seasonal is not None else None,
                'residual': self.decomposition.resid.iloc[-10:].tolist() if self.decomposition.resid is not None else None
            }
        
        return components
    
    def detect_trend_change(self, recent_prices: List[float]) -> Dict[str, Any]:
        """Detect trend changes in recent price data"""
        try:
            if len(recent_prices) < 10:
                return {'trend_change': False, 'confidence': 0.0}
            
            # Calculate short and long term trends
            short_trend = np.polyfit(range(5), recent_prices[-5:], 1)[0]
            long_trend = np.polyfit(range(10), recent_prices[-10:], 1)[0]
            
            # Detect trend change
            trend_change = (short_trend * long_trend) < 0  # Opposite signs
            
            # Calculate confidence in trend change
            trend_strength = abs(short_trend - long_trend) / (np.mean(recent_prices) + 1e-10)
            confidence = min(1.0, trend_strength * 100)
            
            return {
                'trend_change': trend_change,
                'short_trend': short_trend,
                'long_trend': long_trend,
                'confidence': confidence,
                'trend_direction': 'up' if short_trend > 0 else 'down'
            }
            
        except Exception as e:
            logger.error(f"Trend change detection failed: {e}")
            return {'trend_change': False, 'confidence': 0.0}
    
    def get_smoothing_parameters(self) -> Dict[str, float]:
        """Get optimized smoothing parameters"""
        if not self.fitted_model:
            return {}
        
        params = {}
        if hasattr(self.fitted_model, 'params'):
            for param_name, param_value in self.fitted_model.params.items():
                if 'smoothing' in param_name or 'damping' in param_name:
                    params[param_name] = float(param_value)
        
        return params