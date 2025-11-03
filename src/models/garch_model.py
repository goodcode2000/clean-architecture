"""
GARCH Model for BTC Price Prediction
Models time-varying volatility and shocks, captures heteroskedasticity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from arch import arch_model
# GJR_GARCH is not exposed in all versions of the `arch` package.
# Guard the import so the module can still be imported when GJR_GARCH
# symbol is not available in the installed `arch` version.
try:
    from arch.univariate import GARCH, EGARCH, GJR_GARCH
except ImportError:
    try:
        from arch.univariate import GARCH, EGARCH
        GJR_GARCH = None
    except Exception:
        # If even GARCH/EGARCH are unavailable, set them to None so
        # importing the module doesn't raise and we can report errors
        # at runtime when trying to use the functionality.
        GARCH = EGARCH = GJR_GARCH = None
from .base_model import BaseModel
from ..feature_engineering import FeatureSet

logger = logging.getLogger(__name__)

class GARCHPredictionModel(BaseModel):
    """GARCH model for volatility modeling and price prediction"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'model_type': 'GARCH',  # GARCH, EGARCH, GJR-GARCH
            'p': 1,  # GARCH lag order
            'q': 1,  # ARCH lag order
            'mean_model': 'AR',  # Mean model: AR, HAR, LS, Zero
            'ar_lags': 1,  # AR lags for mean model
            'distribution': 't',  # Error distribution: normal, t, skewt
            'vol_model': 'GARCH',  # Volatility model
            'min_periods': 100,  # Minimum periods for training
            'forecast_horizon': 1,  # Forecast horizon
            'rescale': True,  # Rescale returns
            'update_freq': 5  # Update frequency for rolling forecasts
        }
        
        if config:
            default_config.update(config)
        
        super().__init__("GARCH", default_config)
        
        # GARCH-specific attributes
        self.garch_model = None
        self.fitted_model = None
        self.returns_series = None
        self.volatility_forecast = None
        self.conditional_volatility = None
        self.standardized_residuals = None
        
    async def train(self, features: List[FeatureSet], targets: List[float]) -> bool:
        """Train GARCH model with price data"""
        try:
            if len(targets) < self.config['min_periods']:
                logger.warning(f"Insufficient data for GARCH training: {len(targets)} < {self.config['min_periods']}")
                return False
            
            logger.info(f"Training GARCH model with {len(targets)} data points")
            
            # Convert prices to returns
            prices = np.array(targets)
            returns = np.diff(np.log(prices)) * 100  # Log returns in percentage
            
            # Remove any NaN or infinite values
            returns = returns[np.isfinite(returns)]
            
            if len(returns) < self.config['min_periods'] - 1:
                logger.warning("Insufficient valid returns after cleaning")
                return False
            
            # Create pandas Series
            timestamps = [fs.timestamp for fs in features[1:]]  # Skip first timestamp due to diff
            self.returns_series = pd.Series(returns, index=pd.DatetimeIndex(timestamps))
            
            # Create GARCH model
            try:
                if self.config['mean_model'] == 'AR':
                    mean_model = 'AR'
                    lags = self.config['ar_lags']
                elif self.config['mean_model'] == 'Zero':
                    mean_model = 'Zero'
                    lags = None
                else:
                    mean_model = 'Constant'
                    lags = None
                
                # Initialize GARCH model
                self.garch_model = arch_model(
                    self.returns_series,
                    mean=mean_model,
                    lags=lags,
                    vol=self.config['vol_model'],
                    p=self.config['p'],
                    q=self.config['q'],
                    dist=self.config['distribution'],
                    rescale=self.config['rescale']
                )
                
                # Fit the model
                self.fitted_model = self.garch_model.fit(
                    update_freq=self.config['update_freq'],
                    disp='off'  # Suppress output
                )
                
                # Extract model components
                self.conditional_volatility = self.fitted_model.conditional_volatility
                self.standardized_residuals = self.fitted_model.std_resid
                
                # Calculate performance metrics
                log_likelihood = self.fitted_model.loglikelihood
                aic = self.fitted_model.aic
                bic = self.fitted_model.bic
                
                # Calculate volatility forecasting accuracy
                realized_vol = np.sqrt(252) * self.returns_series.rolling(20).std()  # Annualized
                garch_vol = self.conditional_volatility * np.sqrt(252) / 100  # Convert to annualized
                
                # Align series for comparison
                common_index = realized_vol.dropna().index.intersection(garch_vol.index)
                if len(common_index) > 10:
                    vol_mae = np.mean(np.abs(realized_vol[common_index] - garch_vol[common_index]))
                    vol_rmse = np.sqrt(np.mean((realized_vol[common_index] - garch_vol[common_index])**2))
                else:
                    vol_mae = vol_rmse = np.nan
                
                # Update training history
                metrics = {
                    'log_likelihood': log_likelihood,
                    'aic': aic,
                    'bic': bic,
                    'volatility_mae': vol_mae,
                    'volatility_rmse': vol_rmse,
                    'data_points': len(returns)
                }
                
                self.update_training_history(metrics)
                
                # Extract feature importance (parameter significance)
                self.feature_importance = {}
                if hasattr(self.fitted_model, 'params'):
                    for param_name, param_value in self.fitted_model.params.items():
                        # Use absolute t-statistic as importance measure
                        if hasattr(self.fitted_model, 'tvalues') and param_name in self.fitted_model.tvalues:
                            importance = abs(self.fitted_model.tvalues[param_name])
                        else:
                            importance = abs(param_value)
                        self.feature_importance[param_name] = importance
                
                self.is_trained = True
                self.last_training_time = datetime.now()
                
                logger.info(f"GARCH model trained successfully - AIC: {aic:.2f}, BIC: {bic:.2f}")
                return True
                
            except Exception as e:
                logger.error(f"GARCH model fitting failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"GARCH training failed: {e}")
            return False
    
    async def predict(self, features: FeatureSet) -> Tuple[float, float]:
        """Make prediction using GARCH model"""
        try:
            if not self.is_trained or not self.fitted_model:
                logger.warning("GARCH model not trained")
                return 0.0, 0.0
            
            # Get current price from features
            current_price = features.price_features.get('price', 0)
            if current_price <= 0:
                logger.warning("Invalid current price for GARCH prediction")
                return 0.0, 0.0
            
            # Make volatility forecast
            volatility_forecast = self.fitted_model.forecast(
                horizon=self.config['forecast_horizon'],
                method='simulation',
                simulations=1000
            )
            
            # Extract forecasted volatility
            if hasattr(volatility_forecast, 'variance'):
                forecasted_variance = volatility_forecast.variance.iloc[-1, 0]
                forecasted_volatility = np.sqrt(forecasted_variance)
            else:
                forecasted_volatility = self.conditional_volatility.iloc[-1]
            
            # Make return forecast (mean reversion assumption)
            if hasattr(volatility_forecast, 'mean'):
                forecasted_return = volatility_forecast.mean.iloc[-1, 0]
            else:
                # Simple mean reversion to long-term mean
                long_term_mean = self.returns_series.mean()
                recent_return = self.returns_series.iloc[-1]
                forecasted_return = long_term_mean + 0.1 * (recent_return - long_term_mean)
            
            # Convert return forecast to price forecast
            # Return is in percentage, convert back to price
            forecasted_return_decimal = forecasted_return / 100.0
            predicted_price = current_price * np.exp(forecasted_return_decimal)
            
            # Calculate confidence based on volatility
            # Lower volatility = higher confidence
            normalized_volatility = forecasted_volatility / 100.0  # Convert from percentage
            confidence = max(0.1, min(0.9, 1.0 - min(normalized_volatility, 1.0)))
            
            # Adjust confidence based on model fit quality
            if self.training_history:
                latest_metrics = self.training_history[-1]['metrics']
                if 'volatility_mae' in latest_metrics and not np.isnan(latest_metrics['volatility_mae']):
                    # Lower MAE = higher confidence
                    vol_mae = latest_metrics['volatility_mae']
                    confidence *= max(0.5, 1.0 - min(vol_mae / 10.0, 0.5))
            
            # Store volatility forecast for analysis
            self.volatility_forecast = {
                'forecasted_volatility': forecasted_volatility,
                'forecasted_return': forecasted_return,
                'confidence': confidence,
                'timestamp': datetime.now()
            }
            
            logger.debug(f"GARCH prediction: {predicted_price:.2f}, volatility: {forecasted_volatility:.3f}, confidence: {confidence:.3f}")
            return predicted_price, confidence
            
        except Exception as e:
            logger.error(f"GARCH prediction failed: {e}")
            return 0.0, 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get GARCH model information"""
        info = {
            'model_name': self.model_name,
            'model_type': f'GARCH({self.config["p"]},{self.config["q"]})',
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
                'log_likelihood': self.fitted_model.loglikelihood,
                'num_params': self.fitted_model.num_params
            })
        
        if self.training_history:
            latest_metrics = self.training_history[-1]['metrics']
            info['latest_metrics'] = latest_metrics
        
        if self.volatility_forecast:
            info['latest_volatility_forecast'] = self.volatility_forecast
        
        return info
    
    def get_volatility_analysis(self) -> Dict[str, Any]:
        """Get detailed volatility analysis"""
        analysis = {}
        
        if self.conditional_volatility is not None:
            vol_series = self.conditional_volatility
            
            analysis['current_volatility'] = vol_series.iloc[-1]
            analysis['volatility_stats'] = {
                'mean': vol_series.mean(),
                'std': vol_series.std(),
                'min': vol_series.min(),
                'max': vol_series.max(),
                'percentiles': {
                    '25': vol_series.quantile(0.25),
                    '50': vol_series.quantile(0.50),
                    '75': vol_series.quantile(0.75),
                    '95': vol_series.quantile(0.95)
                }
            }
            
            # Volatility clustering analysis
            vol_changes = vol_series.diff().abs()
            analysis['volatility_clustering'] = {
                'persistence': vol_series.autocorr(lag=1),
                'mean_reversion_speed': -np.log(abs(vol_series.autocorr(lag=1))),
                'clustering_strength': vol_changes.autocorr(lag=1)
            }
        
        if self.standardized_residuals is not None:
            residuals = self.standardized_residuals
            
            analysis['residual_analysis'] = {
                'mean': residuals.mean(),
                'std': residuals.std(),
                'skewness': residuals.skew(),
                'kurtosis': residuals.kurtosis(),
                'jarque_bera_pvalue': self._jarque_bera_test(residuals)
            }
        
        return analysis
    
    def detect_volatility_regime(self) -> Dict[str, Any]:
        """Detect current volatility regime"""
        try:
            if self.conditional_volatility is None or len(self.conditional_volatility) < 20:
                return {'regime': 'unknown', 'confidence': 0.0}
            
            current_vol = self.conditional_volatility.iloc[-1]
            recent_vol = self.conditional_volatility.iloc[-20:].mean()
            long_term_vol = self.conditional_volatility.mean()
            
            # Define regime thresholds
            low_threshold = long_term_vol * 0.7
            high_threshold = long_term_vol * 1.3
            
            if current_vol < low_threshold:
                regime = 'low_volatility'
                confidence = min(1.0, (low_threshold - current_vol) / low_threshold)
            elif current_vol > high_threshold:
                regime = 'high_volatility'
                confidence = min(1.0, (current_vol - high_threshold) / high_threshold)
            else:
                regime = 'normal_volatility'
                confidence = 1.0 - abs(current_vol - long_term_vol) / long_term_vol
            
            return {
                'regime': regime,
                'current_volatility': current_vol,
                'recent_volatility': recent_vol,
                'long_term_volatility': long_term_vol,
                'confidence': confidence,
                'volatility_trend': 'increasing' if current_vol > recent_vol else 'decreasing'
            }
            
        except Exception as e:
            logger.error(f"Volatility regime detection failed: {e}")
            return {'regime': 'unknown', 'confidence': 0.0}
    
    def get_shock_analysis(self) -> Dict[str, Any]:
        """Analyze volatility shocks and their persistence"""
        try:
            if self.conditional_volatility is None or len(self.conditional_volatility) < 50:
                return {}
            
            vol_series = self.conditional_volatility
            vol_changes = vol_series.pct_change().dropna()
            
            # Identify shocks (large volatility changes)
            shock_threshold = vol_changes.std() * 2  # 2 standard deviations
            shocks = vol_changes[abs(vol_changes) > shock_threshold]
            
            if len(shocks) == 0:
                return {'no_shocks_detected': True}
            
            # Analyze shock characteristics
            positive_shocks = shocks[shocks > 0]
            negative_shocks = shocks[shocks < 0]
            
            analysis = {
                'total_shocks': len(shocks),
                'positive_shocks': len(positive_shocks),
                'negative_shocks': len(negative_shocks),
                'shock_frequency': len(shocks) / len(vol_changes),
                'average_shock_magnitude': abs(shocks).mean(),
                'max_positive_shock': positive_shocks.max() if len(positive_shocks) > 0 else 0,
                'max_negative_shock': negative_shocks.min() if len(negative_shocks) > 0 else 0
            }
            
            # Recent shock analysis
            recent_period = min(20, len(vol_changes))
            recent_shocks = vol_changes.iloc[-recent_period:]
            recent_shock_count = sum(abs(recent_shocks) > shock_threshold)
            
            analysis['recent_shock_activity'] = {
                'recent_shocks': recent_shock_count,
                'recent_shock_rate': recent_shock_count / recent_period,
                'elevated_activity': recent_shock_count > len(shocks) / len(vol_changes) * recent_period
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Shock analysis failed: {e}")
            return {}
    
    def _jarque_bera_test(self, residuals: pd.Series) -> float:
        """Perform Jarque-Bera test for normality"""
        try:
            from scipy import stats
            
            # Calculate skewness and kurtosis
            skewness = residuals.skew()
            kurtosis = residuals.kurtosis()
            
            # Jarque-Bera statistic
            n = len(residuals)
            jb_stat = n * (skewness**2 / 6 + (kurtosis - 3)**2 / 24)
            
            # P-value from chi-square distribution with 2 degrees of freedom
            p_value = 1 - stats.chi2.cdf(jb_stat, 2)
            
            return p_value
            
        except Exception as e:
            logger.warning(f"Jarque-Bera test failed: {e}")
            return np.nan