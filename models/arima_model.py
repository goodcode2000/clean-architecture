"""ARIMA model for BTC price prediction."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Any
from loguru import logger
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import statsmodels.api as sm
except ImportError:
    logger.error("statsmodels not installed. Install with: pip install statsmodels")
    raise

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class ARIMAPredictor:
    """ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting."""
    
    def __init__(self, max_p=5, max_d=2, max_q=5):
        self.model = None
        self.fitted_model = None
        self.model_params = {'p': 1, 'd': 1, 'q': 1}  # Default ARIMA(1,1,1)
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.is_trained = False
        self.last_training_data = None
        self.prediction_horizon = 1  # Predict 1 step ahead (5 minutes)
        self.aic_score = None
        self.bic_score = None
        self.residuals = None
        
    def prepare_data(self, df: pd.DataFrame) -> pd.Series:
        """
        Prepare data for ARIMA model training.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Time series data for ARIMA
        """
        try:
            # Use close prices as time series
            if 'timestamp' in df.columns:
                ts_data = pd.Series(
                    data=df['close'].values,
                    index=pd.to_datetime(df['timestamp'])
                )
            else:
                ts_data = pd.Series(df['close'].values)
            
            # Remove any NaN values
            ts_data = ts_data.dropna()
            
            # Ensure we have enough data
            if len(ts_data) < 50:
                logger.warning(f"Limited data for ARIMA: {len(ts_data)} observations")
            
            logger.debug(f"Prepared ARIMA data: {len(ts_data)} observations")
            return ts_data
            
        except Exception as e:
            logger.error(f"Failed to prepare ARIMA data: {e}")
            return pd.Series()
    
    def check_stationarity(self, ts_data: pd.Series) -> Tuple[bool, int]:
        """
        Check if time series is stationary and determine differencing order.
        
        Args:
            ts_data: Time series data
            
        Returns:
            Tuple of (is_stationary, suggested_d)
        """
        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(ts_data.dropna())
            adf_pvalue = adf_result[1]
            
            # KPSS test (null hypothesis: series is stationary)
            try:
                kpss_result = kpss(ts_data.dropna(), regression='c')
                kpss_pvalue = kpss_result[1]
            except:
                kpss_pvalue = 0.1  # Default if KPSS fails
            
            # Series is stationary if ADF p-value < 0.05 AND KPSS p-value > 0.05
            is_stationary = (adf_pvalue < 0.05) and (kpss_pvalue > 0.05)
            
            # Suggest differencing order
            if is_stationary:
                suggested_d = 0
            else:
                # Try first difference
                diff_data = ts_data.diff().dropna()
                if len(diff_data) > 10:
                    adf_diff = adfuller(diff_data)[1]
                    if adf_diff < 0.05:
                        suggested_d = 1
                    else:
                        suggested_d = 2  # Second difference
                else:
                    suggested_d = 1
            
            logger.debug(f"Stationarity check - ADF p-value: {adf_pvalue:.4f}, "
                        f"KPSS p-value: {kpss_pvalue:.4f}, Stationary: {is_stationary}, "
                        f"Suggested d: {suggested_d}")
            
            return is_stationary, suggested_d
            
        except Exception as e:
            logger.error(f"Stationarity check failed: {e}")
            return False, 1
    
    def auto_select_parameters(self, ts_data: pd.Series) -> Tuple[int, int, int]:
        """
        Automatically select optimal ARIMA parameters using AIC/BIC criteria.
        
        Args:
            ts_data: Time series data
            
        Returns:
            Tuple of (p, d, q) parameters
        """
        try:
            logger.info("Auto-selecting ARIMA parameters...")
            
            # Check stationarity and get suggested d
            _, suggested_d = self.check_stationarity(ts_data)
            
            best_aic = float('inf')
            best_bic = float('inf')
            best_params_aic = (1, 1, 1)
            best_params_bic = (1, 1, 1)
            
            # Grid search over parameter space
            for p in range(0, min(self.max_p + 1, 6)):
                for d in range(0, min(self.max_d + 1, 3)):
                    for q in range(0, min(self.max_q + 1, 6)):
                        try:
                            # Skip if d is too high for the data
                            if d >= len(ts_data) - 10:
                                continue
                                
                            # Fit ARIMA model
                            model = ARIMA(ts_data, order=(p, d, q))
                            fitted = model.fit()
                            
                            # Check AIC and BIC
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_params_aic = (p, d, q)
                            
                            if fitted.bic < best_bic:
                                best_bic = fitted.bic
                                best_params_bic = (p, d, q)
                                
                        except Exception as e:
                            logger.debug(f"ARIMA({p},{d},{q}) failed: {e}")
                            continue
            
            # Choose parameters based on AIC (generally better for prediction)
            best_params = best_params_aic
            
            # Fallback to suggested d if all models failed
            if best_aic == float('inf'):
                best_params = (1, suggested_d, 1)
                logger.warning(f"Parameter selection failed, using fallback: {best_params}")
            
            logger.info(f"Selected ARIMA{best_params} with AIC: {best_aic:.2f}, BIC: {best_bic:.2f}")
            return best_params
            
        except Exception as e:
            logger.error(f"Auto parameter selection failed: {e}")
            return (1, 1, 1)  # Default fallback
    
    def train(self, df: pd.DataFrame) -> bool:
        """
        Train the ARIMA model.
        
        Args:
            df: DataFrame with historical price data
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            logger.info("Training ARIMA model...")
            
            # Prepare time series data
            ts_data = self.prepare_data(df)
            
            if len(ts_data) < 50:  # Minimum data requirement
                logger.error(f"Insufficient data for ARIMA training: {len(ts_data)} observations")
                return False
            
            # Store training data for incremental updates
            self.last_training_data = ts_data.copy()
            
            # Auto-select parameters
            p, d, q = self.auto_select_parameters(ts_data)
            self.model_params = {'p': p, 'd': d, 'q': q}
            
            # Fit ARIMA model with selected parameters
            try:
                self.model = ARIMA(ts_data, order=(p, d, q))
                self.fitted_model = self.model.fit()
                
                # Store model statistics
                self.aic_score = self.fitted_model.aic
                self.bic_score = self.fitted_model.bic
                self.residuals = self.fitted_model.resid
                
                self.is_trained = True
                
                logger.info(f"ARIMA{(p, d, q)} model trained successfully")
                logger.info(f"AIC: {self.aic_score:.2f}, BIC: {self.bic_score:.2f}")
                
                # Diagnostic checks
                self.perform_diagnostic_checks()
                
                return True
                
            except Exception as e:
                logger.error(f"ARIMA model fitting failed: {e}")
                
                # Try fallback parameters
                logger.info("Trying fallback ARIMA(1,1,1) parameters...")
                try:
                    self.model = ARIMA(ts_data, order=(1, 1, 1))
                    self.fitted_model = self.model.fit()
                    self.model_params = {'p': 1, 'd': 1, 'q': 1}
                    self.aic_score = self.fitted_model.aic
                    self.bic_score = self.fitted_model.bic
                    self.residuals = self.fitted_model.resid
                    self.is_trained = True
                    
                    logger.info("Fallback ARIMA(1,1,1) model trained successfully")
                    return True
                    
                except Exception as e2:
                    logger.error(f"Fallback ARIMA training also failed: {e2}")
                    return False
            
        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
            return False
    
    def perform_diagnostic_checks(self):
        """Perform diagnostic checks on the fitted model."""
        try:
            if self.residuals is None or len(self.residuals) < 10:
                return
            
            # Ljung-Box test for residual autocorrelation
            try:
                lb_test = acorr_ljungbox(self.residuals, lags=10, return_df=True)
                lb_pvalue = lb_test['lb_pvalue'].iloc[-1]
                
                if lb_pvalue < 0.05:
                    logger.warning(f"Ljung-Box test suggests residual autocorrelation (p={lb_pvalue:.4f})")
                else:
                    logger.debug(f"Ljung-Box test passed (p={lb_pvalue:.4f})")
                    
            except Exception as e:
                logger.debug(f"Ljung-Box test failed: {e}")
            
            # Check residual statistics
            residual_mean = np.mean(self.residuals)
            residual_std = np.std(self.residuals)
            
            logger.debug(f"Residual statistics - Mean: {residual_mean:.4f}, Std: {residual_std:.2f}")
            
        except Exception as e:
            logger.debug(f"Diagnostic checks failed: {e}")
    
    def predict(self, steps: int = 1) -> Tuple[float, Tuple[float, float]]:
        """
        Make prediction using the trained ARIMA model.
        
        Args:
            steps: Number of steps ahead to predict
            
        Returns:
            Tuple of (prediction, confidence_interval)
        """
        try:
            if not self.is_trained or self.fitted_model is None:
                logger.error("ARIMA model not trained")
                return 0.0, (0.0, 0.0)
            
            # Make forecast with confidence intervals
            forecast_result = self.fitted_model.get_forecast(steps=steps)
            
            # Get point forecast
            forecast = forecast_result.predicted_mean
            
            # Get confidence intervals
            conf_int = forecast_result.conf_int()
            
            if steps == 1:
                prediction = float(forecast.iloc[0])
                lower_bound = float(conf_int.iloc[0, 0])
                upper_bound = float(conf_int.iloc[0, 1])
            else:
                prediction = float(forecast.iloc[-1])
                lower_bound = float(conf_int.iloc[-1, 0])
                upper_bound = float(conf_int.iloc[-1, 1])
            
            # Ensure confidence interval is reasonable
            if lower_bound >= upper_bound or np.isnan(lower_bound) or np.isnan(upper_bound):
                # Fallback confidence interval based on residual standard error
                if self.residuals is not None and len(self.residuals) > 0:
                    residual_std = np.std(self.residuals)
                    lower_bound = prediction - 1.96 * residual_std
                    upper_bound = prediction + 1.96 * residual_std
                else:
                    margin = abs(prediction) * 0.05
                    lower_bound = prediction - margin
                    upper_bound = prediction + margin
            
            logger.debug(f"ARIMA prediction: {prediction:.2f} [{lower_bound:.2f}, {upper_bound:.2f}]")
            return prediction, (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
            return 0.0, (0.0, 0.0)
    
    def update_model(self, new_data: pd.DataFrame) -> bool:
        """
        Update the model with new data (incremental learning).
        
        Args:
            new_data: DataFrame with new price data
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if not self.is_trained:
                logger.warning("Model not trained, performing full training")
                return self.train(new_data)
            
            # Prepare new time series data
            new_ts = self.prepare_data(new_data)
            
            if len(new_ts) == 0:
                logger.warning("No new data to update ARIMA model")
                return True
            
            # Combine with existing data
            if self.last_training_data is not None:
                # Keep only recent data to avoid memory issues and maintain stationarity
                max_history = 1000  # Keep last 1000 observations
                combined_data = pd.concat([self.last_training_data, new_ts])
                combined_data = combined_data.tail(max_history)
            else:
                combined_data = new_ts
            
            # Check if we need to retrain with new parameters
            # For now, use same parameters but could implement adaptive parameter selection
            try:
                # Update with same parameters
                p, d, q = self.model_params['p'], self.model_params['d'], self.model_params['q']
                
                self.model = ARIMA(combined_data, order=(p, d, q))
                self.fitted_model = self.model.fit()
                
                # Update statistics
                self.aic_score = self.fitted_model.aic
                self.bic_score = self.fitted_model.bic
                self.residuals = self.fitted_model.resid
                
                # Update training data
                self.last_training_data = combined_data
                
                logger.debug(f"ARIMA model updated with {len(new_ts)} new observations")
                return True
                
            except Exception as e:
                logger.warning(f"ARIMA update with same parameters failed: {e}")
                
                # Fallback: retrain with parameter selection
                logger.info("Retraining ARIMA with new parameter selection...")
                self.last_training_data = combined_data
                
                # Auto-select new parameters
                p, d, q = self.auto_select_parameters(combined_data)
                self.model_params = {'p': p, 'd': d, 'q': q}
                
                self.model = ARIMA(combined_data, order=(p, d, q))
                self.fitted_model = self.model.fit()
                
                self.aic_score = self.fitted_model.aic
                self.bic_score = self.fitted_model.bic
                self.residuals = self.fitted_model.resid
                
                logger.info(f"ARIMA model retrained with new parameters: {(p, d, q)}")
                return True
            
        except Exception as e:
            logger.error(f"ARIMA model update failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': 'ARIMA',
            'is_trained': self.is_trained,
            'parameters': self.model_params.copy(),
            'training_data_size': len(self.last_training_data) if self.last_training_data is not None else 0
        }
        
        if self.is_trained and self.fitted_model is not None:
            info.update({
                'aic': self.aic_score,
                'bic': self.bic_score,
                'log_likelihood': getattr(self.fitted_model, 'llf', None),
                'sigma2': getattr(self.fitted_model, 'sigma2', None),
                'residual_std': np.std(self.residuals) if self.residuals is not None else None
            })
        
        return info
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model to file.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if not self.is_trained:
                logger.error("Cannot save untrained ARIMA model")
                return False
            
            model_data = {
                'fitted_model': self.fitted_model,
                'model_params': self.model_params,
                'last_training_data': self.last_training_data,
                'is_trained': self.is_trained,
                'aic_score': self.aic_score,
                'bic_score': self.bic_score,
                'residuals': self.residuals
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"ARIMA model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save ARIMA model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"ARIMA model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.fitted_model = model_data['fitted_model']
            self.model_params = model_data['model_params']
            self.last_training_data = model_data['last_training_data']
            self.is_trained = model_data['is_trained']
            self.aic_score = model_data.get('aic_score')
            self.bic_score = model_data.get('bic_score')
            self.residuals = model_data.get('residuals')
            
            logger.info(f"ARIMA model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ARIMA model: {e}")
            return False
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: DataFrame with test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if not self.is_trained:
                logger.error("Cannot evaluate untrained ARIMA model")
                return {}
            
            test_ts = self.prepare_data(test_data)
            
            if len(test_ts) == 0:
                logger.error("No test data available")
                return {}
            
            predictions = []
            actuals = test_ts.values
            
            # Make one-step-ahead predictions
            for i in range(len(test_ts)):
                pred, _ = self.predict(steps=1)
                predictions.append(pred)
                
                # Update model with actual value for next prediction
                if i < len(test_ts) - 1:
                    new_point = pd.DataFrame({
                        'close': [actuals[i]],
                        'timestamp': [test_ts.index[i]]
                    })
                    self.update_model(new_point)
            
            predictions = np.array(predictions)
            
            # Calculate metrics
            mae = np.mean(np.abs(predictions - actuals))
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
            mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
            
            # Directional accuracy
            if len(actuals) > 1:
                actual_direction = np.sign(np.diff(actuals))
                pred_direction = np.sign(predictions[1:] - actuals[:-1])
                directional_accuracy = np.mean(actual_direction == pred_direction) * 100
            else:
                directional_accuracy = 0.0
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'directional_accuracy': directional_accuracy,
                'mean_prediction': np.mean(predictions),
                'mean_actual': np.mean(actuals)
            }
            
            logger.info(f"ARIMA evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"ARIMA model evaluation failed: {e}")
            return {}