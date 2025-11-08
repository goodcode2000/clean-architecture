"""ETS (Exponential Smoothing) model for BTC price prediction."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Any
from loguru import logger
import joblib
import os
import sys
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class ETSPredictor:
    """ETS (Exponential Smoothing) model for capturing trends and seasonality."""
    
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.model_params = {
            'trend': 'add',  # additive trend
            'seasonal': None,  # no seasonality for 5-minute data
            'seasonal_periods': None,
            'damped_trend': True,  # damped trend for stability
            'use_boxcox': False,
            'initialization_method': 'estimated'
        }
        self.is_trained = False
        self.last_training_data = None
        self.prediction_horizon = 1  # Predict 1 step ahead (5 minutes)
        
    def prepare_data(self, df: pd.DataFrame) -> pd.Series:
        """
        Prepare data for ETS model training.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Time series data for ETS
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
            
            # Ensure positive values (ETS requires positive values)
            if (ts_data <= 0).any():
                logger.warning("Non-positive values found, adding small constant")
                ts_data = ts_data + abs(ts_data.min()) + 1
            
            logger.debug(f"Prepared ETS data: {len(ts_data)} observations")
            return ts_data
            
        except Exception as e:
            logger.error(f"Failed to prepare ETS data: {e}")
            return pd.Series()
    
    def train(self, df: pd.DataFrame) -> bool:
        """
        Train the ETS model.
        
        Args:
            df: DataFrame with historical price data
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            logger.info("Training ETS model...")
            
            # Prepare time series data
            ts_data = self.prepare_data(df)
            
            if len(ts_data) < 50:  # Minimum data requirement
                logger.error(f"Insufficient data for ETS training: {len(ts_data)} observations")
                return False
            
            # Store training data for incremental updates
            self.last_training_data = ts_data.copy()
            
            # Try different ETS configurations
            configs_to_try = [
                {'trend': 'add', 'seasonal': None, 'damped_trend': True},
                {'trend': 'add', 'seasonal': None, 'damped_trend': False},
                {'trend': 'mul', 'seasonal': None, 'damped_trend': True},
                {'trend': None, 'seasonal': None, 'damped_trend': False},
            ]
            
            best_model = None
            best_aic = float('inf')
            best_config = None
            
            for config in configs_to_try:
                try:
                    # Create and fit model
                    model = ExponentialSmoothing(
                        ts_data,
                        trend=config['trend'],
                        seasonal=config['seasonal'],
                        damped_trend=config['damped_trend'],
                        initialization_method='estimated'
                    )
                    
                    fitted = model.fit(optimized=True, use_brute=False)
                    
                    # Check AIC for model selection
                    if hasattr(fitted, 'aic') and fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_model = fitted
                        best_config = config
                        
                except Exception as e:
                    logger.debug(f"ETS config {config} failed: {e}")
                    continue
            
            if best_model is None:
                logger.error("All ETS configurations failed")
                return False
            
            self.fitted_model = best_model
            self.model_params.update(best_config)
            self.is_trained = True
            
            logger.info(f"ETS model trained successfully with config: {best_config}")
            logger.info(f"Model AIC: {best_aic:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"ETS training failed: {e}")
            return False
    
    def predict(self, steps: int = 1) -> Tuple[float, Tuple[float, float]]:
        """
        Make prediction using the trained ETS model.
        
        Args:
            steps: Number of steps ahead to predict
            
        Returns:
            Tuple of (prediction, confidence_interval)
        """
        try:
            if not self.is_trained or self.fitted_model is None:
                logger.error("ETS model not trained")
                return 0.0, (0.0, 0.0)
            
            # Make forecast
            forecast = self.fitted_model.forecast(steps=steps)
            
            # Get confidence intervals if available
            try:
                forecast_ci = self.fitted_model.get_prediction(
                    start=len(self.last_training_data),
                    end=len(self.last_training_data) + steps - 1
                )
                conf_int = forecast_ci.conf_int()
                
                if steps == 1:
                    prediction = float(forecast[0])
                    lower_bound = float(conf_int.iloc[0, 0])
                    upper_bound = float(conf_int.iloc[0, 1])
                else:
                    prediction = float(forecast[-1])
                    lower_bound = float(conf_int.iloc[-1, 0])
                    upper_bound = float(conf_int.iloc[-1, 1])
                    
            except Exception:
                # Fallback: use simple confidence interval based on residuals
                if steps == 1:
                    prediction = float(forecast[0])
                else:
                    prediction = float(forecast[-1])
                
                # Estimate confidence interval from residuals
                residuals = self.fitted_model.resid
                residual_std = residuals.std()
                
                lower_bound = prediction - 1.96 * residual_std
                upper_bound = prediction + 1.96 * residual_std
            
            logger.debug(f"ETS prediction: {prediction:.2f} [{lower_bound:.2f}, {upper_bound:.2f}]")
            return prediction, (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"ETS prediction failed: {e}")
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
                logger.warning("No new data to update ETS model")
                return True
            
            # Combine with existing data
            if self.last_training_data is not None:
                # Keep only recent data to avoid memory issues
                max_history = 2000  # Keep last 2000 observations
                combined_data = pd.concat([self.last_training_data, new_ts])
                combined_data = combined_data.tail(max_history)
            else:
                combined_data = new_ts
            
            # Retrain with combined data
            self.last_training_data = combined_data
            
            # Create new model with same parameters
            model = ExponentialSmoothing(
                combined_data,
                trend=self.model_params.get('trend'),
                seasonal=self.model_params.get('seasonal'),
                damped_trend=self.model_params.get('damped_trend', True),
                initialization_method='estimated'
            )
            
            self.fitted_model = model.fit(optimized=True, use_brute=False)
            
            logger.debug(f"ETS model updated with {len(new_ts)} new observations")
            return True
            
        except Exception as e:
            logger.error(f"ETS model update failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': 'ETS',
            'is_trained': self.is_trained,
            'parameters': self.model_params.copy(),
            'training_data_size': len(self.last_training_data) if self.last_training_data is not None else 0
        }
        
        if self.is_trained and self.fitted_model is not None:
            try:
                info.update({
                    'aic': getattr(self.fitted_model, 'aic', None),
                    'bic': getattr(self.fitted_model, 'bic', None),
                    'sse': getattr(self.fitted_model, 'sse', None),
                    'smoothing_level': getattr(self.fitted_model.params, 'smoothing_level', None),
                    'smoothing_trend': getattr(self.fitted_model.params, 'smoothing_trend', None),
                    'damping_trend': getattr(self.fitted_model.params, 'damping_trend', None)
                })
            except Exception as e:
                logger.debug(f"Could not get detailed ETS info: {e}")
        
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
                logger.error("Cannot save untrained ETS model")
                return False
            
            model_data = {
                'fitted_model': self.fitted_model,
                'model_params': self.model_params,
                'last_training_data': self.last_training_data,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"ETS model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save ETS model: {e}")
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
                logger.error(f"ETS model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.fitted_model = model_data['fitted_model']
            self.model_params = model_data['model_params']
            self.last_training_data = model_data['last_training_data']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"ETS model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ETS model: {e}")
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
                logger.error("Cannot evaluate untrained ETS model")
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
            actual_direction = np.sign(np.diff(actuals))
            pred_direction = np.sign(predictions[1:] - actuals[:-1])
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'directional_accuracy': directional_accuracy,
                'mean_prediction': np.mean(predictions),
                'mean_actual': np.mean(actuals)
            }
            
            logger.info(f"ETS evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"ETS model evaluation failed: {e}")
            return {}