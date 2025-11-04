"""SVR (Support Vector Regression) model for BTC price prediction."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Any
from loguru import logger
import joblib
import os
import sys
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class SVRPredictor:
    """SVR model for capturing non-linear relationships in BTC price data."""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_columns = []
        self.hyperparameters = {
            'kernel': 'rbf',
            'C': 100.0,
            'gamma': 'scale',
            'epsilon': 0.1,
            'cache_size': 1000
        }
        self.best_params = None
        self.training_score = None
    self.feature_importance = {}
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for SVR training.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple of (X, y) arrays
        """
        try:
            # Select important features for SVR
            important_features = [
                'close', 'open', 'high', 'low', 'volume',
                'rsi', 'macd', 'macd_signal', 'bb_position', 'bb_width',
                'sma_5', 'sma_20', 'ema_12', 'ema_26',
                'volatility_20', 'atr', 'returns',
                'close_lag_1', 'close_lag_2', 'close_lag_5',
                'returns_lag_1', 'returns_lag_2',
                'close_mean_5', 'close_std_5', 'close_position_20',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
            ]
            
            # Filter for available features
            available_features = [col for col in important_features if col in df.columns]
            
            if len(available_features) < 5:
                logger.error(f"Insufficient features for SVR: {len(available_features)}")
                return np.array([]), np.array([])
            
            # Exclude target from features
            feature_cols = [col for col in available_features if col != 'close']
            
            X = df[feature_cols].values
            y = df['close'].values
            
            # Remove any rows with NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            self.feature_columns = feature_cols
            
            logger.debug(f"SVR features prepared: {X.shape[1]} features, {X.shape[0]} samples")
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to prepare SVR features: {e}")
            return np.array([]), np.array([])
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize SVR hyperparameters using grid search.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Best hyperparameters
        """
        try:
            logger.info("Optimizing SVR hyperparameters...")
            
            # Define parameter grid
            param_grid = {
                'C': [1, 10, 100, 1000],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'epsilon': [0.01, 0.1, 0.2, 0.5],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
            
            # Use TimeSeriesSplit for time series data
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Create SVR model
            svr = SVR(cache_size=1000)
            
            # Perform grid search
            grid_search = GridSearchCV(
                svr, 
                param_grid, 
                cv=tscv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=0
            )
            
            # Fit grid search
            grid_search.fit(X, y)
            
            best_params = grid_search.best_params_
            best_score = -grid_search.best_score_  # Convert back to positive MAE
            
            logger.info(f"Best SVR parameters: {best_params}")
            logger.info(f"Best cross-validation MAE: {best_score:.2f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"SVR hyperparameter optimization failed: {e}")
            # Return default parameters
            return {
                'C': 100.0,
                'gamma': 'scale',
                'epsilon': 0.1,
                'kernel': 'rbf'
            }
    
    def train(self, df: pd.DataFrame, optimize_params: bool = True) -> bool:
        """
        Train the SVR model.
        
        Args:
            df: DataFrame with engineered features
            optimize_params: Whether to optimize hyperparameters
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            logger.info("Training SVR model...")
            
            # Prepare features
            X, y = self.prepare_features(df)
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No valid data for SVR training")
                return False
            
            if len(X) < 50:
                logger.error(f"Insufficient data for SVR training: {len(X)} samples")
                return False
            
            # Optimize hyperparameters if requested
            if optimize_params:
                self.best_params = self.optimize_hyperparameters(X, y)
                self.hyperparameters.update(self.best_params)
            
            # Create and train SVR model
            self.model = SVR(**self.hyperparameters)
            self.model.fit(X, y)
            
            # Calculate training score
            train_predictions = self.model.predict(X)
            self.training_score = {
                'mae': mean_absolute_error(y, train_predictions),
                'rmse': np.sqrt(mean_squared_error(y, train_predictions)),
                'mape': np.mean(np.abs((y - train_predictions) / y)) * 100
            }
            
            self.is_trained = True
            
            logger.info("SVR model trained successfully")
            logger.info(f"Training MAE: {self.training_score['mae']:.2f}")
            logger.info(f"Training RMSE: {self.training_score['rmse']:.2f}")
            logger.info(f"Training MAPE: {self.training_score['mape']:.2f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"SVR training failed: {e}")
            return False

        finally:
            # After training, compute dynamic feature importance: try SHAP first, else use permutation importance
            try:
                import shap
                logger.info('Computing SHAP values for SVR (this may be slow)')
                # Use a KernelExplainer for model-agnostic SHAP values (approximate)
                background = X[np.random.choice(len(X), min(100, len(X)), replace=False)]
                explainer = shap.KernelExplainer(self.model.predict, background)
                shap_vals = explainer.shap_values(X[:min(200, len(X))])
                # Average absolute SHAP values across samples
                mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
                mean_abs_shap = mean_abs_shap / (np.sum(mean_abs_shap) + 1e-12)
                self.feature_importance = {self.feature_columns[i]: float(mean_abs_shap[i]) for i in range(len(self.feature_columns))}
                logger.info('SHAP feature importance computed for SVR')
            except Exception:
                try:
                    logger.info('SHAP not available or failed; using permutation importance as fallback')
                    perm = permutation_importance(self.model, X, y, n_repeats=5, random_state=42, n_jobs=1)
                    scores = perm.importances_mean
                    scores = scores / (np.sum(scores) + 1e-12)
                    self.feature_importance = {self.feature_columns[i]: float(scores[i]) for i in range(len(self.feature_columns))}
                except Exception as e:
                    logger.warning(f'Failed to compute feature importance for SVR: {e}')
                    self.feature_importance = {}
    
    def predict(self, df: pd.DataFrame) -> Tuple[float, Tuple[float, float]]:
        """
        Make prediction using the trained SVR model.
        
        Args:
            df: DataFrame with features for prediction (single row or multiple rows)
            
        Returns:
            Tuple of (prediction, confidence_interval)
        """
        try:
            if not self.is_trained or self.model is None:
                logger.error("SVR model not trained")
                return 0.0, (0.0, 0.0)
            
            # Prepare features
            X, _ = self.prepare_features(df)
            
            if len(X) == 0:
                logger.error("No valid features for SVR prediction")
                return 0.0, (0.0, 0.0)
            
            # Make prediction (use last row if multiple rows provided)
            prediction = self.model.predict(X[-1:])
            pred_value = float(prediction[0])
            
            # Estimate confidence interval
            # SVR doesn't provide built-in confidence intervals, so we estimate based on training error
            if self.training_score:
                mae = self.training_score['mae']
                # Use MAE as a rough estimate for confidence interval
                lower_bound = pred_value - 1.96 * mae
                upper_bound = pred_value + 1.96 * mae
            else:
                # Fallback: use 5% of prediction value
                margin = abs(pred_value) * 0.05
                lower_bound = pred_value - margin
                upper_bound = pred_value + margin
            
            logger.debug(f"SVR prediction: {pred_value:.2f} [{lower_bound:.2f}, {upper_bound:.2f}]")
            return pred_value, (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"SVR prediction failed: {e}")
            return 0.0, (0.0, 0.0)
    
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make batch predictions using the trained SVR model.
        
        Args:
            df: DataFrame with features for prediction
            
        Returns:
            Array of predictions
        """
        try:
            if not self.is_trained or self.model is None:
                logger.error("SVR model not trained")
                return np.array([])
            
            # Prepare features
            X, _ = self.prepare_features(df)
            
            if len(X) == 0:
                logger.error("No valid features for SVR batch prediction")
                return np.array([])
            
            # Make predictions
            predictions = self.model.predict(X)
            
            logger.debug(f"SVR batch prediction: {len(predictions)} predictions made")
            return predictions
            
        except Exception as e:
            logger.error(f"SVR batch prediction failed: {e}")
            return np.array([])
    
    def update_model(self, new_data: pd.DataFrame) -> bool:
        """
        Update the model with new data (retrain with recent data).
        
        Args:
            new_data: DataFrame with new data
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if not self.is_trained:
                logger.warning("Model not trained, performing full training")
                return self.train(new_data, optimize_params=False)
            
            # For SVR, we retrain with new data (no incremental learning)
            # Use the same hyperparameters to avoid re-optimization
            logger.info("Updating SVR model with new data...")
            return self.train(new_data, optimize_params=False)
            
        except Exception as e:
            logger.error(f"SVR model update failed: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (approximation for SVR).
        
        Returns:
            Dictionary with feature importance scores
        """
        try:
            if not self.is_trained or self.model is None:
                logger.error("SVR model not trained")
                return {}
            
            # SVR doesn't have built-in feature importance
            # We can approximate it using the support vectors
            if hasattr(self.model, 'support_vectors_') and len(self.model.support_vectors_) > 0:
                # Calculate average absolute values of support vectors as importance proxy
                importance_scores = np.mean(np.abs(self.model.support_vectors_), axis=0)
                
                # Normalize to sum to 1
                importance_scores = importance_scores / np.sum(importance_scores)
                
                # Create dictionary with feature names
                feature_importance = {}
                for i, feature in enumerate(self.feature_columns):
                    if i < len(importance_scores):
                        feature_importance[feature] = float(importance_scores[i])
                
                return feature_importance
            else:
                logger.warning("No support vectors available for feature importance")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get SVR feature importance: {e}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': 'SVR',
            'is_trained': self.is_trained,
            'hyperparameters': self.hyperparameters.copy(),
            'n_features': len(self.feature_columns),
            'feature_columns': self.feature_columns.copy()
        }
        
        if self.is_trained and self.model is not None:
            info.update({
                'n_support_vectors': getattr(self.model, 'n_support_', None),
                'support_vectors_shape': getattr(self.model, 'support_vectors_', np.array([])).shape,
                'training_score': self.training_score
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
                logger.error("Cannot save untrained SVR model")
                return False
            
            model_data = {
                'model': self.model,
                'hyperparameters': self.hyperparameters,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained,
                'training_score': self.training_score,
                'best_params': self.best_params
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"SVR model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save SVR model: {e}")
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
                logger.error(f"SVR model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.hyperparameters = model_data['hyperparameters']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            self.training_score = model_data.get('training_score')
            self.best_params = model_data.get('best_params')
            
            logger.info(f"SVR model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load SVR model: {e}")
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
                logger.error("Cannot evaluate untrained SVR model")
                return {}
            
            # Prepare test features
            X_test, y_test = self.prepare_features(test_data)
            
            if len(X_test) == 0 or len(y_test) == 0:
                logger.error("No valid test data for SVR evaluation")
                return {}
            
            # Make predictions
            predictions = self.model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            
            # Directional accuracy
            if len(y_test) > 1:
                actual_direction = np.sign(np.diff(y_test))
                pred_direction = np.sign(np.diff(predictions))
                directional_accuracy = np.mean(actual_direction == pred_direction) * 100
            else:
                directional_accuracy = 0.0
            
            # R-squared
            ss_res = np.sum((y_test - predictions) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'directional_accuracy': directional_accuracy,
                'r2_score': r2_score,
                'mean_prediction': np.mean(predictions),
                'mean_actual': np.mean(y_test),
                'n_test_samples': len(y_test)
            }
            
            logger.info(f"SVR evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"SVR model evaluation failed: {e}")
            return {}