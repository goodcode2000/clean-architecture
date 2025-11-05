"""XGBoost model for BTC price prediction."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Any, List
from loguru import logger
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
except ImportError:
    logger.error("XGBoost not installed. Install with: pip install xgboost")
    raise

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class XGBoostPredictor:
    """XGBoost model for powerful gradient boosting optimized for time series prediction."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        self.feature_importance = {}
        self.hyperparameters = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'booster': 'gbtree',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 10,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
        }
        self.best_params = None
        self.training_score = None
        self.validation_score = None
        self.best_iteration = None
        self.quantile_models = {}  # For confidence intervals
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for XGBoost training.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple of (X, y) arrays
        """
        try:
            # XGBoost can handle many features efficiently
            exclude_columns = ['timestamp', 'close']  # Exclude target and timestamp
            
            # Get all numeric columns except the target
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_columns if col not in exclude_columns]
            
            if len(feature_cols) < 5:
                logger.error(f"Insufficient features for XGBoost: {len(feature_cols)}")
                return np.array([]), np.array([])
            
            X = df[feature_cols].values
            y = df['close'].values
            
            # Remove any rows with NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            # Handle infinite values
            X = np.where(np.isinf(X), np.nan, X)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features for better performance
            if not self.is_trained:
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
            
            self.feature_columns = feature_cols
            
            logger.debug(f"XGBoost features prepared: {X.shape[1]} features, {X.shape[0]} samples")
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to prepare XGBoost features: {e}")
            return np.array([]), np.array([])
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters using grid search with time series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Best hyperparameters
        """
        try:
            logger.info("Optimizing XGBoost hyperparameters...")
            
            # Define parameter grid optimized for time series
            param_grid = {
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [50, 100, 200],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 1.0],
                'reg_lambda': [1, 1.5, 2.0]
            }
            
            # Use TimeSeriesSplit for time series data
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Create XGBoost model
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric='mae',
                random_state=42,
                n_jobs=-1
            )
            
            # Perform grid search
            grid_search = GridSearchCV(
                xgb_model, 
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
            
            logger.info(f"Best XGBoost parameters: {best_params}")
            logger.info(f"Best cross-validation MAE: {best_score:.2f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"XGBoost hyperparameter optimization failed: {e}")
            # Return default parameters
            return {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            }
    
    def train_quantile_models(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Train quantile regression models for confidence intervals.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Training XGBoost quantile models for confidence intervals...")
            
            # Train models for different quantiles
            quantiles = [0.025, 0.975]  # For 95% confidence interval
            
            for quantile in quantiles:
                # Create quantile-specific parameters
                quantile_params = self.hyperparameters.copy()
                quantile_params['objective'] = f'reg:quantileerror'
                quantile_params['quantile_alpha'] = quantile
                
                # Train quantile model
                quantile_model = xgb.XGBRegressor(**quantile_params)
                quantile_model.fit(X, y)
                
                self.quantile_models[quantile] = quantile_model
            
            logger.info(f"Trained {len(quantiles)} quantile models")
            return True
            
        except Exception as e:
            logger.warning(f"Quantile model training failed: {e}")
            return False
    
    def train(self, df: pd.DataFrame, optimize_params: bool = True, validation_split: float = 0.2) -> bool:
        """
        Train the XGBoost model.
        
        Args:
            df: DataFrame with engineered features
            optimize_params: Whether to optimize hyperparameters
            validation_split: Fraction of data to use for validation
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            logger.info("Training XGBoost model...")
            
            # Prepare features
            X, y = self.prepare_features(df)
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No valid data for XGBoost training")
                return False
            
            if len(X) < 50:
                logger.error(f"Insufficient data for XGBoost training: {len(X)} samples")
                return False
            
            # Split data for training and validation (time-based split)
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Optimize hyperparameters if requested
            if optimize_params:
                self.best_params = self.optimize_hyperparameters(X_train, y_train)
                self.hyperparameters.update(self.best_params)
            
            # Train main model with early stopping
            self.model = xgb.XGBRegressor(**self.hyperparameters)
            
            # Fit with validation set for early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
            
            # Get best iteration
            self.best_iteration = self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.hyperparameters['n_estimators']
            
            # Get feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(
                    self.feature_columns, 
                    self.model.feature_importances_
                ))
            
            # Calculate training and validation scores
            train_predictions = self.model.predict(X_train)
            val_predictions = self.model.predict(X_val)
            
            self.training_score = {
                'mae': mean_absolute_error(y_train, train_predictions),
                'rmse': np.sqrt(mean_squared_error(y_train, train_predictions)),
                'mape': np.mean(np.abs((y_train - train_predictions) / y_train)) * 100
            }
            
            self.validation_score = {
                'mae': mean_absolute_error(y_val, val_predictions),
                'rmse': np.sqrt(mean_squared_error(y_val, val_predictions)),
                'mape': np.mean(np.abs((y_val - val_predictions) / y_val)) * 100
            }
            
            # Train quantile models for confidence intervals
            self.train_quantile_models(X, y)
            
            self.is_trained = True
            
            logger.info("XGBoost model trained successfully")
            logger.info(f"Best iteration: {self.best_iteration}")
            logger.info(f"Training MAE: {self.training_score['mae']:.2f}")
            logger.info(f"Validation MAE: {self.validation_score['mae']:.2f}")
            logger.info(f"Training MAPE: {self.training_score['mape']:.2f}%")
            logger.info(f"Validation MAPE: {self.validation_score['mape']:.2f}%")
            
            # Log top feature importances
            if self.feature_importance:
                top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                logger.info(f"Top 5 features: {[(f, round(imp, 4)) for f, imp in top_features]}")
            
            return True
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return False
    
    def predict(self, df: pd.DataFrame) -> Tuple[float, Tuple[float, float]]:
        """
        Make prediction using the trained XGBoost model.
        
        Args:
            df: DataFrame with features for prediction
            
        Returns:
            Tuple of (prediction, confidence_interval)
        """
        try:
            if not self.is_trained or self.model is None:
                logger.error("XGBoost model not trained")
                return 0.0, (0.0, 0.0)
            
            # Prepare features
            X, _ = self.prepare_features(df)
            
            if len(X) == 0:
                logger.error("No valid features for XGBoost prediction")
                return 0.0, (0.0, 0.0)
            
            # Make prediction (use last row if multiple rows provided)
            prediction = self.model.predict(X[-1:])
            pred_value = float(prediction[0])
            
            # Calculate confidence interval using quantile models
            lower_bound, upper_bound = self.calculate_confidence_interval(X[-1:], pred_value)
            
            logger.debug(f"XGBoost prediction: {pred_value:.2f} [{lower_bound:.2f}, {upper_bound:.2f}]")
            return pred_value, (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return 0.0, (0.0, 0.0)
    
    def calculate_confidence_interval(self, X: np.ndarray, prediction: float) -> Tuple[float, float]:
        """
        Calculate confidence interval for prediction.
        
        Args:
            X: Feature matrix for prediction
            prediction: Point prediction
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        try:
            # Use quantile models if available
            if len(self.quantile_models) >= 2:
                quantiles = sorted(self.quantile_models.keys())
                lower_quantile = quantiles[0]  # 0.025
                upper_quantile = quantiles[-1]  # 0.975
                
                lower_bound = float(self.quantile_models[lower_quantile].predict(X)[0])
                upper_bound = float(self.quantile_models[upper_quantile].predict(X)[0])
                
                # Ensure bounds are reasonable
                if lower_bound >= upper_bound:
                    raise ValueError("Invalid quantile bounds")
                
                return lower_bound, upper_bound
            
            # Fallback: use validation error
            elif self.validation_score:
                mae = self.validation_score['mae']
                lower_bound = prediction - 1.96 * mae
                upper_bound = prediction + 1.96 * mae
                return lower_bound, upper_bound
            
            # Final fallback: percentage-based interval
            else:
                margin = abs(prediction) * 0.05
                lower_bound = prediction - margin
                upper_bound = prediction + margin
                return lower_bound, upper_bound
                
        except Exception as e:
            logger.debug(f"Confidence interval calculation failed: {e}")
            # Fallback confidence interval
            margin = abs(prediction) * 0.05
            return prediction - margin, prediction + margin
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        return self.feature_importance.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        info = {
            'model_type': 'XGBoost',
            'is_trained': self.is_trained,
            'hyperparameters': self.hyperparameters.copy(),
            'n_features': len(self.feature_columns),
            'feature_columns': self.feature_columns.copy(),
            'has_quantile_models': len(self.quantile_models) > 0
        }
        
        if self.is_trained and self.model is not None:
            info.update({
                'best_iteration': self.best_iteration,
                'training_score': self.training_score,
                'validation_score': self.validation_score,
                'n_quantile_models': len(self.quantile_models)
            })
        
        return info
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model to file."""
        try:
            if not self.is_trained:
                logger.error("Cannot save untrained XGBoost model")
                return False
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'hyperparameters': self.hyperparameters,
                'feature_columns': self.feature_columns,
                'feature_importance': self.feature_importance,
                'is_trained': self.is_trained,
                'training_score': self.training_score,
                'validation_score': self.validation_score,
                'best_iteration': self.best_iteration,
                'quantile_models': self.quantile_models
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"XGBoost model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save XGBoost model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model from file."""
        try:
            if not os.path.exists(filepath):
                logger.error(f"XGBoost model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.hyperparameters = model_data['hyperparameters']
            self.feature_columns = model_data['feature_columns']
            self.feature_importance = model_data['feature_importance']
            self.is_trained = model_data['is_trained']
            self.training_score = model_data.get('training_score')
            self.validation_score = model_data.get('validation_score')
            self.best_iteration = model_data.get('best_iteration')
            self.quantile_models = model_data.get('quantile_models', {})
            
            logger.info(f"XGBoost model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            return False
    
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make batch predictions using the trained XGBoost model.
        
        Args:
            df: DataFrame with features for prediction
            
        Returns:
            Array of predictions
        """
        try:
            if not self.is_trained or self.model is None:
                logger.error("XGBoost model not trained")
                return np.array([])
            
            # Prepare features
            X, _ = self.prepare_features(df)
            
            if len(X) == 0:
                logger.error("No valid features for XGBoost batch prediction")
                return np.array([])
            
            # Make predictions
            predictions = self.model.predict(X)
            
            logger.debug(f"XGBoost batch prediction: {len(predictions)} predictions made")
            return predictions
            
        except Exception as e:
            logger.error(f"XGBoost batch prediction failed: {e}")
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
            
            # For XGBoost, we retrain with new data
            # Use the same hyperparameters to avoid re-optimization
            logger.info("Updating XGBoost model with new data...")
            return self.train(new_data, optimize_params=False)
            
        except Exception as e:
            logger.error(f"XGBoost model update failed: {e}")
            return False
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N most important features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of (feature_name, importance_score) tuples
        """
        feature_importance = self.get_feature_importance()
        
        if not feature_importance:
            return []
        
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_features[:n]
    
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
                logger.error("Cannot evaluate untrained XGBoost model")
                return {}
            
            # Prepare test features
            X_test, y_test = self.prepare_features(test_data)
            
            if len(X_test) == 0 or len(y_test) == 0:
                logger.error("No valid test data for XGBoost evaluation")
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
            
            logger.info(f"XGBoost evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"XGBoost model evaluation failed: {e}")
            return {}
    
    def detect_overfitting(self) -> Dict[str, Any]:
        """
        Detect potential overfitting by comparing training and validation scores.
        
        Returns:
            Dictionary with overfitting analysis
        """
        try:
            if not self.training_score or not self.validation_score:
                return {'overfitting_detected': False, 'reason': 'No scores available'}
            
            train_mae = self.training_score['mae']
            val_mae = self.validation_score['mae']
            
            # Calculate overfitting ratio
            overfitting_ratio = val_mae / train_mae if train_mae > 0 else 1.0
            
            # Detect overfitting (validation error significantly higher than training error)
            overfitting_threshold = 1.5  # 50% higher validation error
            overfitting_detected = overfitting_ratio > overfitting_threshold
            
            analysis = {
                'overfitting_detected': overfitting_detected,
                'overfitting_ratio': overfitting_ratio,
                'training_mae': train_mae,
                'validation_mae': val_mae,
                'threshold': overfitting_threshold
            }
            
            if overfitting_detected:
                analysis['recommendation'] = 'Increase regularization or reduce model complexity'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Overfitting detection failed: {e}")
            return {'overfitting_detected': False, 'reason': f'Error: {e}'}