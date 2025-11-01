"""LightGBM model for BTC price prediction."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Any, List
from loguru import logger
import joblib
import os
import sys
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class LightGBMPredictor:
    """LightGBM model for powerful gradient boosting and complex pattern recognition."""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_columns = []
        self.feature_importance = {}
        self.hyperparameters = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 100,
            'early_stopping_rounds': 10
        }
        self.best_params = None
        self.training_score = None
        self.validation_score = None
        self.best_iteration = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for LightGBM training.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple of (X, y) arrays
        """
        try:
            # LightGBM can handle many features efficiently
            exclude_columns = ['timestamp', 'close']  # Exclude target and timestamp
            
            # Get all numeric columns except the target
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_columns if col not in exclude_columns]
            
            if len(feature_cols) < 5:
                logger.error(f"Insufficient features for LightGBM: {len(feature_cols)}")
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
            
            self.feature_columns = feature_cols
            
            logger.debug(f"LightGBM features prepared: {X.shape[1]} features, {X.shape[0]} samples")
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to prepare LightGBM features: {e}")
            return np.array([]), np.array([])
    
    def train(self, df: pd.DataFrame, optimize_params: bool = True, validation_split: float = 0.2) -> bool:
        """
        Train the LightGBM model.
        
        Args:
            df: DataFrame with engineered features
            optimize_params: Whether to optimize hyperparameters
            validation_split: Fraction of data to use for validation
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            logger.info("Training LightGBM model...")
            
            # Prepare features
            X, y = self.prepare_features(df)
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No valid data for LightGBM training")
                return False
            
            if len(X) < 50:
                logger.error(f"Insufficient data for LightGBM training: {len(X)} samples")
                return False
            
            # Split data for training and validation (time-based split)
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Optimize hyperparameters if requested
            if optimize_params:
                self.best_params = self.optimize_hyperparameters(X_train, y_train)
                self.hyperparameters.update(self.best_params)
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model with early stopping
            callbacks = [
                lgb.early_stopping(self.hyperparameters.get('early_stopping_rounds', 10)),
                lgb.log_evaluation(0)  # Suppress training logs
            ]
            
            self.model = lgb.train(
                self.hyperparameters,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'eval'],
                callbacks=callbacks
            )
            
            # Get best iteration
            self.best_iteration = self.model.best_iteration
            
            # Get feature importance
            self.feature_importance = dict(zip(
                self.feature_columns, 
                self.model.feature_importance(importance_type='gain')
            ))
            
            # Calculate training and validation scores
            train_predictions = self.model.predict(X_train, num_iteration=self.best_iteration)
            val_predictions = self.model.predict(X_val, num_iteration=self.best_iteration)
            
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
            
            self.is_trained = True
            
            logger.info("LightGBM model trained successfully")
            logger.info(f"Best iteration: {self.best_iteration}")
            logger.info(f"Training MAE: {self.training_score['mae']:.2f}")
            logger.info(f"Validation MAE: {self.validation_score['mae']:.2f}")
            logger.info(f"Training MAPE: {self.training_score['mape']:.2f}%")
            logger.info(f"Validation MAPE: {self.validation_score['mape']:.2f}%")
            
            # Log top feature importances
            top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"Top 5 features: {[(f, round(imp, 2)) for f, imp in top_features]}")
            
            return True
            
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            return False
    
    def predict(self, df: pd.DataFrame) -> Tuple[float, Tuple[float, float]]:
        """
        Make prediction using the trained LightGBM model.
        
        Args:
            df: DataFrame with features for prediction
            
        Returns:
            Tuple of (prediction, confidence_interval)
        """
        try:
            if not self.is_trained or self.model is None:
                logger.error("LightGBM model not trained")
                return 0.0, (0.0, 0.0)
            
            # Prepare features
            X, _ = self.prepare_features(df)
            
            if len(X) == 0:
                logger.error("No valid features for LightGBM prediction")
                return 0.0, (0.0, 0.0)
            
            # Make prediction (use last row if multiple rows provided)
            prediction = self.model.predict(X[-1:], num_iteration=self.best_iteration)
            pred_value = float(prediction[0])
            
            # Estimate confidence interval using validation error
            if self.validation_score:
                mae = self.validation_score['mae']
                lower_bound = pred_value - 1.96 * mae
                upper_bound = pred_value + 1.96 * mae
            else:
                margin = abs(pred_value) * 0.05
                lower_bound = pred_value - margin
                upper_bound = pred_value + margin
            
            logger.debug(f"LightGBM prediction: {pred_value:.2f} [{lower_bound:.2f}, {upper_bound:.2f}]")
            return pred_value, (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"LightGBM prediction failed: {e}")
            return 0.0, (0.0, 0.0)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        return self.feature_importance.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        info = {
            'model_type': 'LightGBM',
            'is_trained': self.is_trained,
            'hyperparameters': self.hyperparameters.copy(),
            'n_features': len(self.feature_columns),
            'feature_columns': self.feature_columns.copy()
        }
        
        if self.is_trained and self.model is not None:
            info.update({
                'best_iteration': self.best_iteration,
                'training_score': self.training_score,
                'validation_score': self.validation_score
            })
        
        return info
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model to file."""
        try:
            if not self.is_trained:
                logger.error("Cannot save untrained LightGBM model")
                return False
            
            model_data = {
                'model': self.model,
                'hyperparameters': self.hyperparameters,
                'feature_columns': self.feature_columns,
                'feature_importance': self.feature_importance,
                'is_trained': self.is_trained,
                'training_score': self.training_score,
                'validation_score': self.validation_score,
                'best_iteration': self.best_iteration
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"LightGBM model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save LightGBM model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model from file."""
        try:
            if not os.path.exists(filepath):
                logger.error(f"LightGBM model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.hyperparameters = model_data['hyperparameters']
            self.feature_columns = model_data['feature_columns']
            self.feature_importance = model_data['feature_importance']
            self.is_trained = model_data['is_trained']
            self.training_score = model_data.get('training_score')
            self.validation_score = model_data.get('validation_score')
            self.best_iteration = model_data.get('best_iteration')
            
            logger.info(f"LightGBM model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LightGBM model: {e}")
            return False
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters using grid search.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Best hyperparameters
        """
        try:
            logger.info("Optimizing LightGBM hyperparameters...")
            
            # Define parameter grid
            param_grid = {
                'num_leaves': [15, 31, 63],
                'learning_rate': [0.01, 0.05, 0.1],
                'feature_fraction': [0.8, 0.9, 1.0],
                'bagging_fraction': [0.8, 0.9, 1.0],
                'n_estimators': [50, 100, 200],
                'min_child_samples': [10, 20, 30]
            }
            
            # Use TimeSeriesSplit for time series data
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Create LightGBM model
            lgbm = lgb.LGBMRegressor(
                objective='regression',
                metric='mae',
                boosting_type='gbdt',
                verbose=-1,
                random_state=42
            )
            
            # Perform grid search
            grid_search = GridSearchCV(
                lgbm, 
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
            
            logger.info(f"Best LightGBM parameters: {best_params}")
            logger.info(f"Best cross-validation MAE: {best_score:.2f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"LightGBM hyperparameter optimization failed: {e}")
            # Return default parameters
            return {
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'n_estimators': 100
            }
    
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make batch predictions using the trained LightGBM model.
        
        Args:
            df: DataFrame with features for prediction
            
        Returns:
            Array of predictions
        """
        try:
            if not self.is_trained or self.model is None:
                logger.error("LightGBM model not trained")
                return np.array([])
            
            # Prepare features
            X, _ = self   
 def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters using grid search.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Best hyperparameters
        """
        try:
            logger.info("Optimizing LightGBM hyperparameters...")
            
            # Define parameter grid
            param_grid = {
                'num_leaves': [15, 31, 63],
                'learning_rate': [0.01, 0.05, 0.1],
                'feature_fraction': [0.8, 0.9, 1.0],
                'bagging_fraction': [0.8, 0.9, 1.0],
                'n_estimators': [50, 100, 200],
                'min_child_samples': [10, 20, 30]
            }
            
            # Use TimeSeriesSplit for time series data
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Create LightGBM model
            lgbm = lgb.LGBMRegressor(
                objective='regression',
                metric='mae',
                boosting_type='gbdt',
                verbose=-1,
                random_state=42
            )
            
            # Perform grid search
            grid_search = GridSearchCV(
                lgbm, 
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
            
            logger.info(f"Best LightGBM parameters: {best_params}")
            logger.info(f"Best cross-validation MAE: {best_score:.2f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"LightGBM hyperparameter optimization failed: {e}")
            # Return default parameters
            return {
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'n_estimators': 100
            }
    
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make batch predictions using the trained LightGBM model.
        
        Args:
            df: DataFrame with features for prediction
            
        Returns:
            Array of predictions
        """
        try:
            if not self.is_trained or self.model is None:
                logger.error("LightGBM model not trained")
                return np.array([])
            
            # Prepare features
            X, _ = self.prepare_features(df)
            
            if len(X) == 0:
                logger.error("No valid features for LightGBM batch prediction")
                return np.array([])
            
            # Make predictions
            predictions = self.model.predict(X, num_iteration=self.best_iteration)
            
            logger.debug(f"LightGBM batch prediction: {len(predictions)} predictions made")
            return predictions
            
        except Exception as e:
            logger.error(f"LightGBM batch prediction failed: {e}")
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
            
            # For LightGBM, we retrain with new data
            # Use the same hyperparameters to avoid re-optimization
            logger.info("Updating LightGBM model with new data...")
            return self.train(new_data, optimize_params=False)
            
        except Exception as e:
            logger.error(f"LightGBM model update failed: {e}")
            return False
    
    def get_top_features(self, n: int = 10, importance_type: str = 'gain') -> List[Tuple[str, float]]:
        """
        Get top N most important features.
        
        Args:
            n: Number of top features to return
            importance_type: Type of importance ('gain', 'split')
            
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
                logger.error("Cannot evaluate untrained LightGBM model")
                return {}
            
            # Prepare test features
            X_test, y_test = self.prepare_features(test_data)
            
            if len(X_test) == 0 or len(y_test) == 0:
                logger.error("No valid test data for LightGBM evaluation")
                return {}
            
            # Make predictions
            predictions = self.model.predict(X_test, num_iteration=self.best_iteration)
            
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
            
            logger.info(f"LightGBM evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"LightGBM model evaluation failed: {e}")
            return {}