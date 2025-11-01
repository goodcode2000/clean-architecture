"""Random Forest model for BTC price prediction."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Any, List
from loguru import logger
import joblib
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class RandomForestPredictor:
    """Random Forest model for handling non-linearities and noisy data."""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_columns = []
        self.feature_importance = {}
        self.hyperparameters = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        }
        self.best_params = None
        self.training_score = None
        self.oob_score = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for Random Forest training.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple of (X, y) arrays
        """
        try:
            # Random Forest can handle many features, so use most available features
            exclude_columns = ['timestamp', 'close']  # Exclude target and timestamp
            
            # Get all numeric columns except the target
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_columns if col not in exclude_columns]
            
            if len(feature_cols) < 5:
                logger.error(f"Insufficient features for Random Forest: {len(feature_cols)}")
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
            
            logger.debug(f"Random Forest features prepared: {X.shape[1]} features, {X.shape[0]} samples")
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to prepare Random Forest features: {e}")
            return np.array([]), np.array([])
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize Random Forest hyperparameters using grid search.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Best hyperparameters
        """
        try:
            logger.info("Optimizing Random Forest hyperparameters...")
            
            # Define parameter grid (smaller for faster optimization)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Use TimeSeriesSplit for time series data
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Create Random Forest model
            rf = RandomForestRegressor(
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                oob_score=True
            )
            
            # Perform grid search
            grid_search = GridSearchCV(
                rf, 
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
            
            logger.info(f"Best Random Forest parameters: {best_params}")
            logger.info(f"Best cross-validation MAE: {best_score:.2f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Random Forest hyperparameter optimization failed: {e}")
            # Return default parameters
            return {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt'
            }
    
    def train(self, df: pd.DataFrame, optimize_params: bool = True) -> bool:
        """
        Train the Random Forest model.
        
        Args:
            df: DataFrame with engineered features
            optimize_params: Whether to optimize hyperparameters
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            logger.info("Training Random Forest model...")
            
            # Prepare features
            X, y = self.prepare_features(df)
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No valid data for Random Forest training")
                return False
            
            if len(X) < 50:
                logger.error(f"Insufficient data for Random Forest training: {len(X)} samples")
                return False
            
            # Optimize hyperparameters if requested
            if optimize_params:
                self.best_params = self.optimize_hyperparameters(X, y)
                self.hyperparameters.update(self.best_params)
            
            # Create and train Random Forest model
            self.model = RandomForestRegressor(**self.hyperparameters)
            self.model.fit(X, y)
            
            # Get feature importance
            self.feature_importance = dict(zip(
                self.feature_columns, 
                self.model.feature_importances_
            ))
            
            # Get OOB score if available
            if hasattr(self.model, 'oob_score_'):
                self.oob_score = self.model.oob_score_
            
            # Calculate training score
            train_predictions = self.model.predict(X)
            self.training_score = {
                'mae': mean_absolute_error(y, train_predictions),
                'rmse': np.sqrt(mean_squared_error(y, train_predictions)),
                'mape': np.mean(np.abs((y - train_predictions) / y)) * 100
            }
            
            self.is_trained = True
            
            logger.info("Random Forest model trained successfully")
            logger.info(f"Training MAE: {self.training_score['mae']:.2f}")
            logger.info(f"Training RMSE: {self.training_score['rmse']:.2f}")
            logger.info(f"Training MAPE: {self.training_score['mape']:.2f}%")
            if self.oob_score:
                logger.info(f"OOB Score: {self.oob_score:.4f}")
            
            # Log top feature importances
            top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"Top 5 features: {[(f, round(imp, 4)) for f, imp in top_features]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            return False
    
    def predict(self, df: pd.DataFrame) -> Tuple[float, Tuple[float, float]]:
        """
        Make prediction using the trained Random Forest model.
        
        Args:
            df: DataFrame with features for prediction (single row or multiple rows)
            
        Returns:
            Tuple of (prediction, confidence_interval)
        """
        try:
            if not self.is_trained or self.model is None:
                logger.error("Random Forest model not trained")
                return 0.0, (0.0, 0.0)
            
            # Prepare features
            X, _ = self.prepare_features(df)
            
            if len(X) == 0:
                logger.error("No valid features for Random Forest prediction")
                return 0.0, (0.0, 0.0)
            
            # Make prediction (use last row if multiple rows provided)
            prediction = self.model.predict(X[-1:])
            pred_value = float(prediction[0])
            
            # Estimate confidence interval using individual tree predictions
            try:
                # Get predictions from all trees
                tree_predictions = np.array([
                    tree.predict(X[-1:]) for tree in self.model.estimators_
                ])
                tree_predictions = tree_predictions.flatten()
                
                # Calculate confidence interval from tree prediction variance
                std_pred = np.std(tree_predictions)
                lower_bound = pred_value - 1.96 * std_pred
                upper_bound = pred_value + 1.96 * std_pred
                
            except Exception:
                # Fallback: use training error for confidence interval
                if self.training_score:
                    mae = self.training_score['mae']
                    lower_bound = pred_value - 1.96 * mae
                    upper_bound = pred_value + 1.96 * mae
                else:
                    # Last fallback: use 5% of prediction value
                    margin = abs(pred_value) * 0.05
                    lower_bound = pred_value - margin
                    upper_bound = pred_value + margin
            
            logger.debug(f"Random Forest prediction: {pred_value:.2f} [{lower_bound:.2f}, {upper_bound:.2f}]")
            return pred_value, (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"Random Forest prediction failed: {e}")
            return 0.0, (0.0, 0.0)
    
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make batch predictions using the trained Random Forest model.
        
        Args:
            df: DataFrame with features for prediction
            
        Returns:
            Array of predictions
        """
        try:
            if not self.is_trained or self.model is None:
                logger.error("Random Forest model not trained")
                return np.array([])
            
            # Prepare features
            X, _ = self.prepare_features(df)
            
            if len(X) == 0:
                logger.error("No valid features for Random Forest batch prediction")
                return np.array([])
            
            # Make predictions
            predictions = self.model.predict(X)
            
            logger.debug(f"Random Forest batch prediction: {len(predictions)} predictions made")
            return predictions
            
        except Exception as e:
            logger.error(f"Random Forest batch prediction failed: {e}")
            return np.array([])
    
    def get_prediction_intervals(self, df: pd.DataFrame, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get prediction intervals using quantile regression forests approach.
        
        Args:
            df: DataFrame with features for prediction
            confidence: Confidence level for intervals
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        try:
            if not self.is_trained or self.model is None:
                logger.error("Random Forest model not trained")
                return np.array([]), np.array([]), np.array([])
            
            # Prepare features
            X, _ = self.prepare_features(df)
            
            if len(X) == 0:
                return np.array([]), np.array([]), np.array([])
            
            # Get predictions from all trees
            all_tree_predictions = np.array([
                tree.predict(X) for tree in self.model.estimators_
            ])
            
            # Calculate quantiles
            alpha = 1 - confidence
            lower_quantile = alpha / 2
            upper_quantile = 1 - alpha / 2
            
            predictions = np.mean(all_tree_predictions, axis=0)
            lower_bounds = np.quantile(all_tree_predictions, lower_quantile, axis=0)
            upper_bounds = np.quantile(all_tree_predictions, upper_quantile, axis=0)
            
            return predictions, lower_bounds, upper_bounds
            
        except Exception as e:
            logger.error(f"Failed to get Random Forest prediction intervals: {e}")
            return np.array([]), np.array([]), np.array([])
    
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
            
            # For Random Forest, we retrain with new data
            # Use the same hyperparameters to avoid re-optimization
            logger.info("Updating Random Forest model with new data...")
            return self.train(new_data, optimize_params=False)
            
        except Exception as e:
            logger.error(f"Random Forest model update failed: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained Random Forest.
        
        Returns:
            Dictionary with feature importance scores
        """
        return self.feature_importance.copy()
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N most important features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of (feature_name, importance_score) tuples
        """
        if not self.feature_importance:
            return []
        
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_features[:n]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': 'RandomForest',
            'is_trained': self.is_trained,
            'hyperparameters': self.hyperparameters.copy(),
            'n_features': len(self.feature_columns),
            'feature_columns': self.feature_columns.copy(),
            'n_estimators': self.hyperparameters.get('n_estimators', 0)
        }
        
        if self.is_trained and self.model is not None:
            info.update({
                'n_features_in': getattr(self.model, 'n_features_in_', None),
                'n_outputs': getattr(self.model, 'n_outputs_', None),
                'oob_score': self.oob_score,
                'training_score': self.training_score,
                'feature_importance_available': len(self.feature_importance) > 0
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
                logger.error("Cannot save untrained Random Forest model")
                return False
            
            model_data = {
                'model': self.model,
                'hyperparameters': self.hyperparameters,
                'feature_columns': self.feature_columns,
                'feature_importance': self.feature_importance,
                'is_trained': self.is_trained,
                'training_score': self.training_score,
                'oob_score': self.oob_score,
                'best_params': self.best_params
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Random Forest model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save Random Forest model: {e}")
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
                logger.error(f"Random Forest model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.hyperparameters = model_data['hyperparameters']
            self.feature_columns = model_data['feature_columns']
            self.feature_importance = model_data['feature_importance']
            self.is_trained = model_data['is_trained']
            self.training_score = model_data.get('training_score')
            self.oob_score = model_data.get('oob_score')
            self.best_params = model_data.get('best_params')
            
            logger.info(f"Random Forest model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Random Forest model: {e}")
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
                logger.error("Cannot evaluate untrained Random Forest model")
                return {}
            
            # Prepare test features
            X_test, y_test = self.prepare_features(test_data)
            
            if len(X_test) == 0 or len(y_test) == 0:
                logger.error("No valid test data for Random Forest evaluation")
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
            
            logger.info(f"Random Forest evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"Random Forest model evaluation failed: {e}")
            return {}