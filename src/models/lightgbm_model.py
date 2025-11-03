"""
LightGBM Model with Residual Correction for BTC Price Prediction
Handles complex nonlinear interactions and incorporates rich features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .base_model import BaseModel
from ..feature_engineering import FeatureSet

logger = logging.getLogger(__name__)

class LightGBMPredictionModel(BaseModel):
    """LightGBM model with residual correction for nonlinear interactions"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            # LightGBM parameters
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
            'n_estimators': 1000,
            'early_stopping_rounds': 100,
            'max_depth': -1,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            
            # Training parameters
            'min_periods': 100,
            'validation_split': 0.2,
            'time_series_cv_splits': 5,
            'feature_selection': True,
            'residual_correction': True,
            'ensemble_models': 3,  # Number of models in ensemble
            
            # Feature engineering
            'lag_features': [1, 2, 3, 5, 10],
            'rolling_features': [5, 10, 20],
            'target_encoding': True,
            'interaction_features': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__("LightGBM", default_config)
        
        # LightGBM-specific attributes
        self.models = []  # Ensemble of models
        self.feature_names = []
        self.scaler = StandardScaler()
        self.residual_model = None
        self.feature_selector = None
        self.validation_scores = []
        
    async def train(self, features: List[FeatureSet], targets: List[float]) -> bool:
        """Train LightGBM model with feature engineering and residual correction"""
        try:
            if len(targets) < self.config['min_periods']:
                logger.warning(f"Insufficient data for LightGBM training: {len(targets)} < {self.config['min_periods']}")
                return False
            
            logger.info(f"Training LightGBM model with {len(targets)} data points")
            
            # Prepare features and targets
            X, y = self._prepare_training_data(features, targets)
            
            if X.shape[0] < self.config['min_periods']:
                logger.warning("Insufficient data after feature engineering")
                return False
            
            # Feature selection
            if self.config['feature_selection']:
                X = self._select_features(X, y)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.config['time_series_cv_splits'])
            cv_scores = []
            
            # Train ensemble of models
            self.models = []
            for i in range(self.config['ensemble_models']):
                model_scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Create LightGBM datasets
                    train_data = lgb.Dataset(X_train, label=y_train)
                    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                    
                    # Train model with different random seeds for ensemble diversity
                    model_config = self.config.copy()
                    model_config['random_state'] = self.config['random_state'] + i
                    
                    model = lgb.train(
                        model_config,
                        train_data,
                        valid_sets=[val_data],
                        callbacks=[lgb.early_stopping(self.config['early_stopping_rounds']),
                                 lgb.log_evaluation(0)]  # Suppress output
                    )
                    
                    # Validate
                    val_pred = model.predict(X_val)
                    val_score = mean_absolute_error(y_val, val_pred)
                    model_scores.append(val_score)
                
                # Train final model on all data
                train_data = lgb.Dataset(X, label=y)
                final_model = lgb.train(
                    model_config,
                    train_data,
                    callbacks=[lgb.log_evaluation(0)]
                )
                
                self.models.append(final_model)
                cv_scores.append(np.mean(model_scores))
                
                logger.info(f"Model {i+1}/{self.config['ensemble_models']} - CV MAE: {np.mean(model_scores):.4f}")
            
            # Store validation scores
            self.validation_scores = cv_scores
            
            # Train residual correction model
            if self.config['residual_correction']:
                await self._train_residual_correction(X, y)
            
            # Calculate feature importance
            self._calculate_feature_importance()
            
            # Calculate performance metrics
            ensemble_pred = self._ensemble_predict(X)
            mae = mean_absolute_error(y, ensemble_pred)
            rmse = np.sqrt(mean_squared_error(y, ensemble_pred))
            mape = np.mean(np.abs((y - ensemble_pred) / (y + 1e-10))) * 100
            
            # Update training history
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'n_features': X.shape[1],
                'data_points': len(y)
            }
            
            self.update_training_history(metrics)
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            logger.info(f"LightGBM ensemble trained successfully - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            return False
    
    async def predict(self, features: FeatureSet) -> Tuple[float, float]:
        """Make prediction using LightGBM ensemble with residual correction"""
        try:
            if not self.is_trained or not self.models:
                logger.warning("LightGBM model not trained")
                return 0.0, 0.0
            
            # Prepare features
            X = self._prepare_prediction_features(features)
            
            if X.size == 0:
                logger.warning("Failed to prepare features for prediction")
                return 0.0, 0.0
            
            # Ensemble prediction
            base_prediction = self._ensemble_predict(X.reshape(1, -1))[0]
            
            # Apply residual correction
            if self.config['residual_correction'] and self.residual_model:
                try:
                    residual = self.residual_model.predict(X.reshape(1, -1))[0]
                    corrected_prediction = base_prediction + residual
                except Exception as e:
                    logger.warning(f"Residual correction failed: {e}")
                    corrected_prediction = base_prediction
            else:
                corrected_prediction = base_prediction
            
            # Calculate prediction confidence
            confidence = self._calculate_prediction_confidence(X, base_prediction)
            
            # Ensure prediction is positive
            corrected_prediction = max(corrected_prediction, 0.01)
            
            logger.debug(f"LightGBM prediction: {corrected_prediction:.2f}, confidence: {confidence:.3f}")
            return corrected_prediction, confidence
            
        except Exception as e:
            logger.error(f"LightGBM prediction failed: {e}")
            return 0.0, 0.0
    
    def _prepare_training_data(self, features: List[FeatureSet], targets: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with feature engineering"""
        try:
            # Convert features to DataFrame
            feature_dicts = [fs.to_dict() for fs in features]
            df = pd.DataFrame(feature_dicts)
            
            # Add targets
            df['target'] = targets
            
            # Add lag features
            for lag in self.config['lag_features']:
                df[f'target_lag_{lag}'] = df['target'].shift(lag)
                
                # Add lag features for key indicators
                key_features = ['price', 'volume', 'rsi', 'volatility_5d']
                for feature in key_features:
                    if feature in df.columns:
                        df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
            
            # Add rolling features
            for window in self.config['rolling_features']:
                df[f'target_rolling_mean_{window}'] = df['target'].rolling(window).mean()
                df[f'target_rolling_std_{window}'] = df['target'].rolling(window).std()
                
                # Rolling features for key indicators
                key_features = ['price', 'volume']
                for feature in key_features:
                    if feature in df.columns:
                        df[f'{feature}_rolling_mean_{window}'] = df[feature].rolling(window).mean()
                        df[f'{feature}_rolling_std_{window}'] = df[feature].rolling(window).std()
            
            # Add interaction features
            if self.config['interaction_features']:
                df = self._add_interaction_features(df)
            
            # Remove rows with NaN values
            df = df.dropna()
            
            # Separate features and targets
            target_col = 'target'
            feature_cols = [col for col in df.columns if col != target_col]
            
            X = df[feature_cols].values
            y = df[target_col].values
            
            # Store feature names
            self.feature_names = feature_cols
            
            # Scale features
            X = self.scaler.fit_transform(X)
            
            logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Training data preparation failed: {e}")
            return np.array([]), np.array([])
    
    def _prepare_prediction_features(self, features: FeatureSet) -> np.ndarray:
        """Prepare features for prediction"""
        try:
            # Convert to dictionary
            feature_dict = features.to_dict()
            
            # Create DataFrame with single row
            df = pd.DataFrame([feature_dict])
            
            # Add missing features with default values
            for feature_name in self.feature_names:
                if feature_name not in df.columns:
                    if 'lag_' in feature_name or 'rolling_' in feature_name:
                        # For lag and rolling features, use current values as approximation
                        base_feature = feature_name.split('_')[0]
                        if base_feature in df.columns:
                            df[feature_name] = df[base_feature]
                        else:
                            df[feature_name] = 0.0
                    else:
                        df[feature_name] = 0.0
            
            # Reorder columns to match training
            df = df[self.feature_names]
            
            # Scale features
            X = self.scaler.transform(df.values)
            
            return X.flatten()
            
        except Exception as e:
            logger.error(f"Prediction feature preparation failed: {e}")
            return np.array([])
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features"""
        try:
            # Key feature pairs for interactions
            interaction_pairs = [
                ('price', 'volume'),
                ('rsi', 'volatility_5d'),
                ('bb_position', 'rsi'),
                ('momentum_5', 'volatility_5d'),
                ('price', 'sma_20')
            ]
            
            for feat1, feat2 in interaction_pairs:
                if feat1 in df.columns and feat2 in df.columns:
                    # Multiplicative interaction
                    df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                    
                    # Ratio interaction
                    df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-10)
            
            return df
            
        except Exception as e:
            logger.warning(f"Interaction feature creation failed: {e}")
            return df
    
    def _select_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Select important features using LightGBM feature importance"""
        try:
            from sklearn.feature_selection import SelectFromModel
            
            # Train a simple model for feature selection
            selector_model = lgb.LGBMRegressor(
                n_estimators=100,
                random_state=self.config['random_state'],
                verbose=-1
            )
            
            selector_model.fit(X, y)
            
            # Select features based on importance
            selector = SelectFromModel(selector_model, prefit=True)
            X_selected = selector.transform(X)
            
            # Update feature names
            selected_indices = selector.get_support(indices=True)
            self.feature_names = [self.feature_names[i] for i in selected_indices]
            
            logger.info(f"Feature selection: {X.shape[1]} -> {X_selected.shape[1]} features")
            return X_selected
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            return X
    
    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble prediction"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    async def _train_residual_correction(self, X: np.ndarray, y: np.ndarray):
        """Train residual correction model"""
        try:
            # Get base predictions
            base_predictions = self._ensemble_predict(X)
            
            # Calculate residuals
            residuals = y - base_predictions
            
            # Train residual model
            self.residual_model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.config['random_state'],
                verbose=-1
            )
            
            self.residual_model.fit(X, residuals)
            
            logger.info("Residual correction model trained")
            
        except Exception as e:
            logger.warning(f"Residual correction training failed: {e}")
            self.residual_model = None
    
    def _calculate_feature_importance(self):
        """Calculate ensemble feature importance"""
        try:
            if not self.models or not self.feature_names:
                return
            
            # Average feature importance across models
            importance_sum = np.zeros(len(self.feature_names))
            
            for model in self.models:
                importance_sum += model.feature_importance(importance_type='gain')
            
            importance_avg = importance_sum / len(self.models)
            
            # Create feature importance dictionary
            self.feature_importance = {}
            for i, feature_name in enumerate(self.feature_names):
                self.feature_importance[feature_name] = float(importance_avg[i])
            
            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
    
    def _calculate_prediction_confidence(self, X: np.ndarray, prediction: float) -> float:
        """Calculate prediction confidence based on model agreement"""
        try:
            # Get predictions from all models
            predictions = []
            for model in self.models:
                pred = model.predict(X.reshape(1, -1))[0]
                predictions.append(pred)
            
            # Calculate prediction variance
            pred_std = np.std(predictions)
            pred_mean = np.mean(predictions)
            
            # Calculate coefficient of variation
            cv = pred_std / (abs(pred_mean) + 1e-10)
            
            # Convert to confidence (lower variance = higher confidence)
            confidence = max(0.1, min(0.9, 1.0 - min(cv, 1.0)))
            
            # Adjust based on validation performance
            if self.validation_scores:
                avg_cv_score = np.mean(self.validation_scores)
                # Lower validation error = higher confidence
                confidence *= max(0.5, 1.0 - min(avg_cv_score / 1000.0, 0.5))
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get LightGBM model information"""
        info = {
            'model_name': self.model_name,
            'model_type': 'LightGBM Ensemble with Residual Correction',
            'is_trained': self.is_trained,
            'config': self.config.copy(),
            'feature_importance': dict(list(self.feature_importance.items())[:20]),  # Top 20 features
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'ensemble_size': len(self.models),
            'n_features': len(self.feature_names)
        }
        
        if self.validation_scores:
            info['validation_scores'] = {
                'mean': np.mean(self.validation_scores),
                'std': np.std(self.validation_scores),
                'scores': self.validation_scores
            }
        
        if self.training_history:
            latest_metrics = self.training_history[-1]['metrics']
            info['latest_metrics'] = latest_metrics
        
        return info
    
    def get_top_features(self, n: int = 10) -> Dict[str, float]:
        """Get top N most important features"""
        if not self.feature_importance:
            return {}
        
        return dict(list(self.feature_importance.items())[:n])
    
    def analyze_predictions(self, X: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction components and uncertainty"""
        try:
            if not self.models:
                return {}
            
            # Get predictions from all models
            model_predictions = []
            for i, model in enumerate(self.models):
                pred = model.predict(X)
                model_predictions.append(pred)
            
            model_predictions = np.array(model_predictions)
            
            # Calculate statistics
            mean_pred = np.mean(model_predictions, axis=0)
            std_pred = np.std(model_predictions, axis=0)
            min_pred = np.min(model_predictions, axis=0)
            max_pred = np.max(model_predictions, axis=0)
            
            analysis = {
                'ensemble_mean': mean_pred.tolist(),
                'ensemble_std': std_pred.tolist(),
                'prediction_range': {
                    'min': min_pred.tolist(),
                    'max': max_pred.tolist()
                },
                'model_agreement': (1.0 - std_pred / (np.abs(mean_pred) + 1e-10)).tolist(),
                'individual_predictions': model_predictions.tolist()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Prediction analysis failed: {e}")
            return {}