"""Data preprocessing utilities for BTC prediction models."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class DataPreprocessor:
    """Handles data preprocessing for ML models."""
    
    def __init__(self):
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'close'
        self.models_dir = Config.MODELS_DIR
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
    
    def split_features_target(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features and target.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            
        Returns:
            Tuple of (features_df, target_series)
        """
        try:
            if target_column is None:
                target_column = self.target_column
            
            # Exclude non-feature columns
            exclude_columns = ['timestamp', target_column]
            feature_columns = [col for col in df.columns if col not in exclude_columns]
            
            # Handle boolean columns (convert to int)
            features_df = df[feature_columns].copy()
            for col in features_df.columns:
                if features_df[col].dtype == bool:
                    features_df[col] = features_df[col].astype(int)
            
            target_series = df[target_column].copy()
            
            self.feature_columns = feature_columns
            
            logger.debug(f"Split data: {len(feature_columns)} features, {len(target_series)} samples")
            return features_df, target_series
            
        except Exception as e:
            logger.error(f"Failed to split features and target: {e}")
            return pd.DataFrame(), pd.Series()
    
    def create_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2, 
                               time_based: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create train/test split for time series data.
        
        Args:
            df: DataFrame with features and target
            test_size: Proportion of data for testing
            time_based: If True, use time-based split (last X% for testing)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            # Split features and target
            X, y = self.split_features_target(df)
            
            if time_based:
                # Time-based split (use last test_size% for testing)
                split_idx = int(len(df) * (1 - test_size))
                
                X_train = X.iloc[:split_idx].copy()
                X_test = X.iloc[split_idx:].copy()
                y_train = y.iloc[:split_idx].copy()
                y_test = y.iloc[split_idx:].copy()
            else:
                # Random split (not recommended for time series)
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            
            logger.info(f"Train/test split: {len(X_train)} train, {len(X_test)} test samples")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Failed to create train/test split: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    
    def fit_scalers(self, X_train: pd.DataFrame, scaling_method: str = 'standard') -> bool:
        """
        Fit scalers on training data.
        
        Args:
            X_train: Training features
            scaling_method: Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.scalers = {}
            
            # Select scaler
            if scaling_method == 'standard':
                scaler_class = StandardScaler
            elif scaling_method == 'minmax':
                scaler_class = MinMaxScaler
            elif scaling_method == 'robust':
                scaler_class = RobustScaler
            else:
                logger.error(f"Unknown scaling method: {scaling_method}")
                return False
            
            # Fit scaler for each numeric column
            for col in X_train.columns:
                if X_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    scaler = scaler_class()
                    scaler.fit(X_train[[col]])
                    self.scalers[col] = scaler
            
            logger.info(f"Fitted {len(self.scalers)} scalers using {scaling_method} method")
            return True
            
        except Exception as e:
            logger.error(f"Failed to fit scalers: {e}")
            return False
    
    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scalers.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        try:
            if not self.scalers:
                logger.warning("No scalers fitted. Returning original data.")
                return X
            
            X_scaled = X.copy()
            
            for col, scaler in self.scalers.items():
                if col in X_scaled.columns:
                    X_scaled[col] = scaler.transform(X_scaled[[col]]).flatten()
            
            logger.debug(f"Transformed {len(self.scalers)} features")
            return X_scaled
            
        except Exception as e:
            logger.error(f"Failed to transform features: {e}")
            return X
    
    def inverse_transform_target(self, y_scaled: np.ndarray, target_column: str = None) -> np.ndarray:
        """
        Inverse transform target values.
        
        Args:
            y_scaled: Scaled target values
            target_column: Name of target column
            
        Returns:
            Original scale target values
        """
        try:
            if target_column is None:
                target_column = self.target_column
            
            if target_column in self.scalers:
                scaler = self.scalers[target_column]
                y_original = scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
                return y_original
            else:
                logger.warning(f"No scaler found for target column: {target_column}")
                return y_scaled
                
        except Exception as e:
            logger.error(f"Failed to inverse transform target: {e}")
            return y_scaled
    
    def save_scalers(self, filename: str = 'scalers.joblib') -> bool:
        """
        Save fitted scalers to file.
        
        Args:
            filename: Name of file to save scalers
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = os.path.join(self.models_dir, filename)
            
            scaler_data = {
                'scalers': self.scalers,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column
            }
            
            joblib.dump(scaler_data, filepath)
            logger.info(f"Scalers saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save scalers: {e}")
            return False
    
    def load_scalers(self, filename: str = 'scalers.joblib') -> bool:
        """
        Load scalers from file.
        
        Args:
            filename: Name of file to load scalers from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = os.path.join(self.models_dir, filename)
            
            if not os.path.exists(filepath):
                logger.warning(f"Scaler file not found: {filepath}")
                return False
            
            scaler_data = joblib.load(filepath)
            
            self.scalers = scaler_data.get('scalers', {})
            self.feature_columns = scaler_data.get('feature_columns', [])
            self.target_column = scaler_data.get('target_column', 'close')
            
            logger.info(f"Scalers loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load scalers: {e}")
            return False
    
    def prepare_data_for_model(self, df: pd.DataFrame, model_type: str = 'sklearn') -> Dict[str, Any]:
        """
        Prepare data for specific model type.
        
        Args:
            df: DataFrame with features
            model_type: Type of model ('sklearn', 'lstm', 'ensemble')
            
        Returns:
            Dictionary with prepared data
        """
        try:
            result = {}
            
            if model_type == 'sklearn':
                # Standard sklearn format
                X, y = self.split_features_target(df)
                X_scaled = self.transform_features(X)
                
                result = {
                    'X': X_scaled.values,
                    'y': y.values,
                    'feature_names': X_scaled.columns.tolist(),
                    'n_features': X_scaled.shape[1]
                }
                
            elif model_type == 'lstm':
                # LSTM sequence format
                from services.feature_engineering import FeatureEngineer
                feature_engineer = FeatureEngineer()
                
                X, y = self.split_features_target(df)
                X_scaled = self.transform_features(X)
                
                # Add target back for sequence creation
                df_for_sequences = X_scaled.copy()
                df_for_sequences[self.target_column] = y
                
                X_sequences, y_sequences = feature_engineer.prepare_sequences_for_lstm(df_for_sequences)
                
                result = {
                    'X': X_sequences,
                    'y': y_sequences,
                    'sequence_length': Config.LSTM_SEQUENCE_LENGTH,
                    'n_features': X_sequences.shape[2] if len(X_sequences.shape) > 2 else 0
                }
                
            elif model_type == 'ensemble':
                # Prepare data for ensemble (multiple formats)
                X, y = self.split_features_target(df)
                X_scaled = self.transform_features(X)
                
                # Standard format for most models
                sklearn_data = {
                    'X': X_scaled.values,
                    'y': y.values,
                    'feature_names': X_scaled.columns.tolist()
                }
                
                # LSTM format
                df_for_sequences = X_scaled.copy()
                df_for_sequences[self.target_column] = y
                
                from services.feature_engineering import FeatureEngineer
                feature_engineer = FeatureEngineer()
                X_sequences, y_sequences = feature_engineer.prepare_sequences_for_lstm(df_for_sequences)
                
                lstm_data = {
                    'X': X_sequences,
                    'y': y_sequences
                }
                
                result = {
                    'sklearn': sklearn_data,
                    'lstm': lstm_data,
                    'original_df': df
                }
            
            logger.info(f"Data prepared for {model_type} model")
            return result
            
        except Exception as e:
            logger.error(f"Failed to prepare data for {model_type}: {e}")
            return {}
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality for model training.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_results = {
                'is_valid': True,
                'issues': [],
                'warnings': [],
                'statistics': {}
            }
            
            # Check for missing values
            missing_counts = df.isnull().sum()
            missing_percentage = (missing_counts / len(df)) * 100
            
            high_missing = missing_percentage[missing_percentage > 50]
            if len(high_missing) > 0:
                validation_results['issues'].append(f"High missing values: {high_missing.to_dict()}")
                validation_results['is_valid'] = False
            
            moderate_missing = missing_percentage[(missing_percentage > 10) & (missing_percentage <= 50)]
            if len(moderate_missing) > 0:
                validation_results['warnings'].append(f"Moderate missing values: {moderate_missing.to_dict()}")
            
            # Check for infinite values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            inf_columns = []
            for col in numeric_columns:
                if np.isinf(df[col]).any():
                    inf_columns.append(col)
            
            if inf_columns:
                validation_results['issues'].append(f"Infinite values in: {inf_columns}")
                validation_results['is_valid'] = False
            
            # Check for constant columns
            constant_columns = []
            for col in numeric_columns:
                if df[col].nunique() <= 1:
                    constant_columns.append(col)
            
            if constant_columns:
                validation_results['warnings'].append(f"Constant columns: {constant_columns}")
            
            # Check data size
            if len(df) < 100:
                validation_results['issues'].append(f"Insufficient data: {len(df)} records")
                validation_results['is_valid'] = False
            elif len(df) < 500:
                validation_results['warnings'].append(f"Limited data: {len(df)} records")
            
            # Statistics
            validation_results['statistics'] = {
                'total_records': len(df),
                'total_features': len(df.columns),
                'numeric_features': len(numeric_columns),
                'missing_percentage': missing_percentage.mean(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            logger.info(f"Data validation completed: {'Valid' if validation_results['is_valid'] else 'Invalid'}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return {'is_valid': False, 'issues': [str(e)], 'warnings': [], 'statistics': {}}
    
    def clean_data_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data for model training.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        try:
            df_clean = df.copy()
            
            # Remove rows with too many missing values
            missing_threshold = 0.5  # Remove rows with >50% missing values
            df_clean = df_clean.dropna(thresh=int(len(df_clean.columns) * missing_threshold))
            
            # Forward fill remaining missing values (limited)
            df_clean = df_clean.ffill(limit=3)
            
            # Remove any remaining rows with missing values
            df_clean = df_clean.dropna()
            
            # Remove infinite values
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df_clean = df_clean[~np.isinf(df_clean[col])]
            
            # Remove extreme outliers (beyond 5 standard deviations)
            for col in numeric_columns:
                if col != 'timestamp':
                    mean_val = df_clean[col].mean()
                    std_val = df_clean[col].std()
                    
                    if std_val > 0:  # Avoid division by zero
                        outlier_mask = np.abs(df_clean[col] - mean_val) > (5 * std_val)
                        df_clean = df_clean[~outlier_mask]
            
            logger.info(f"Data cleaning: {len(df)} -> {len(df_clean)} records")
            return df_clean
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            return df