"""Data storage and validation system for BTC price data."""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from loguru import logger
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class BTCDataStorage:
    """Handles storage, validation, and retrieval of BTC price data."""
    
    def __init__(self):
        self.data_dir = Config.DATA_DIR
        self.historical_file = os.path.join(self.data_dir, 'btc_historical.csv')
        self.predictions_file = Config.PREDICTIONS_FILE
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.predictions_file), exist_ok=True)
    
    def validate_price_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate BTC price data for quality and consistency.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Check required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                issues.append(f"Missing columns: {missing_columns}")
            
            if not issues:  # Only continue if we have required columns
                # Check for null values
                null_counts = df[required_columns].isnull().sum()
                if null_counts.any():
                    issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
                
                # Check for negative prices
                price_columns = ['open', 'high', 'low', 'close']
                for col in price_columns:
                    if (df[col] <= 0).any():
                        issues.append(f"Non-positive values in {col}")
                
                # Check OHLC logic (High >= Low, etc.)
                if (df['high'] < df['low']).any():
                    issues.append("High price is less than low price in some records")
                
                if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
                    issues.append("High price is less than open/close in some records")
                
                if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
                    issues.append("Low price is greater than open/close in some records")
                
                # Check for extreme price movements (more than 50% in 5 minutes)
                df_sorted = df.sort_values('timestamp')
                price_changes = df_sorted['close'].pct_change().abs()
                extreme_changes = price_changes > 0.5
                if extreme_changes.any():
                    issues.append(f"Extreme price changes detected: {extreme_changes.sum()} records")
                
                # Check timestamp ordering and gaps
                if not df['timestamp'].is_monotonic_increasing:
                    issues.append("Timestamps are not in ascending order")
                
                # Check for duplicate timestamps
                if df['timestamp'].duplicated().any():
                    issues.append("Duplicate timestamps found")
                
                # Check for reasonable timestamp range (not too far in future/past)
                now = datetime.now()
                future_records = df['timestamp'] > now
                if future_records.any():
                    issues.append(f"Future timestamps found: {future_records.sum()} records")
                
                old_records = df['timestamp'] < (now - timedelta(days=365))
                if old_records.any():
                    issues.append(f"Very old timestamps (>1 year): {old_records.sum()} records")
        
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and fix common issues in price data.
        
        Args:
            df: Raw price DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        try:
            df_clean = df.copy()
            
            # Remove duplicates based on timestamp
            df_clean = df_clean.drop_duplicates(subset=['timestamp'], keep='last')
            
            # Sort by timestamp
            df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
            
            # Forward fill small gaps (up to 3 missing intervals)
            df_clean = df_clean.set_index('timestamp')
            df_clean = df_clean.resample(f'{Config.DATA_INTERVAL_MINUTES}min').last()
            
            # Forward fill missing values (limited to 3 periods)
            df_clean = df_clean.fillna(method='ffill', limit=3)
            
            # Remove any remaining NaN rows
            df_clean = df_clean.dropna()
            
            # Reset index to get timestamp back as column
            df_clean = df_clean.reset_index()
            
            # Fix OHLC inconsistencies
            for idx in df_clean.index:
                row = df_clean.loc[idx]
                
                # Ensure high is the maximum and low is the minimum
                prices = [row['open'], row['high'], row['low'], row['close']]
                df_clean.loc[idx, 'high'] = max(prices)
                df_clean.loc[idx, 'low'] = min(prices)
            
            # Remove extreme outliers (more than 3 standard deviations)
            price_changes = df_clean['close'].pct_change()
            mean_change = price_changes.mean()
            std_change = price_changes.std()
            
            outlier_threshold = 3 * std_change
            outliers = abs(price_changes - mean_change) > outlier_threshold
            
            if outliers.any():
                logger.warning(f"Removing {outliers.sum()} outlier records")
                df_clean = df_clean[~outliers].reset_index(drop=True)
            
            logger.info(f"Data cleaning completed: {len(df)} -> {len(df_clean)} records")
            return df_clean
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            return df
    
    def save_historical_data(self, df: pd.DataFrame, validate: bool = True) -> bool:
        """
        Save historical price data to CSV file.
        
        Args:
            df: DataFrame with price data
            validate: Whether to validate data before saving
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if validate:
                is_valid, issues = self.validate_price_data(df)
                if not is_valid:
                    logger.warning(f"Data validation issues: {issues}")
                    df = self.clean_price_data(df)
                    
                    # Re-validate after cleaning
                    is_valid, issues = self.validate_price_data(df)
                    if not is_valid:
                        logger.error(f"Data still invalid after cleaning: {issues}")
                        return False
            
            # Save to CSV
            df.to_csv(self.historical_file, index=False)
            logger.info(f"Historical data saved: {len(df)} records to {self.historical_file}")
            
            # Create backup
            backup_file = f"{self.historical_file}.backup"
            df.to_csv(backup_file, index=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save historical data: {e}")
            return False
    
    def load_historical_data(self) -> Optional[pd.DataFrame]:
        """
        Load historical price data from CSV file.
        
        Returns:
            DataFrame with historical data or None if failed
        """
        try:
            if not os.path.exists(self.historical_file):
                logger.warning("Historical data file not found")
                return None
            
            df = pd.read_csv(self.historical_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"Loaded historical data: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            
            # Try backup file
            backup_file = f"{self.historical_file}.backup"
            if os.path.exists(backup_file):
                try:
                    logger.info("Trying backup file")
                    df = pd.read_csv(backup_file)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    logger.info(f"Loaded from backup: {len(df)} records")
                    return df
                except Exception as backup_error:
                    logger.error(f"Backup file also failed: {backup_error}")
            
            return None
    
    def get_latest_records(self, n: int = 100) -> Optional[pd.DataFrame]:
        """
        Get the latest N records from historical data.
        
        Args:
            n: Number of latest records to retrieve
            
        Returns:
            DataFrame with latest records or None if failed
        """
        try:
            df = self.load_historical_data()
            if df is None:
                return None
            
            latest_df = df.tail(n).copy()
            logger.debug(f"Retrieved latest {len(latest_df)} records")
            return latest_df
            
        except Exception as e:
            logger.error(f"Failed to get latest records: {e}")
            return None
    
    def save_prediction(self, timestamp: datetime, predicted_price: float, 
                       confidence_interval: Tuple[float, float], 
                       model_contributions: Dict[str, float],
                       actual_price: Optional[float] = None) -> bool:
        """
        Save a prediction record to the predictions file.
        
        Args:
            timestamp: When the prediction was made
            predicted_price: Predicted BTC price
            confidence_interval: Lower and upper bounds
            model_contributions: Individual model predictions
            actual_price: Actual price (filled later)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            prediction_record = {
                'timestamp': timestamp,
                'predicted_price': predicted_price,
                'confidence_lower': confidence_interval[0],
                'confidence_upper': confidence_interval[1],
                'actual_price': actual_price,
                'error': None if actual_price is None else abs(predicted_price - actual_price),
                'error_percentage': None if actual_price is None else abs(predicted_price - actual_price) / actual_price * 100,
                **{f'model_{k}': v for k, v in model_contributions.items()}
            }
            
            # Create DataFrame for this prediction
            pred_df = pd.DataFrame([prediction_record])
            
            # Append to existing file or create new
            if os.path.exists(self.predictions_file):
                pred_df.to_csv(self.predictions_file, mode='a', header=False, index=False)
            else:
                pred_df.to_csv(self.predictions_file, index=False)
            
            logger.debug(f"Prediction saved: ${predicted_price:.2f} at {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            return False
    
    def load_predictions(self, days: int = 7) -> Optional[pd.DataFrame]:
        """
        Load prediction history from the last N days.
        
        Args:
            days: Number of days of predictions to load
            
        Returns:
            DataFrame with predictions or None if failed
        """
        try:
            if not os.path.exists(self.predictions_file):
                logger.info("No predictions file found")
                return pd.DataFrame()
            
            df = pd.read_csv(self.predictions_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter for last N days
            cutoff_time = datetime.now() - timedelta(days=days)
            recent_df = df[df['timestamp'] >= cutoff_time].copy()
            
            logger.info(f"Loaded {len(recent_df)} predictions from last {days} days")
            return recent_df
            
        except Exception as e:
            logger.error(f"Failed to load predictions: {e}")
            return None
    
    def update_prediction_actual(self, timestamp: datetime, actual_price: float) -> bool:
        """
        Update a prediction record with the actual price.
        
        Args:
            timestamp: Timestamp of the prediction to update
            actual_price: The actual BTC price observed
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            if not os.path.exists(self.predictions_file):
                logger.warning("No predictions file to update")
                return False
            
            df = pd.read_csv(self.predictions_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Find the prediction to update
            mask = df['timestamp'] == timestamp
            if not mask.any():
                logger.warning(f"No prediction found for timestamp {timestamp}")
                return False
            
            # Update actual price and calculate error
            df.loc[mask, 'actual_price'] = actual_price
            predicted_price = df.loc[mask, 'predicted_price'].iloc[0]
            
            error = abs(predicted_price - actual_price)
            error_percentage = error / actual_price * 100
            
            df.loc[mask, 'error'] = error
            df.loc[mask, 'error_percentage'] = error_percentage
            
            # Save updated file
            df.to_csv(self.predictions_file, index=False)
            
            logger.debug(f"Updated prediction: predicted=${predicted_price:.2f}, actual=${actual_price:.2f}, error={error_percentage:.2f}%")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update prediction: {e}")
            return False
    
    def get_data_statistics(self) -> Dict:
        """
        Get statistics about the stored data.
        
        Returns:
            Dictionary with data statistics
        """
        stats = {
            'historical_records': 0,
            'prediction_records': 0,
            'data_range': None,
            'last_update': None,
            'data_quality': 'unknown'
        }
        
        try:
            # Historical data stats
            historical_df = self.load_historical_data()
            if historical_df is not None:
                stats['historical_records'] = len(historical_df)
                stats['data_range'] = (
                    historical_df['timestamp'].min(),
                    historical_df['timestamp'].max()
                )
                stats['last_update'] = historical_df['timestamp'].max()
                
                # Check data quality
                is_valid, issues = self.validate_price_data(historical_df)
                stats['data_quality'] = 'good' if is_valid else f'issues: {len(issues)}'
            
            # Prediction stats
            predictions_df = self.load_predictions(days=30)
            if predictions_df is not None:
                stats['prediction_records'] = len(predictions_df)
            
        except Exception as e:
            logger.error(f"Failed to get data statistics: {e}")
        
        return stats