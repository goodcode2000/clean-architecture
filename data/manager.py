"""Data manager that coordinates collection and storage of BTC price data."""
import pandas as pd
import schedule
import time
from datetime import datetime, timedelta
from typing import Optional, Dict
from loguru import logger
import threading
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from data.collector import BTCDataCollector
from data.storage import BTCDataStorage

class BTCDataManager:
    """Manages BTC price data collection, storage, and updates."""
    
    def __init__(self):
        self.collector = BTCDataCollector()
        self.storage = BTCDataStorage()
        self.update_interval = Config.DATA_INTERVAL_MINUTES
        self.is_running = False
        self.update_thread = None
        
    def initialize_data(self, force_refresh: bool = False) -> bool:
        """
        Initialize the data system with historical data.
        
        Args:
            force_refresh: If True, fetch fresh data even if cached data exists
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing BTC data system...")
            
            # Check if we have existing data
            existing_data = self.storage.load_historical_data()
            
            if existing_data is not None and not force_refresh:
                # Check if existing data is recent enough
                latest_time = existing_data['timestamp'].max()
                time_diff = datetime.now() - latest_time
                
                if time_diff < timedelta(hours=1):
                    logger.info("Using existing historical data")
                    return True
                else:
                    logger.info("Existing data is outdated, fetching fresh data")
            
            # Fetch historical data
            historical_data = self.collector.fetch_historical_data(force_refresh=force_refresh)
            
            if historical_data is None:
                logger.error("Failed to fetch historical data")
                return False
            
            # Save historical data
            if not self.storage.save_historical_data(historical_data):
                logger.error("Failed to save historical data")
                return False
            
            logger.info("Data initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Data initialization failed: {e}")
            return False
    
    def get_current_price(self) -> Optional[float]:
        """
        Get the current BTC price.
        
        Returns:
            Current BTC price or None if failed
        """
        return self.collector.get_current_price()
    
    def get_latest_data(self, n_records: int = 100) -> Optional[pd.DataFrame]:
        """
        Get the latest N records of historical data.
        
        Args:
            n_records: Number of latest records to retrieve
            
        Returns:
            DataFrame with latest data or None if failed
        """
        return self.storage.get_latest_records(n_records)
    
    def update_data_once(self) -> bool:
        """
        Perform a single data update.
        
        Returns:
            True if update successful, False otherwise
        """
        try:
            logger.debug("Performing data update...")
            
            # Update historical data with latest records
            updated_data = self.collector.update_data()
            
            if updated_data is None:
                logger.warning("Data update returned no data")
                return False
            
            # Save updated data
            if not self.storage.save_historical_data(updated_data, validate=False):
                logger.error("Failed to save updated data")
                return False
            
            logger.debug("Data update completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Data update failed: {e}")
            return False
    
    def start_automatic_updates(self):
        """Start automatic data updates every 5 minutes."""
        if self.is_running:
            logger.warning("Automatic updates already running")
            return
        
        logger.info(f"Starting automatic data updates every {self.update_interval} minutes")
        
        # Schedule updates
        schedule.every(self.update_interval).minutes.do(self.update_data_once)
        
        self.is_running = True
        
        # Run scheduler in separate thread
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
        
        self.update_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.update_thread.start()
        
        logger.info("Automatic data updates started")
    
    def stop_automatic_updates(self):
        """Stop automatic data updates."""
        if not self.is_running:
            logger.warning("Automatic updates not running")
            return
        
        logger.info("Stopping automatic data updates")
        self.is_running = False
        
        # Clear scheduled jobs
        schedule.clear()
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)
        
        logger.info("Automatic data updates stopped")
    
    def get_data_for_training(self, lookback_days: int = None) -> Optional[pd.DataFrame]:
        """
        Get data prepared for model training.
        
        Args:
            lookback_days: Number of days to look back (default: Config.HISTORICAL_DAYS)
            
        Returns:
            DataFrame ready for training or None if failed
        """
        try:
            if lookback_days is None:
                lookback_days = Config.HISTORICAL_DAYS
            
            # Load historical data
            df = self.storage.load_historical_data()
            
            if df is None:
                logger.error("No historical data available for training")
                return None
            
            # Filter for specified lookback period
            cutoff_time = datetime.now() - timedelta(days=lookback_days)
            training_data = df[df['timestamp'] >= cutoff_time].copy()
            
            if len(training_data) < 100:  # Minimum data requirement
                logger.error(f"Insufficient data for training: {len(training_data)} records")
                return None
            
            # Sort by timestamp
            training_data = training_data.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Prepared {len(training_data)} records for training")
            return training_data
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return None
    
    def save_prediction(self, predicted_price: float, confidence_interval: tuple,
                       model_contributions: Dict[str, float]) -> bool:
        """
        Save a prediction made by the ensemble model.
        
        Args:
            predicted_price: The predicted BTC price
            confidence_interval: Lower and upper bounds
            model_contributions: Individual model predictions
            
        Returns:
            True if saved successfully, False otherwise
        """
        timestamp = datetime.now()
        return self.storage.save_prediction(
            timestamp=timestamp,
            predicted_price=predicted_price,
            confidence_interval=confidence_interval,
            model_contributions=model_contributions
        )
    
    def update_prediction_with_actual(self, prediction_time: datetime, actual_price: float) -> bool:
        """
        Update a prediction with the actual observed price.
        
        Args:
            prediction_time: When the prediction was made
            actual_price: The actual price observed
            
        Returns:
            True if updated successfully, False otherwise
        """
        return self.storage.update_prediction_actual(prediction_time, actual_price)
    
    def get_prediction_accuracy(self, days: int = 7) -> Dict:
        """
        Calculate prediction accuracy metrics for the last N days.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with accuracy metrics
        """
        try:
            predictions_df = self.storage.load_predictions(days=days)
            
            if predictions_df is None or len(predictions_df) == 0:
                return {'error': 'No predictions available'}
            
            # Filter for predictions with actual prices
            completed_predictions = predictions_df.dropna(subset=['actual_price'])
            
            if len(completed_predictions) == 0:
                return {'error': 'No completed predictions available'}
            
            # Calculate metrics
            errors = completed_predictions['error']
            error_percentages = completed_predictions['error_percentage']
            
            metrics = {
                'total_predictions': len(predictions_df),
                'completed_predictions': len(completed_predictions),
                'mean_absolute_error': errors.mean(),
                'median_absolute_error': errors.median(),
                'mean_percentage_error': error_percentages.mean(),
                'median_percentage_error': error_percentages.median(),
                'max_error': errors.max(),
                'min_error': errors.min(),
                'accuracy_within_1_percent': (error_percentages <= 1.0).sum() / len(completed_predictions) * 100,
                'accuracy_within_5_percent': (error_percentages <= 5.0).sum() / len(completed_predictions) * 100,
            }
            
            logger.info(f"Prediction accuracy (last {days} days): {metrics['mean_percentage_error']:.2f}% average error")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate prediction accuracy: {e}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict:
        """
        Get overall system status and statistics.
        
        Returns:
            Dictionary with system status information
        """
        try:
            status = {
                'data_manager_running': self.is_running,
                'last_update_check': datetime.now(),
                'current_price': self.get_current_price(),
            }
            
            # Add data statistics
            data_stats = self.storage.get_data_statistics()
            status.update(data_stats)
            
            # Add recent accuracy
            accuracy = self.get_prediction_accuracy(days=1)
            if 'error' not in accuracy:
                status['recent_accuracy'] = accuracy.get('mean_percentage_error', 'N/A')
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Clean up resources and stop background processes."""
        logger.info("Cleaning up data manager...")
        self.stop_automatic_updates()
        logger.info("Data manager cleanup completed")