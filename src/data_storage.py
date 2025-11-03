"""
CSV-based data storage system with 90-day rolling window
"""

import os
import csv
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from .data_models import PriceData, Prediction, ModelPerformance
from .config import Config

logger = logging.getLogger(__name__)

class DataStorage:
    """Handles CSV-based storage with atomic operations and rolling windows"""
    
    def __init__(self, config: Config):
        self.config = config
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize storage system and create necessary files"""
        try:
            # Create CSV files with headers if they don't exist
            await self._ensure_csv_file(
                self.config.price_data_path,
                ['timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
            )
            
            await self._ensure_csv_file(
                self.config.predictions_path,
                ['id', 'timestamp', 'current_price', 'predicted_price', 'confidence_score', 
                 'model_contributions', 'features_used', 'prediction_horizon']
            )
            
            await self._ensure_csv_file(
                self.config.performance_path,
                ['model_name', 'timestamp', 'mae', 'rmse', 'accuracy_within_threshold', 
                 'rapid_change_detection_rate']
            )
            
            logger.info("Data storage initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing data storage: {e}")
            raise
    
    async def _ensure_csv_file(self, file_path: str, headers: List[str]):
        """Ensure CSV file exists with proper headers"""
        if not os.path.exists(file_path):
            async with self._lock:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                logger.info(f"Created CSV file: {file_path}")
    
    async def store_price_data(self, price_data: Dict[str, Any]) -> bool:
        """Store price data with atomic write operation"""
        try:
            # Validate and convert to PriceData object
            if isinstance(price_data, dict):
                validated_data = PriceData(
                    timestamp=price_data.get('timestamp', datetime.now()),
                    open_price=float(price_data['open_price']),
                    high_price=float(price_data['high_price']),
                    low_price=float(price_data['low_price']),
                    close_price=float(price_data['close_price']),
                    volume=float(price_data['volume'])
                )
            else:
                validated_data = price_data
            
            async with self._lock:
                # Append new data
                with open(self.config.price_data_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=validated_data.to_dict().keys())
                    writer.writerow(validated_data.to_dict())
                
                # Maintain 90-day rolling window
                await self._maintain_rolling_window(self.config.price_data_path, self.config.HISTORICAL_DAYS)
            
            logger.debug(f"Stored price data: ${validated_data.close_price:,.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing price data: {e}")
            return False
    
    async def store_prediction(self, prediction: Prediction) -> bool:
        """Store prediction data"""
        try:
            async with self._lock:
                with open(self.config.predictions_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=prediction.to_dict().keys())
                    writer.writerow(prediction.to_dict())
            
            logger.debug(f"Stored prediction: ${prediction.predicted_price:,.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            return False
    
    async def store_performance(self, performance: ModelPerformance) -> bool:
        """Store model performance metrics"""
        try:
            async with self._lock:
                with open(self.config.performance_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=performance.to_dict().keys())
                    writer.writerow(performance.to_dict())
            
            logger.debug(f"Stored performance for {performance.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing performance: {e}")
            return False
    
    async def get_recent_price_data(self, hours: int = 5) -> List[PriceData]:
        """Get recent price data for specified hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            async with self._lock:
                df = pd.read_csv(self.config.price_data_path)
                
                if df.empty:
                    return []
                
                # Filter by timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                recent_df = df[df['timestamp'] >= cutoff_time]
                
                # Convert to PriceData objects
                price_data_list = []
                for _, row in recent_df.iterrows():
                    price_data = PriceData.from_dict(row.to_dict())
                    price_data_list.append(price_data)
                
                return price_data_list
                
        except Exception as e:
            logger.error(f"Error getting recent price data: {e}")
            return []
    
    async def get_historical_data(self, days: int = 90) -> List[PriceData]:
        """Get historical price data for specified days"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            async with self._lock:
                df = pd.read_csv(self.config.price_data_path)
                
                if df.empty:
                    return []
                
                # Filter by timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                historical_df = df[df['timestamp'] >= cutoff_time]
                
                # Convert to PriceData objects
                price_data_list = []
                for _, row in historical_df.iterrows():
                    price_data = PriceData.from_dict(row.to_dict())
                    price_data_list.append(price_data)
                
                return price_data_list
                
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return []
    
    async def get_recent_predictions(self, count: int = 100) -> List[Prediction]:
        """Get recent predictions"""
        try:
            async with self._lock:
                df = pd.read_csv(self.config.predictions_path)
                
                if df.empty:
                    return []
                
                # Get most recent predictions
                recent_df = df.tail(count)
                
                # Convert to Prediction objects
                predictions = []
                for _, row in recent_df.iterrows():
                    prediction = Prediction.from_dict(row.to_dict())
                    predictions.append(prediction)
                
                return predictions
                
        except Exception as e:
            logger.error(f"Error getting recent predictions: {e}")
            return []
    
    async def _maintain_rolling_window(self, file_path: str, days: int):
        """Maintain rolling window by removing old data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # Read current data
            df = pd.read_csv(file_path)
            
            if df.empty:
                return
            
            # Filter to keep only recent data
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            filtered_df = df[df['timestamp'] >= cutoff_time]
            
            # Write back to file
            filtered_df.to_csv(file_path, index=False)
            
            removed_count = len(df) - len(filtered_df)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} old records from {os.path.basename(file_path)}")
                
        except Exception as e:
            logger.error(f"Error maintaining rolling window for {file_path}: {e}")
    
    async def backup_data(self) -> bool:
        """Create backup of all data files"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = os.path.join(self.config.DATA_DIR, 'backups', timestamp)
            os.makedirs(backup_dir, exist_ok=True)
            
            # Copy all CSV files to backup directory
            import shutil
            
            files_to_backup = [
                self.config.price_data_path,
                self.config.predictions_path,
                self.config.performance_path
            ]
            
            for file_path in files_to_backup:
                if os.path.exists(file_path):
                    backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, backup_path)
            
            logger.info(f"Data backup created: {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    async def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored data"""
        try:
            stats = {}
            
            # Price data statistics
            if os.path.exists(self.config.price_data_path):
                df = pd.read_csv(self.config.price_data_path)
                stats['price_data_count'] = len(df)
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    stats['price_data_range'] = {
                        'start': df['timestamp'].min().isoformat(),
                        'end': df['timestamp'].max().isoformat()
                    }
            
            # Predictions statistics
            if os.path.exists(self.config.predictions_path):
                df = pd.read_csv(self.config.predictions_path)
                stats['predictions_count'] = len(df)
            
            # Performance statistics
            if os.path.exists(self.config.performance_path):
                df = pd.read_csv(self.config.performance_path)
                stats['performance_records_count'] = len(df)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting data statistics: {e}")
            return {}