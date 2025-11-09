"""Automated training and prediction pipeline for BTC price prediction."""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional
from loguru import logger
import schedule
import time
import threading
from datetime import datetime, timedelta
import sys
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import TimeSeriesSplit

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from data.manager import BTCDataManager
from models.ensemble_model import EnsemblePredictor
from services.offset_correction import OffsetCorrectionSystem
from services.feature_engineering import FeatureEngineer
from services.preprocessing import DataPreprocessor

class PredictionPipeline:
    """Automated pipeline for training models and making predictions every 5 minutes."""
    
    def __init__(self):
        # Core components
        self.data_manager = BTCDataManager()
        self.ensemble_model = EnsemblePredictor()
        self.offset_correction = OffsetCorrectionSystem()
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = DataPreprocessor()
        
        # Pipeline state
        self.is_running = False
        self.pipeline_thread = None
        self.last_prediction_time = None
        self.last_training_time = None
        self.prediction_interval = Config.DATA_INTERVAL_MINUTES  # 5 minutes
        self.retrain_interval_hours = Config.RETRAIN_INTERVAL_HOURS  # 6 hours
        
        # Performance tracking
        self.prediction_history = []
        self.training_history = []
        self.pipeline_metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'total_retrains': 0,
            'successful_retrains': 0,
            'last_error': None
        }
        
        # Market condition validation
        self.market_conditions = {
            'bullish_threshold': 0.02,  # 2% price increase
            'bearish_threshold': -0.02,  # 2% price decrease
            'high_volatility_threshold': 0.05  # 5% price change
        }
        
    def initialize_pipeline(self) -> bool:
        """
        Initialize the prediction pipeline.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing prediction pipeline...")
            
            # Initialize data manager
            if not self.data_manager.initialize_data():
                logger.error("Failed to initialize data manager")
                return False
            
            # Start automatic data updates
            self.data_manager.start_automatic_updates()
            
            # Initial model training (full training including LSTM)
            if not self.train_models(initial_training=True):
                logger.error("Initial model training failed")
                return False
            
            logger.info("Prediction pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return False
    
    def train_models(self, force_retrain: bool = False, initial_training: bool = False) -> bool:
        """
        Train or retrain the ensemble models.
        
        Args:
            force_retrain: Force retraining even if recently trained
            initial_training: If True, perform full training including LSTM (slower)
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            # Check if retraining is needed
            if not force_retrain and self.last_training_time:
                time_since_training = datetime.now() - self.last_training_time
                if time_since_training < timedelta(hours=self.retrain_interval_hours):
                    logger.debug("Skipping training - too soon since last training")
                    return True
            
            # Determine if we should skip LSTM for fast retraining
            skip_lstm = not initial_training  # Skip LSTM for periodic retrains, include for initial
            
            if skip_lstm:
                logger.info("Starting fast model retraining (LSTM excluded)...")
            else:
                logger.info("Starting full model training (including LSTM)...")
            
            # Get training data
            training_data = self.data_manager.get_data_for_training()
            
            if training_data is None or len(training_data) < 100:
                logger.error("Insufficient training data")
                return False
            
            # Validate market conditions in training data
            if not self.validate_training_data(training_data):
                logger.warning("Training data validation issues detected")
            
            # Handle class imbalance for rapid movement detection
            enhanced_data = self.handle_class_imbalance(training_data)
            
            # Train ensemble model
            training_start = datetime.now()
            success = self.ensemble_model.train(enhanced_data, skip_lstm=skip_lstm)
            training_duration = datetime.now() - training_start
            
            if success:
                self.last_training_time = datetime.now()
                self.pipeline_metrics['total_retrains'] += 1
                self.pipeline_metrics['successful_retrains'] += 1
                
                # Record training history
                training_record = {
                    'timestamp': self.last_training_time,
                    'duration_seconds': training_duration.total_seconds(),
                    'training_samples': len(enhanced_data),
                    'success': True,
                    'lstm_included': not skip_lstm,
                    'model_info': self.ensemble_model.get_model_info()
                }
                
                self.training_history.append(training_record)
                
                # Keep only recent training history
                if len(self.training_history) > 50:
                    self.training_history = self.training_history[-50:]
                
                if skip_lstm:
                    logger.info(f"Fast retraining completed in {training_duration.total_seconds():.1f}s (LSTM skipped)")
                else:
                    logger.info(f"Full training completed in {training_duration.total_seconds():.1f}s (LSTM included)")
                return True
            else:
                self.pipeline_metrics['total_retrains'] += 1
                logger.error("Model training failed")
                return False
                
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            self.pipeline_metrics['last_error'] = str(e)
            return False
    
    def handle_class_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle class imbalance using SMOTE for rapid movement detection.
        
        Args:
            df: Training data
            
        Returns:
            Enhanced training data
        """
        try:
            logger.info("Handling class imbalance for rapid movement detection...")
            
            # Create rapid movement labels
            df_enhanced = df.copy()
            
            # Calculate price changes
            price_changes = df_enhanced['close'].pct_change()
            
            # Label rapid movements
            rapid_rise = price_changes > self.market_conditions['bullish_threshold']
            rapid_fall = price_changes < self.market_conditions['bearish_threshold']
            high_volatility = abs(price_changes) > self.market_conditions['high_volatility_threshold']
            
            # Create binary labels for rapid movements
            df_enhanced['rapid_movement'] = (rapid_rise | rapid_fall | high_volatility).astype(int)
            
            # Check class distribution
            rapid_count = df_enhanced['rapid_movement'].sum()
            normal_count = len(df_enhanced) - rapid_count
            
            logger.info(f"Class distribution - Normal: {normal_count}, Rapid: {rapid_count}")
            
            # Apply SMOTE if imbalance is significant
            if rapid_count > 0 and normal_count / rapid_count > 3:  # If imbalance ratio > 3:1
                try:
                    # Prepare features for SMOTE
                    feature_columns = [col for col in df_enhanced.columns 
                                     if col not in ['timestamp', 'rapid_movement']]
                    
                    X = df_enhanced[feature_columns].fillna(0)
                    y = df_enhanced['rapid_movement']
                    
                    # Apply SMOTE
                    smote = SMOTE(random_state=42, k_neighbors=min(5, rapid_count-1))
                    X_resampled, y_resampled = smote.fit_resample(X, y)
                    
                    # Create resampled DataFrame
                    df_resampled = pd.DataFrame(X_resampled, columns=feature_columns)
                    df_resampled['rapid_movement'] = y_resampled
                    
                    # Add back timestamp (approximate)
                    df_resampled['timestamp'] = pd.date_range(
                        start=df_enhanced['timestamp'].min(),
                        periods=len(df_resampled),
                        freq='5min'
                    )
                    
                    logger.info(f"SMOTE applied: {len(df_enhanced)} -> {len(df_resampled)} samples")
                    return df_resampled
                    
                except Exception as e:
                    logger.warning(f"SMOTE failed, using original data: {e}")
            
            return df_enhanced
            
        except Exception as e:
            logger.error(f"Class imbalance handling failed: {e}")
            return df
    
    def validate_training_data(self, df: pd.DataFrame) -> bool:
        """
        Validate training data includes both bullish and bearish conditions.
        
        Args:
            df: Training data
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            if len(df) < 100:
                logger.warning("Insufficient data for market condition validation")
                return False
            
            # Calculate price changes
            price_changes = df['close'].pct_change().dropna()
            
            # Check for bullish periods
            bullish_periods = (price_changes > self.market_conditions['bullish_threshold']).sum()
            bearish_periods = (price_changes < self.market_conditions['bearish_threshold']).sum()
            high_vol_periods = (abs(price_changes) > self.market_conditions['high_volatility_threshold']).sum()
            
            total_periods = len(price_changes)
            
            bullish_pct = (bullish_periods / total_periods) * 100
            bearish_pct = (bearish_periods / total_periods) * 100
            high_vol_pct = (high_vol_periods / total_periods) * 100
            
            logger.info(f"Market conditions in training data:")
            logger.info(f"  Bullish periods: {bullish_pct:.1f}% ({bullish_periods})")
            logger.info(f"  Bearish periods: {bearish_pct:.1f}% ({bearish_periods})")
            logger.info(f"  High volatility: {high_vol_pct:.1f}% ({high_vol_periods})")
            
            # Validation criteria
            min_condition_pct = 5.0  # At least 5% of each condition
            
            validation_passed = True
            
            if bullish_pct < min_condition_pct:
                logger.warning(f"Insufficient bullish conditions: {bullish_pct:.1f}% < {min_condition_pct}%")
                validation_passed = False
            
            if bearish_pct < min_condition_pct:
                logger.warning(f"Insufficient bearish conditions: {bearish_pct:.1f}% < {min_condition_pct}%")
                validation_passed = False
            
            return validation_passed
            
        except Exception as e:
            logger.error(f"Training data validation failed: {e}")
            return False
    
    def make_prediction(self) -> Optional[Dict[str, Any]]:
        """
        Make a prediction using the ensemble model.
        
        Returns:
            Dictionary with prediction results or None if failed
        """
        try:
            # Get current data for prediction
            current_data = self.data_manager.get_latest_data(n_records=100)
            
            if current_data is None or len(current_data) == 0:
                logger.error("No current data available for prediction")
                return None
            
            # Get current price for comparison
            current_price = self.data_manager.get_current_price()
            
            if current_price is None:
                logger.error("Could not get current price")
                return None
            
            # Make ensemble prediction
            prediction, confidence_interval = self.ensemble_model.predict(current_data)
            
            if prediction == 0.0:
                logger.error("Ensemble prediction failed")
                return None
            
            # Apply offset correction
            corrected_prediction = self.offset_correction.apply_correction('ensemble', prediction)
            
            # Get individual model contributions
            model_contributions = self.ensemble_model.model_contributions.copy()
            
            # Apply corrections to individual models
            corrected_contributions = {}
            for model_name, contrib in model_contributions.items():
                corrected_contributions[model_name] = self.offset_correction.apply_correction(model_name, contrib)
            
            # Create prediction record
            prediction_record = {
                'timestamp': datetime.now(),
                'current_price': current_price,
                'raw_prediction': prediction,
                'corrected_prediction': corrected_prediction,
                'confidence_interval': confidence_interval,
                'model_contributions': model_contributions,
                'corrected_contributions': corrected_contributions,
                'rapid_movement_detected': self.ensemble_model.rapid_movement_detected,
                'market_volatility': self.offset_correction.current_volatility_regime
            }
            
            # Save prediction
            self.data_manager.save_prediction(
                predicted_price=corrected_prediction,
                confidence_interval=confidence_interval,
                model_contributions=corrected_contributions
            )
            
            # Update pipeline metrics
            self.pipeline_metrics['total_predictions'] += 1
            self.pipeline_metrics['successful_predictions'] += 1
            self.last_prediction_time = datetime.now()
            
            # Store in prediction history
            self.prediction_history.append(prediction_record)
            
            # Keep only recent predictions
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            
            logger.info(f"Prediction made: ${corrected_prediction:.2f} (raw: ${prediction:.2f})")
            logger.info(f"Current price: ${current_price:.2f}, Confidence: [{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}]")
            
            return prediction_record
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            self.pipeline_metrics['last_error'] = str(e)
            return None
    
    def update_prediction_accuracy(self):
        """Update prediction accuracy with actual prices."""
        try:
            # Get predictions that need actual price updates
            cutoff_time = datetime.now() - timedelta(minutes=self.prediction_interval + 1)
            
            for prediction in self.prediction_history:
                pred_time = prediction['timestamp']
                
                # Check if enough time has passed and we haven't updated yet
                if pred_time < cutoff_time and 'actual_price' not in prediction:
                    # Get actual price at prediction time + 5 minutes
                    actual_price = self.data_manager.get_current_price()
                    
                    if actual_price:
                        prediction['actual_price'] = actual_price
                        
                        # Calculate error
                        predicted_price = prediction['corrected_prediction']
                        error = predicted_price - actual_price
                        
                        prediction['error'] = error
                        prediction['error_percentage'] = (error / actual_price) * 100
                        
                        # Update offset correction system
                        self.offset_correction.add_prediction_error(
                            'ensemble', predicted_price, actual_price, pred_time
                        )
                        
                        # Update individual model corrections
                        for model_name, contrib in prediction['corrected_contributions'].items():
                            self.offset_correction.add_prediction_error(
                                model_name, contrib, actual_price, pred_time
                            )
                        
                        # Update data manager
                        self.data_manager.update_prediction_with_actual(pred_time, actual_price)
                        
                        logger.debug(f"Updated prediction accuracy: predicted=${predicted_price:.2f}, actual=${actual_price:.2f}, error={error:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to update prediction accuracy: {e}")
    
    def run_prediction_cycle(self):
        """Run a single prediction cycle."""
        try:
            logger.info("Running prediction cycle...")
            
            # Update prediction accuracy from previous predictions
            self.update_prediction_accuracy()
            
            # Update offset corrections
            self.offset_correction.update_all_corrections()
            
            # Make new prediction
            prediction_result = self.make_prediction()
            
            if prediction_result:
                logger.info("Prediction cycle completed successfully")
            else:
                logger.error("Prediction cycle failed")
            
            # Check if retraining is needed
            if self.should_retrain():
                logger.info("Triggering model retraining...")
                self.train_models()
            
        except Exception as e:
            logger.error(f"Prediction cycle failed: {e}")
            self.pipeline_metrics['last_error'] = str(e)
    
    def should_retrain(self) -> bool:
        """
        Determine if models should be retrained.
        
        Returns:
            True if retraining is needed, False otherwise
        """
        try:
            # Time-based retraining (fast retrain without LSTM)
            if self.last_training_time:
                time_since_training = datetime.now() - self.last_training_time
                if time_since_training >= timedelta(hours=self.retrain_interval_hours):
                    logger.info("Fast retraining due to time interval (LSTM excluded)")
                    return True
            
            # Performance-based retraining (fast retrain without LSTM)
            if len(self.prediction_history) >= 20:
                recent_predictions = [p for p in self.prediction_history[-20:] if 'error_percentage' in p]
                
                if len(recent_predictions) >= 10:
                    avg_error = np.mean([abs(p['error_percentage']) for p in recent_predictions])
                    
                    if avg_error > 10.0:  # If average error > 10%
                        logger.info(f"Fast retraining due to high error rate: {avg_error:.2f}%")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check retraining criteria: {e}")
            return False
    
    def retrain_lstm_full(self) -> bool:
        """
        Perform full retraining including LSTM model.
        This is slower but more comprehensive.
        Call this manually when needed (e.g., daily or weekly).
        
        Returns:
            True if retraining successful, False otherwise
        """
        try:
            logger.info("Starting full LSTM retraining (manual trigger)...")
            return self.train_models(force_retrain=True, initial_training=True)
        except Exception as e:
            logger.error(f"Full LSTM retraining failed: {e}")
            return False
    
    def start_pipeline(self):
        """Start the automated prediction pipeline."""
        if self.is_running:
            logger.warning("Pipeline already running")
            return
        
        logger.info("Starting automated prediction pipeline...")
        
        # Initialize pipeline
        if not self.initialize_pipeline():
            logger.error("Failed to initialize pipeline")
            return
        
        # Schedule prediction cycles
        schedule.every(self.prediction_interval).minutes.do(self.run_prediction_cycle)
        
        self.is_running = True
        
        # Run scheduler in separate thread
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
        
        self.pipeline_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.pipeline_thread.start()
        
        # Run initial prediction cycle
        self.run_prediction_cycle()
        
        logger.info(f"Prediction pipeline started - predictions every {self.prediction_interval} minutes")
    
    def stop_pipeline(self):
        """Stop the automated prediction pipeline."""
        if not self.is_running:
            logger.warning("Pipeline not running")
            return
        
        logger.info("Stopping prediction pipeline...")
        
        self.is_running = False
        schedule.clear()
        
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            self.pipeline_thread.join(timeout=5)
        
        # Stop data manager
        self.data_manager.cleanup()
        
        logger.info("Prediction pipeline stopped")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline status.
        
        Returns:
            Dictionary with pipeline status
        """
        try:
            status = {
                'is_running': self.is_running,
                'last_prediction_time': self.last_prediction_time,
                'last_training_time': self.last_training_time,
                'prediction_interval_minutes': self.prediction_interval,
                'retrain_interval_hours': self.retrain_interval_hours,
                'metrics': self.pipeline_metrics.copy(),
                'recent_predictions': len(self.prediction_history),
                'training_history': len(self.training_history)
            }
            
            # Add model status
            if self.ensemble_model.is_trained:
                status['model_status'] = self.ensemble_model.get_model_info()
            
            # Add offset correction status
            status['offset_correction'] = self.offset_correction.get_system_status()
            
            # Add data manager status
            status['data_status'] = self.data_manager.get_system_status()
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            return {'error': str(e)}