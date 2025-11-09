"""LSTM neural network model for BTC price prediction."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Any
from loguru import logger
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class LSTMPredictor:
    """LSTM model for capturing temporal and sequential dependencies."""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.sequence_length = Config.LSTM_SEQUENCE_LENGTH
        self.layers = Config.LSTM_LAYERS
        self.batch_size = Config.LSTM_BATCH_SIZE
        self.epochs = Config.LSTM_EPOCHS
        self.feature_columns = []
        self.training_history = None
        self.scaler = None
        
    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training."""
        try:
            # Use feature engineering to create sequences
            from services.feature_engineering import FeatureEngineer
            feature_engineer = FeatureEngineer()
            
            X_sequences, y_sequences = feature_engineer.prepare_sequences_for_lstm(df)
            
            logger.debug(f"LSTM sequences prepared: {X_sequences.shape}")
            return X_sequences, y_sequences
            
        except Exception as e:
            logger.error(f"Failed to prepare LSTM sequences: {e}")
            return np.array([]), np.array([])
    
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build LSTM model architecture."""
        try:
            model = Sequential()
            
            # Build LSTM layers dynamically based on configuration
            for i, units in enumerate(self.layers):
                if i == 0:
                    # First layer with input shape
                    model.add(LSTM(
                        units, 
                        return_sequences=(i < len(self.layers) - 1),  # Return sequences if not last layer
                        input_shape=input_shape
                    ))
                else:
                    # Subsequent layers
                    model.add(LSTM(
                        units, 
                        return_sequences=(i < len(self.layers) - 1)  # Return sequences if not last layer
                    ))
                
                # Add dropout after each LSTM layer
                model.add(Dropout(0.2))
            
            # Output layer
            model.add(Dense(1))
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            logger.info(f"LSTM model built: {model.count_params()} parameters")
            return model
            
        except Exception as e:
            logger.error(f"Failed to build LSTM model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def train(self, df: pd.DataFrame, validation_split: float = 0.2) -> bool:
        """Train the LSTM model."""
        try:
            logger.info("Training LSTM model...")
            
            # Prepare sequences
            X, y = self.prepare_sequences(df)
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No valid sequences for LSTM training")
                return False
            
            if len(X) < 100:
                logger.error(f"Insufficient sequences for LSTM training: {len(X)}")
                return False
            
            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Build model
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self.build_model(input_shape)
            
            if self.model is None:
                return False
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint('best_lstm_model.h5', save_best_only=True)
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            self.training_history = history.history
            self.is_trained = True
            
            # Calculate final scores
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            
            logger.info("LSTM model trained successfully")
            logger.info(f"Final training loss: {train_loss:.4f}")
            logger.info(f"Final validation loss: {val_loss:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            return False
    
    def predict(self, df: pd.DataFrame) -> Tuple[float, Tuple[float, float]]:
        """Make prediction using the trained LSTM model."""
        try:
            if not self.is_trained or self.model is None:
                logger.error("LSTM model not trained")
                return 0.0, (0.0, 0.0)
            
            # Prepare sequences
            X, _ = self.prepare_sequences(df)
            
            if len(X) == 0:
                logger.error("No valid sequences for LSTM prediction")
                return 0.0, (0.0, 0.0)
            
            # Make prediction (use last sequence)
            prediction = self.model.predict(X[-1:], verbose=0)
            pred_value = float(prediction[0][0])
            
            # Estimate confidence interval (simple approach)
            margin = abs(pred_value) * 0.05
            lower_bound = pred_value - margin
            upper_bound = pred_value + margin
            
            logger.debug(f"LSTM prediction: {pred_value:.2f}")
            return pred_value, (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return 0.0, (0.0, 0.0)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        info = {
            'model_type': 'LSTM',
            'is_trained': self.is_trained,
            'sequence_length': self.sequence_length,
            'layers': self.layers,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }
        
        if self.is_trained and self.model is not None:
            info.update({
                'total_params': self.model.count_params(),
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape
            })
        
        return info
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model to file."""
        try:
            if not self.is_trained:
                logger.error("Cannot save untrained LSTM model")
                return False
            
            # Save Keras model
            model_path = filepath.replace('.joblib', '.h5')
            self.model.save(model_path)
            
            logger.info(f"LSTM model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save LSTM model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model from file."""
        try:
            model_path = filepath.replace('.joblib', '.h5')
            
            if not os.path.exists(model_path):
                logger.error(f"LSTM model file not found: {model_path}")
                return False
            
            self.model = tf.keras.models.load_model(model_path)
            self.is_trained = True
            
            logger.info(f"LSTM model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            return False