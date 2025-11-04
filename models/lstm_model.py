"""LSTM neural network model for BTC price prediction with multi-source data."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Any
from loguru import logger
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, MultiHeadAttention, GlobalAveragePooling1D,
    Input, Concatenate, LayerNormalization, Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class LSTMPredictor:
    """Enhanced LSTM model for multi-source market data analysis."""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.sequence_length = Config.SEQUENCE_LENGTH
        self.layers = Config.LSTM_LAYERS
        self.batch_size = Config.LSTM_BATCH_SIZE
        self.epochs = Config.LSTM_EPOCHS
        self.feature_groups = Config.FEATURE_GROUPS
        self.training_history = None
        self.scalers = {}  # One scaler per feature group
        
    def prepare_sequences(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prepare sequences for LSTM training with multiple feature groups."""
        try:
            # Use enhanced feature engineering
            from services.feature_engineering import FeatureEngineer
            feature_engineer = FeatureEngineer()
            
            # Get start and end times
            start_time = df.index.min()
            end_time = df.index.max()
            
            # Prepare all features including new data sources
            features_dict = feature_engineer.prepare_features(start_time, end_time)
            
            logger.debug(f"LSTM sequences prepared for {len(self.feature_groups)} feature groups")
            return features_dict
            return X_sequences, y_sequences
            
        except Exception as e:
            logger.error(f"Failed to prepare LSTM sequences: {e}")
            return np.array([]), np.array([])
    
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build LSTM model architecture."""
        try:
            model = Sequential()
            # First LSTM layer (return sequences for attention)
            model.add(LSTM(self.layers[0], return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))

            # Second LSTM layer
            model.add(LSTM(self.layers[1], return_sequences=True))
            model.add(Dropout(0.2))

            # Apply a lightweight multi-head attention over the sequence outputs
            # We'll use a TimeDistributed-like attention via MultiHeadAttention
            try:
                # Wrap sequential model with a functional attention block by inserting
                # a GlobalAveragePooling on the attention outputs.
                # Keras Sequential doesn't directly support inserting MHA easily here,
                # but MultiHeadAttention can be applied using the Sequential API if
                # the tensors are compatible. We'll append an attention layer that
                # expects (batch, seq, features).
                model.add(MultiHeadAttention(num_heads=4, key_dim=16))
            except Exception:
                # If not available, simply continue without attention
                pass

            # Pool across time and finish
            model.add(GlobalAveragePooling1D())
            model.add(Dropout(0.2))
            model.add(Dense(self.layers[2], activation='relu'))
            model.add(Dropout(0.2))
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
            return None
    
    def train(self, df: pd.DataFrame, validation_split: float = 0.2) -> bool:
        """Train the enhanced LSTM model with multi-source data."""
        try:
            logger.info("Training enhanced LSTM model...")
            
            # Prepare sequences with all feature groups
            feature_dict = self.prepare_sequences(df)
            
            if not feature_dict or not all(len(v) > 0 for v in feature_dict.values()):
                logger.error("No valid sequences for LSTM training")
                return False
            
            # Get sequence dimensions for each feature group
            feature_dims = {
                group: (arr.shape[0], arr.shape[1], arr.shape[2])
                for group, arr in feature_dict.items()
            }
            
            # Split data for each feature group
            train_data = {}
            val_data = {}
            for group, sequences in feature_dict.items():
                split_idx = int(len(sequences) * (1 - validation_split))
                train_data[group] = sequences[:split_idx]
                val_data[group] = sequences[split_idx:]
            
            # Build model with proper input shapes
            input_shapes = {
                group: (seq_len, timesteps, features)
                for group, (seq_len, timesteps, features) in feature_dims.items()
            }
            self.model = self.build_model(input_shapes)
            
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
        """Make prediction using the enhanced LSTM model."""
        try:
            if not self.is_trained or self.model is None:
                logger.error("LSTM model not trained")
                return 0.0, (0.0, 0.0)
            
            # Prepare sequences for all feature groups
            feature_dict = self.prepare_sequences(df)
            
            if not feature_dict or not all(len(v) > 0 for v in feature_dict.values()):
                logger.error("No valid sequences for LSTM prediction")
                return 0.0, (0.0, 0.0)
            
            # Prepare input dictionary for prediction
            model_inputs = {
                group: sequences[-1:]  # Use last sequence for each group
                for group, sequences in feature_dict.items()
            }
            
            # Make prediction
            prediction = self.model.predict(model_inputs, verbose=0)
            pred_value = float(prediction[0][0])
            
            # Calculate confidence interval using feature importance
            uncertainties = []
            for group in feature_dict:
                # Make predictions without this feature group
                tmp_inputs = model_inputs.copy()
                tmp_inputs[group] *= 0  # Zero out this feature group
                pred_without = self.model.predict(tmp_inputs, verbose=0)[0][0]
                uncertainties.append(abs(pred_value - pred_without))
            
            # Use the max uncertainty to set confidence bounds
            margin = max(uncertainties)
            lower_bound = pred_value - margin
            upper_bound = pred_value + margin
            
            logger.debug(f"Enhanced LSTM prediction: {pred_value:.2f} [{lower_bound:.2f}, {upper_bound:.2f}]")
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