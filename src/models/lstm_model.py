"""
Attention-based BiLSTM Model for BTC Price Prediction
Captures long-range temporal dependencies and nonlinear sequences
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Attention, Input, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .base_model import BaseModel
from ..feature_engineering import FeatureSet

logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

class AttentionLayer(Layer):
    """Custom attention layer for LSTM"""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # Calculate attention scores
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        
        # Apply attention weights
        output = x * a
        
        return tf.reduce_sum(output, axis=1)

class LSTMPredictionModel(BaseModel):
    """Attention-based BiLSTM model for temporal sequence prediction"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            # Model architecture
            'sequence_length': 60,  # 60 time steps (5 hours for 5-min data)
            'lstm_units': [128, 64],  # LSTM layer sizes
            'dense_units': [32, 16],  # Dense layer sizes
            'dropout_rate': 0.2,
            'recurrent_dropout': 0.2,
            'use_bidirectional': True,
            'use_attention': True,
            'activation': 'relu',
            
            # Training parameters
            'epochs': 200,
            'batch_size': 32,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'early_stopping_patience': 20,
            'reduce_lr_patience': 10,
            'reduce_lr_factor': 0.5,
            'min_lr': 1e-7,
            
            # Regularization
            'l1_reg': 0.01,
            'l2_reg': 0.01,
            'use_batch_norm': False,
            
            # Data parameters
            'min_periods': 200,
            'feature_scaling': True,
            'target_scaling': True,
            'shuffle_training': False,  # Keep temporal order
            
            # Prediction parameters
            'prediction_intervals': True,
            'monte_carlo_samples': 100
        }
        
        if config:
            default_config.update(config)
        
        super().__init__("LSTM", default_config)
        
        # LSTM-specific attributes
        self.model = None
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.training_history = []
        self.sequence_data = None
        
    async def train(self, features: List[FeatureSet], targets: List[float]) -> bool:
        """Train attention-based BiLSTM model"""
        try:
            if len(targets) < self.config['min_periods']:
                logger.warning(f"Insufficient data for LSTM training: {len(targets)} < {self.config['min_periods']}")
                return False
            
            logger.info(f"Training LSTM model with {len(targets)} data points")
            
            # Prepare sequence data
            X, y = self._prepare_sequence_data(features, targets)
            
            if X.shape[0] < self.config['sequence_length']:
                logger.warning("Insufficient data for sequence creation")
                return False
            
            # Build model
            self.model = self._build_model(X.shape[1], X.shape[2])
            
            # Prepare callbacks
            callbacks = self._prepare_callbacks()
            
            # Train model
            history = self.model.fit(
                X, y,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                validation_split=self.config['validation_split'],
                callbacks=callbacks,
                shuffle=self.config['shuffle_training'],
                verbose=0
            )
            
            # Store training history
            self.training_history = history.history
            
            # Calculate performance metrics
            train_pred = self.model.predict(X, verbose=0)
            
            # Inverse transform predictions and targets for evaluation
            if self.config['target_scaling']:
                train_pred_inv = self.target_scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
                y_inv = self.target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
            else:
                train_pred_inv = train_pred.flatten()
                y_inv = y.flatten()
            
            mae = mean_absolute_error(y_inv, train_pred_inv)
            rmse = np.sqrt(mean_squared_error(y_inv, train_pred_inv))
            mape = np.mean(np.abs((y_inv - train_pred_inv) / (y_inv + 1e-10))) * 100
            
            # Update training metrics
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history else None,
                'epochs_trained': len(history.history['loss']),
                'sequence_length': self.config['sequence_length'],
                'n_features': X.shape[2]
            }
            
            self.update_training_history(metrics)
            
            # Calculate feature importance (approximate using gradients)
            self._calculate_feature_importance(X, y)
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            logger.info(f"LSTM model trained successfully - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            return False
    
    async def predict(self, features: FeatureSet) -> Tuple[float, float]:
        """Make prediction using trained LSTM model"""
        try:
            if not self.is_trained or not self.model:
                logger.warning("LSTM model not trained")
                return 0.0, 0.0
            
            # Prepare sequence for prediction
            X_pred = self._prepare_prediction_sequence(features)
            
            if X_pred is None or X_pred.shape[0] == 0:
                logger.warning("Failed to prepare prediction sequence")
                return 0.0, 0.0
            
            # Make prediction
            pred_scaled = self.model.predict(X_pred, verbose=0)
            
            # Inverse transform prediction
            if self.config['target_scaling']:
                prediction = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
            else:
                prediction = pred_scaled[0, 0]
            
            # Calculate prediction confidence
            confidence = self._calculate_prediction_confidence(X_pred, prediction)
            
            # Ensure prediction is positive
            prediction = max(prediction, 0.01)
            
            logger.debug(f"LSTM prediction: {prediction:.2f}, confidence: {confidence:.3f}")
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return 0.0, 0.0
    
    def _prepare_sequence_data(self, features: List[FeatureSet], targets: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequence data for LSTM training"""
        try:
            # Convert features to DataFrame
            feature_dicts = [fs.to_dict() for fs in features]
            df = pd.DataFrame(feature_dicts)
            
            # Add targets
            df['target'] = targets
            
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Select features (exclude non-numeric or problematic features)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'target' in numeric_cols:
                numeric_cols.remove('target')
            
            # Limit number of features to prevent overfitting
            if len(numeric_cols) > 50:
                # Select top features based on correlation with target
                correlations = df[numeric_cols].corrwith(df['target']).abs()
                top_features = correlations.nlargest(50).index.tolist()
                numeric_cols = top_features
            
            feature_data = df[numeric_cols].values
            target_data = df['target'].values
            
            # Scale features
            if self.config['feature_scaling']:
                feature_data = self.feature_scaler.fit_transform(feature_data)
            
            # Scale targets
            if self.config['target_scaling']:
                target_data = self.target_scaler.fit_transform(target_data.reshape(-1, 1)).flatten()
            
            # Create sequences
            X, y = [], []
            seq_len = self.config['sequence_length']
            
            for i in range(seq_len, len(feature_data)):
                X.append(feature_data[i-seq_len:i])
                y.append(target_data[i])
            
            X = np.array(X)
            y = np.array(y)
            
            # Store sequence info
            self.sequence_data = {
                'feature_columns': numeric_cols,
                'n_features': len(numeric_cols),
                'sequence_length': seq_len
            }
            
            logger.info(f"Created sequences: {X.shape[0]} samples, {X.shape[1]} timesteps, {X.shape[2]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Sequence data preparation failed: {e}")
            return np.array([]), np.array([])
    
    def _prepare_prediction_sequence(self, features: FeatureSet) -> Optional[np.ndarray]:
        """Prepare sequence for prediction"""
        try:
            if not self.sequence_data:
                logger.warning("No sequence data available for prediction")
                return None
            
            # This is a simplified version - in practice, you'd need to maintain
            # a rolling window of recent features
            feature_dict = features.to_dict()
            
            # Create feature vector matching training features
            feature_vector = []
            for col in self.sequence_data['feature_columns']:
                value = feature_dict.get(col, 0.0)
                feature_vector.append(value)
            
            feature_vector = np.array(feature_vector)
            
            # Scale features
            if self.config['feature_scaling']:
                feature_vector = self.feature_scaler.transform(feature_vector.reshape(1, -1)).flatten()
            
            # Create sequence by repeating the current features
            # In practice, you'd use actual historical sequence
            sequence = np.tile(feature_vector, (self.config['sequence_length'], 1))
            
            # Reshape for model input
            X_pred = sequence.reshape(1, self.config['sequence_length'], len(feature_vector))
            
            return X_pred
            
        except Exception as e:
            logger.error(f"Prediction sequence preparation failed: {e}")
            return None
    
    def _build_model(self, timesteps: int, n_features: int) -> Model:
        """Build attention-based BiLSTM model"""
        try:
            model = Sequential()
            
            # Input layer
            model.add(Input(shape=(timesteps, n_features)))
            
            # LSTM layers
            for i, units in enumerate(self.config['lstm_units']):
                return_sequences = (i < len(self.config['lstm_units']) - 1) or self.config['use_attention']
                
                if self.config['use_bidirectional']:
                    lstm_layer = Bidirectional(
                        LSTM(
                            units,
                            return_sequences=return_sequences,
                            dropout=self.config['dropout_rate'],
                            recurrent_dropout=self.config['recurrent_dropout'],
                            kernel_regularizer=l1_l2(self.config['l1_reg'], self.config['l2_reg'])
                        )
                    )
                else:
                    lstm_layer = LSTM(
                        units,
                        return_sequences=return_sequences,
                        dropout=self.config['dropout_rate'],
                        recurrent_dropout=self.config['recurrent_dropout'],
                        kernel_regularizer=l1_l2(self.config['l1_reg'], self.config['l2_reg'])
                    )
                
                model.add(lstm_layer)
            
            # Attention layer
            if self.config['use_attention']:
                model.add(AttentionLayer())
            
            # Dense layers
            for units in self.config['dense_units']:
                model.add(Dense(
                    units,
                    activation=self.config['activation'],
                    kernel_regularizer=l1_l2(self.config['l1_reg'], self.config['l2_reg'])
                ))
                model.add(Dropout(self.config['dropout_rate']))
            
            # Output layer
            model.add(Dense(1, activation='linear'))
            
            # Compile model
            optimizer = Adam(learning_rate=self.config['learning_rate'])
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            
            logger.info(f"Built LSTM model with {model.count_params()} parameters")
            return model
            
        except Exception as e:
            logger.error(f"Model building failed: {e}")
            raise
    
    def _prepare_callbacks(self) -> List:
        """Prepare training callbacks"""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=0
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config['reduce_lr_factor'],
            patience=self.config['reduce_lr_patience'],
            min_lr=self.config['min_lr'],
            verbose=0
        )
        callbacks.append(reduce_lr)
        
        return callbacks
    
    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """Calculate approximate feature importance using gradients"""
        try:
            if not self.sequence_data:
                return
            
            # Sample a subset for gradient calculation
            sample_size = min(100, X.shape[0])
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
            
            # Calculate gradients
            with tf.GradientTape() as tape:
                X_tensor = tf.Variable(X_sample, dtype=tf.float32)
                predictions = self.model(X_tensor)
                loss = tf.reduce_mean(tf.square(predictions))
            
            gradients = tape.gradient(loss, X_tensor)
            
            # Calculate feature importance as mean absolute gradient
            importance = np.mean(np.abs(gradients.numpy()), axis=(0, 1))
            
            # Create feature importance dictionary
            self.feature_importance = {}
            for i, feature_name in enumerate(self.sequence_data['feature_columns']):
                self.feature_importance[feature_name] = float(importance[i])
            
            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
    
    def _calculate_prediction_confidence(self, X_pred: np.ndarray, prediction: float) -> float:
        """Calculate prediction confidence using Monte Carlo dropout"""
        try:
            if not self.config['prediction_intervals']:
                return self.calculate_prediction_confidence(prediction, None)
            
            # Enable dropout during prediction for uncertainty estimation
            predictions = []
            
            for _ in range(self.config['monte_carlo_samples']):
                # Make prediction with dropout enabled
                pred = self.model(X_pred, training=True)
                
                if self.config['target_scaling']:
                    pred_inv = self.target_scaler.inverse_transform(pred.numpy().reshape(-1, 1))[0, 0]
                else:
                    pred_inv = pred.numpy()[0, 0]
                
                predictions.append(pred_inv)
            
            predictions = np.array(predictions)
            
            # Calculate uncertainty
            pred_std = np.std(predictions)
            pred_mean = np.mean(predictions)
            
            # Convert to confidence (lower uncertainty = higher confidence)
            cv = pred_std / (abs(pred_mean) + 1e-10)
            confidence = max(0.1, min(0.9, 1.0 - min(cv, 1.0)))
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get LSTM model information"""
        info = {
            'model_name': self.model_name,
            'model_type': 'Attention-based Bidirectional LSTM',
            'is_trained': self.is_trained,
            'config': self.config.copy(),
            'feature_importance': dict(list(self.feature_importance.items())[:20]) if self.feature_importance else {},
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None
        }
        
        if self.model:
            info['model_params'] = self.model.count_params()
            info['model_layers'] = len(self.model.layers)
        
        if self.sequence_data:
            info['sequence_info'] = self.sequence_data
        
        if self.training_history:
            latest_metrics = self.training_history[-1]['metrics']
            info['latest_metrics'] = latest_metrics
            
            # Add training curve info
            if 'loss' in self.training_history:
                info['training_curve'] = {
                    'final_loss': self.training_history['loss'][-1],
                    'final_val_loss': self.training_history['val_loss'][-1] if 'val_loss' in self.training_history else None,
                    'epochs': len(self.training_history['loss'])
                }
        
        return info
    
    def get_attention_weights(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Get attention weights for interpretability"""
        try:
            if not self.model or not self.config['use_attention']:
                return None
            
            # Create a model that outputs attention weights
            attention_layer = None
            for layer in self.model.layers:
                if isinstance(layer, AttentionLayer):
                    attention_layer = layer
                    break
            
            if attention_layer is None:
                return None
            
            # This would require modifying the model architecture to output attention weights
            # For now, return None as it requires more complex implementation
            return None
            
        except Exception as e:
            logger.warning(f"Attention weights extraction failed: {e}")
            return None
    
    def predict_with_uncertainty(self, features: FeatureSet, n_samples: int = 100) -> Dict[str, float]:
        """Make prediction with uncertainty quantification"""
        try:
            if not self.is_trained or not self.model:
                return {'prediction': 0.0, 'uncertainty': 0.0, 'confidence': 0.0}
            
            X_pred = self._prepare_prediction_sequence(features)
            if X_pred is None:
                return {'prediction': 0.0, 'uncertainty': 0.0, 'confidence': 0.0}
            
            # Monte Carlo predictions
            predictions = []
            for _ in range(n_samples):
                pred = self.model(X_pred, training=True)
                
                if self.config['target_scaling']:
                    pred_inv = self.target_scaler.inverse_transform(pred.numpy().reshape(-1, 1))[0, 0]
                else:
                    pred_inv = pred.numpy()[0, 0]
                
                predictions.append(max(pred_inv, 0.01))  # Ensure positive
            
            predictions = np.array(predictions)
            
            return {
                'prediction': float(np.mean(predictions)),
                'uncertainty': float(np.std(predictions)),
                'confidence': float(max(0.1, min(0.9, 1.0 - np.std(predictions) / (np.mean(predictions) + 1e-10)))),
                'prediction_interval': {
                    'lower': float(np.percentile(predictions, 5)),
                    'upper': float(np.percentile(predictions, 95))
                }
            }
            
        except Exception as e:
            logger.error(f"Uncertainty prediction failed: {e}")
            return {'prediction': 0.0, 'uncertainty': 0.0, 'confidence': 0.0}