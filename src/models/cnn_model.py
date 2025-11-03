"""
CNN Model for BTC Price Prediction
Extracts local temporal patterns, edges, and features from input series
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import Flatten, Concatenate, GlobalAveragePooling1D
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

class CNNPredictionModel(BaseModel):
    """CNN model for extracting local temporal patterns and features"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            # Model architecture
            'sequence_length': 120,  # 120 time steps (10 hours for 5-min data)
            'conv_layers': [
                {'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
                {'filters': 128, 'kernel_size': 3, 'activation': 'relu'},
                {'filters': 64, 'kernel_size': 5, 'activation': 'relu'},
                {'filters': 32, 'kernel_size': 7, 'activation': 'relu'}
            ],
            'pooling_size': 2,
            'dense_units': [128, 64, 32],
            'dropout_rate': 0.3,
            'use_batch_norm': True,
            'use_global_pooling': True,  # Use both max and average pooling
            'activation': 'relu',
            
            # Multi-scale convolution
            'multi_scale': True,
            'kernel_sizes': [3, 5, 7, 11],  # Different kernel sizes for multi-scale
            
            # Training parameters
            'epochs': 150,
            'batch_size': 64,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'early_stopping_patience': 15,
            'reduce_lr_patience': 8,
            'reduce_lr_factor': 0.7,
            'min_lr': 1e-6,
            
            # Regularization
            'l1_reg': 0.001,
            'l2_reg': 0.001,
            
            # Data parameters
            'min_periods': 300,
            'feature_scaling': True,
            'target_scaling': True,
            'data_augmentation': True,
            
            # Pattern detection
            'pattern_analysis': True,
            'edge_detection': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__("CNN", default_config)
        
        # CNN-specific attributes
        self.model = None
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.training_history = []
        self.sequence_data = None
        self.pattern_filters = None
        
    async def train(self, features: List[FeatureSet], targets: List[float]) -> bool:
        """Train CNN model for pattern extraction"""
        try:
            if len(targets) < self.config['min_periods']:
                logger.warning(f"Insufficient data for CNN training: {len(targets)} < {self.config['min_periods']}")
                return False
            
            logger.info(f"Training CNN model with {len(targets)} data points")
            
            # Prepare sequence data
            X, y = self._prepare_sequence_data(features, targets)
            
            if X.shape[0] < self.config['sequence_length']:
                logger.warning("Insufficient data for sequence creation")
                return False
            
            # Data augmentation
            if self.config['data_augmentation']:
                X, y = self._augment_data(X, y)
            
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
                shuffle=True,  # CNN can handle shuffled data
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
            
            # Extract learned patterns
            if self.config['pattern_analysis']:
                self._extract_learned_patterns(X[:100])  # Use subset for analysis
            
            # Calculate feature importance
            self._calculate_feature_importance(X, y)
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            logger.info(f"CNN model trained successfully - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"CNN training failed: {e}")
            return False
    
    async def predict(self, features: FeatureSet) -> Tuple[float, float]:
        """Make prediction using trained CNN model"""
        try:
            if not self.is_trained or not self.model:
                logger.warning("CNN model not trained")
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
            
            logger.debug(f"CNN prediction: {prediction:.2f}, confidence: {confidence:.3f}")
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"CNN prediction failed: {e}")
            return 0.0, 0.0
    
    def _prepare_sequence_data(self, features: List[FeatureSet], targets: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequence data for CNN training"""
        try:
            # Convert features to DataFrame
            feature_dicts = [fs.to_dict() for fs in features]
            df = pd.DataFrame(feature_dicts)
            
            # Add targets
            df['target'] = targets
            
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Select numeric features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'target' in numeric_cols:
                numeric_cols.remove('target')
            
            # Limit features for CNN (focus on most relevant)
            if len(numeric_cols) > 30:
                # Select features based on correlation and variance
                correlations = df[numeric_cols].corrwith(df['target']).abs()
                variances = df[numeric_cols].var()
                
                # Combine correlation and variance scores
                scores = correlations * variances
                top_features = scores.nlargest(30).index.tolist()
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
            
            logger.info(f"Created CNN sequences: {X.shape[0]} samples, {X.shape[1]} timesteps, {X.shape[2]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"CNN sequence data preparation failed: {e}")
            return np.array([]), np.array([])
    
    def _prepare_prediction_sequence(self, features: FeatureSet) -> Optional[np.ndarray]:
        """Prepare sequence for CNN prediction"""
        try:
            if not self.sequence_data:
                logger.warning("No sequence data available for prediction")
                return None
            
            # Create feature vector
            feature_dict = features.to_dict()
            feature_vector = []
            
            for col in self.sequence_data['feature_columns']:
                value = feature_dict.get(col, 0.0)
                feature_vector.append(value)
            
            feature_vector = np.array(feature_vector)
            
            # Scale features
            if self.config['feature_scaling']:
                feature_vector = self.feature_scaler.transform(feature_vector.reshape(1, -1)).flatten()
            
            # Create sequence (simplified - in practice use actual historical data)
            sequence = np.tile(feature_vector, (self.config['sequence_length'], 1))
            
            # Add some variation to simulate temporal patterns
            for i in range(1, self.config['sequence_length']):
                noise = np.random.normal(0, 0.01, len(feature_vector))
                sequence[i] += noise * (i / self.config['sequence_length'])
            
            # Reshape for model input
            X_pred = sequence.reshape(1, self.config['sequence_length'], len(feature_vector))
            
            return X_pred
            
        except Exception as e:
            logger.error(f"CNN prediction sequence preparation failed: {e}")
            return None
    
    def _build_model(self, timesteps: int, n_features: int) -> Model:
        """Build CNN model with multi-scale convolutions"""
        try:
            if self.config['multi_scale']:
                return self._build_multiscale_model(timesteps, n_features)
            else:
                return self._build_simple_model(timesteps, n_features)
                
        except Exception as e:
            logger.error(f"CNN model building failed: {e}")
            raise
    
    def _build_simple_model(self, timesteps: int, n_features: int) -> Model:
        """Build simple CNN model"""
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(timesteps, n_features)))
        
        # Convolutional layers
        for i, conv_config in enumerate(self.config['conv_layers']):
            model.add(Conv1D(
                filters=conv_config['filters'],
                kernel_size=conv_config['kernel_size'],
                activation=conv_config['activation'],
                padding='same',
                kernel_regularizer=l1_l2(self.config['l1_reg'], self.config['l2_reg'])
            ))
            
            if self.config['use_batch_norm']:
                model.add(BatchNormalization())
            
            model.add(MaxPooling1D(pool_size=self.config['pooling_size']))
            model.add(Dropout(self.config['dropout_rate']))
        
        # Global pooling
        if self.config['use_global_pooling']:
            model.add(GlobalMaxPooling1D())
        else:
            model.add(Flatten())
        
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
        
        return model
    
    def _build_multiscale_model(self, timesteps: int, n_features: int) -> Model:
        """Build multi-scale CNN model with different kernel sizes"""
        # Input
        input_layer = Input(shape=(timesteps, n_features))
        
        # Multi-scale convolution branches
        conv_branches = []
        
        for kernel_size in self.config['kernel_sizes']:
            # Convolution branch
            x = Conv1D(
                filters=64,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                kernel_regularizer=l1_l2(self.config['l1_reg'], self.config['l2_reg'])
            )(input_layer)
            
            if self.config['use_batch_norm']:
                x = BatchNormalization()(x)
            
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(self.config['dropout_rate'])(x)
            
            # Second convolution
            x = Conv1D(
                filters=32,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                kernel_regularizer=l1_l2(self.config['l1_reg'], self.config['l2_reg'])
            )(x)
            
            if self.config['use_batch_norm']:
                x = BatchNormalization()(x)
            
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(self.config['dropout_rate'])(x)
            
            # Global pooling
            x = GlobalMaxPooling1D()(x)
            
            conv_branches.append(x)
        
        # Concatenate all branches
        if len(conv_branches) > 1:
            merged = Concatenate()(conv_branches)
        else:
            merged = conv_branches[0]
        
        # Dense layers
        x = merged
        for units in self.config['dense_units']:
            x = Dense(
                units,
                activation=self.config['activation'],
                kernel_regularizer=l1_l2(self.config['l1_reg'], self.config['l2_reg'])
            )(x)
            x = Dropout(self.config['dropout_rate'])(x)
        
        # Output
        output = Dense(1, activation='linear')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output)
        
        # Compile
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Built multi-scale CNN with {model.count_params()} parameters")
        return model
    
    def _augment_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation techniques"""
        try:
            augmented_X = [X]
            augmented_y = [y]
            
            # Add noise
            noise_factor = 0.01
            X_noise = X + np.random.normal(0, noise_factor, X.shape)
            augmented_X.append(X_noise)
            augmented_y.append(y)
            
            # Time warping (simple scaling)
            scale_factors = [0.95, 1.05]
            for scale in scale_factors:
                X_scaled = X * scale
                augmented_X.append(X_scaled)
                augmented_y.append(y)
            
            # Combine all augmented data
            X_aug = np.concatenate(augmented_X, axis=0)
            y_aug = np.concatenate(augmented_y, axis=0)
            
            logger.info(f"Data augmentation: {X.shape[0]} -> {X_aug.shape[0]} samples")
            return X_aug, y_aug
            
        except Exception as e:
            logger.warning(f"Data augmentation failed: {e}")
            return X, y
    
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
    
    def _extract_learned_patterns(self, X_sample: np.ndarray):
        """Extract learned patterns from CNN filters"""
        try:
            if not self.model:
                return
            
            # Get first convolutional layer
            conv_layer = None
            for layer in self.model.layers:
                if isinstance(layer, Conv1D):
                    conv_layer = layer
                    break
            
            if conv_layer is None:
                return
            
            # Get filter weights
            filters = conv_layer.get_weights()[0]  # Shape: (kernel_size, input_dim, num_filters)
            
            # Store pattern information
            self.pattern_filters = {
                'filter_weights': filters,
                'num_filters': filters.shape[2],
                'kernel_size': filters.shape[0],
                'input_features': filters.shape[1]
            }
            
            logger.info(f"Extracted {filters.shape[2]} learned patterns")
            
        except Exception as e:
            logger.warning(f"Pattern extraction failed: {e}")
    
    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """Calculate feature importance using gradients"""
        try:
            if not self.sequence_data:
                return
            
            # Sample subset for gradient calculation
            sample_size = min(50, X.shape[0])
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
            
            # Calculate gradients
            with tf.GradientTape() as tape:
                X_tensor = tf.Variable(X_sample, dtype=tf.float32)
                predictions = self.model(X_tensor)
                loss = tf.reduce_mean(tf.square(predictions))
            
            gradients = tape.gradient(loss, X_tensor)
            
            # Calculate feature importance
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
        """Calculate prediction confidence"""
        try:
            # Use model's internal confidence estimation
            # This is a simplified version - could be enhanced with ensemble methods
            
            # Base confidence from training performance
            confidence = 0.6
            
            # Adjust based on training history
            if self.training_history:
                latest_metrics = self.training_history[-1]['metrics']
                if 'final_val_loss' in latest_metrics and latest_metrics['final_val_loss']:
                    val_loss = latest_metrics['final_val_loss']
                    # Lower validation loss = higher confidence
                    confidence += max(0, 0.3 - val_loss / 1000.0)
            
            # Adjust based on pattern recognition
            if self.pattern_filters:
                # Higher confidence if more patterns were learned
                num_patterns = self.pattern_filters['num_filters']
                confidence += min(0.1, num_patterns / 1000.0)
            
            return max(0.1, min(0.9, confidence))
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get CNN model information"""
        info = {
            'model_name': self.model_name,
            'model_type': 'Multi-scale CNN for Pattern Extraction',
            'is_trained': self.is_trained,
            'config': self.config.copy(),
            'feature_importance': dict(list(self.feature_importance.items())[:15]) if self.feature_importance else {},
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None
        }
        
        if self.model:
            info['model_params'] = self.model.count_params()
            info['model_layers'] = len(self.model.layers)
        
        if self.sequence_data:
            info['sequence_info'] = self.sequence_data
        
        if self.pattern_filters:
            info['pattern_info'] = {
                'num_patterns': self.pattern_filters['num_filters'],
                'kernel_size': self.pattern_filters['kernel_size'],
                'input_features': self.pattern_filters['input_features']
            }
        
        if self.training_history:
            latest_metrics = self.training_history[-1]['metrics']
            info['latest_metrics'] = latest_metrics
        
        return info
    
    def detect_patterns(self, X: np.ndarray) -> Dict[str, Any]:
        """Detect patterns in input sequences"""
        try:
            if not self.model or not self.pattern_filters:
                return {}
            
            # Get activations from first conv layer
            conv_layer_model = Model(
                inputs=self.model.input,
                outputs=self.model.layers[1].output  # Assuming first conv layer is at index 1
            )
            
            activations = conv_layer_model.predict(X, verbose=0)
            
            # Analyze activations
            pattern_analysis = {
                'max_activations': np.max(activations, axis=(0, 1)).tolist(),
                'mean_activations': np.mean(activations, axis=(0, 1)).tolist(),
                'activation_patterns': activations.shape,
                'dominant_patterns': np.argmax(np.max(activations, axis=(0, 1)))
            }
            
            return pattern_analysis
            
        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}")
            return {}
    
    def get_learned_filters(self) -> Optional[Dict[str, Any]]:
        """Get learned convolutional filters"""
        return self.pattern_filters