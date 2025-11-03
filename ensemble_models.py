"""
Ensemble model components: ETS, GARCH, LightGBM, TCN, CNN
"""
import pandas as pd
import numpy as np
try:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
except ImportError:
    # Fallback for different statsmodels versions
    try:
        from statsmodels.tsa.ets.etsmodel import ETSModel
    except ImportError:
        # Use Holt-Winters as fallback (will need API adjustment)
        ETSModel = None
        print("Warning: ETSModel not available, ETS predictions will use simple fallback")
from arch import arch_model
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers
try:
    # Enable GPU memory growth to avoid OOM issues on TensorFlow
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    # Optional: enable mixed precision on GPUs (can speed up training)
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
    except Exception:
        pass
except Exception:
    pass
from sklearn.preprocessing import StandardScaler
import pickle
import os
from config import MODEL_DIR, RANDOM_STATE


class ETSModelWrapper:
    """ETS (Error, Trend, Seasonality) model wrapper"""
    
    def __init__(self):
        self.model = None
        self.last_fitted_data = None
        
    def fit(self, prices):
        """Fit ETS model to price series"""
        if ETSModel is None:
            self.last_fitted_data = prices
            return self
        
        try:
            # Use additive model for BTC (multiplicative can be unstable)
            self.model = ETSModel(
                prices,
                error='add',
                trend='add',
                seasonal=None,  # No seasonality for BTC
                damped_trend=True
            ).fit(disp=False)
            self.last_fitted_data = prices
            return self
        except Exception as e:
            print(f"ETS fitting error: {e}")
            try:
                # Fallback to simple exponential smoothing
                self.model = ETSModel(prices, error='add', trend=None).fit(disp=False)
                self.last_fitted_data = prices
            except:
                # Final fallback: just store last value
                self.last_fitted_data = prices
            return self
    
    def predict(self, steps=1):
        """Predict next steps"""
        if self.model is None:
            return np.array([self.last_fitted_data.iloc[-1] if self.last_fitted_data is not None else 0])
        
        forecast = self.model.forecast(steps=steps)
        return forecast if hasattr(forecast, '__len__') else np.array([forecast])
    
    def save(self, filepath):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'last_fitted': self.last_fitted_data}, f)
    
    def load(self, filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.last_fitted_data = data['last_fitted']


class GARCHModelWrapper:
    """GARCH model wrapper for volatility modeling"""
    
    def __init__(self):
        self.model = None
        self.last_price = None
        
    def fit(self, returns):
        """Fit GARCH model to returns"""
        try:
            # Remove NaN and inf
            returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
            if len(returns_clean) < 50:
                self.last_price = returns_clean.iloc[-1] if len(returns_clean) > 0 else 0
                return self
            
            # Fit GARCH(1,1) model
            self.model = arch_model(returns_clean * 100, vol='Garch', p=1, q=1, dist='normal')
            self.model = self.model.fit(disp='off', show_warning=False)
            return self
        except Exception as e:
            print(f"GARCH fitting error: {e}")
            self.last_price = returns.iloc[-1] if len(returns) > 0 else 0
            return self
    
    def predict(self, steps=1):
        """Predict volatility and use it for price prediction"""
        if self.model is None:
            return np.array([self.last_price] * steps) if self.last_price else np.zeros(steps)
        
        try:
            forecast = self.model.forecast(horizon=steps, reindex=False)
            # Use mean variance forecast
            vol_forecast = np.sqrt(forecast.variance.values[-1, :]) / 100
            # Simple prediction: assume returns follow volatility
            return vol_forecast
        except:
            return np.array([self.last_price] * steps) if self.last_price else np.zeros(steps)
    
    def save(self, filepath):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'last_price': self.last_price}, f)
    
    def load(self, filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.last_price = data['last_price']


class LightGBMModel:
    """LightGBM model with residual correction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        
    def fit(self, X, y, feature_cols):
        """Fit LightGBM model"""
        self.feature_cols = feature_cols
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create dataset
        train_data = lgb.Dataset(X_scaled, label=y)
        
        # Parameters optimized for BTC prediction
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': RANDOM_STATE,
            'max_depth': 7,
            'min_data_in_leaf': 20,
        }
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)]
        )
        
        return self
    
    def predict(self, X):
        """Predict using LightGBM"""
        if self.model is None:
            return np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, filepath):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_model(filepath.replace('.pkl', '.lgb'))
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_cols': self.feature_cols
            }, f)
    
    def load(self, filepath):
        """Load model"""
        self.model = lgb.Booster(model_file=filepath.replace('.pkl', '.lgb'))
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.feature_cols = data['feature_cols']


def create_tcn_model(input_shape, output_dim=1):
    """Create Temporal Convolutional Network model"""
    inputs = keras.Input(shape=input_shape)
    
    # TCN layers with dilated convolutions
    x = inputs
    num_filters = 64
    kernel_size = 3
    dilations = [1, 2, 4, 8, 16]
    
    for i, dilation in enumerate(dilations):
        # Residual block
        residual = x
        
        # First conv
        x = layers.Conv1D(num_filters, kernel_size, dilation_rate=dilation, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Second conv
        x = layers.Conv1D(num_filters, kernel_size, dilation_rate=dilation, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Residual connection
        if residual.shape[-1] != num_filters:
            residual = layers.Conv1D(num_filters, 1)(residual)
        x = layers.Add()([x, residual])
    
    # Global pooling and output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(output_dim)(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model


def create_cnn_model(input_shape, output_dim=1):
    """Create CNN model for pattern extraction"""
    inputs = keras.Input(shape=input_shape)
    
    # CNN layers
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(output_dim)(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model


class TCNModel:
    """TCN model wrapper"""
    
    def __init__(self, sequence_length=60):
        self.model = None
        self.scaler = StandardScaler()
        self.sequence_length = sequence_length
        
    def create_sequences(self, data, target_col='close'):
        """Create sequences for TCN"""
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(data)):
            seq = data.iloc[i-self.sequence_length:i][['close', 'volume', 'returns']].values
            target = data.iloc[i][target_col]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def fit(self, data, target_col='close'):
        """Fit TCN model"""
        sequences, targets = self.create_sequences(data, target_col)
        
        # Normalize sequences
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        sequences_normalized = self.scaler.fit_transform(sequences_reshaped)
        sequences_normalized = sequences_normalized.reshape(sequences.shape)
        
        # Create model
        input_shape = (self.sequence_length, sequences.shape[2])
        self.model = create_tcn_model(input_shape)
        
        # Train
        self.model.fit(
            sequences_normalized, targets,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        return self
    
    def predict(self, data):
        """Predict using TCN"""
        if self.model is None:
            return np.array([0])
        
        # Get last sequence
        if len(data) < self.sequence_length:
            return np.array([data['close'].iloc[-1]])
        
        seq = data.iloc[-self.sequence_length:][['close', 'volume', 'returns']].values
        seq_reshaped = seq.reshape(-1, seq.shape[-1])
        seq_normalized = self.scaler.transform(seq_reshaped)
        seq_normalized = seq_normalized.reshape(1, self.sequence_length, seq.shape[1])
        
        prediction = self.model.predict(seq_normalized, verbose=0)[0][0]
        return np.array([prediction])
    
    def save(self, filepath):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath.replace('.pkl', '.h5'))
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'sequence_length': self.sequence_length
            }, f)
    
    def load(self, filepath):
        """Load model"""
        self.model = keras.models.load_model(filepath.replace('.pkl', '.h5'))
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.sequence_length = data['sequence_length']


class CNNModel:
    """CNN model wrapper"""
    
    def __init__(self, sequence_length=30):
        self.model = None
        self.scaler = StandardScaler()
        self.sequence_length = sequence_length
        
    def create_sequences(self, data, target_col='close'):
        """Create sequences for CNN"""
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(data)):
            seq = data.iloc[i-self.sequence_length:i][['close', 'volume', 'returns']].values
            target = data.iloc[i][target_col]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def fit(self, data, target_col='close'):
        """Fit CNN model"""
        sequences, targets = self.create_sequences(data, target_col)
        
        # Normalize
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        sequences_normalized = self.scaler.fit_transform(sequences_reshaped)
        sequences_normalized = sequences_normalized.reshape(sequences.shape)
        
        # Create model
        input_shape = (self.sequence_length, sequences.shape[2])
        self.model = create_cnn_model(input_shape)
        
        # Train
        self.model.fit(
            sequences_normalized, targets,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        return self
    
    def predict(self, data):
        """Predict using CNN"""
        if self.model is None:
            return np.array([0])
        
        if len(data) < self.sequence_length:
            return np.array([data['close'].iloc[-1]])
        
        seq = data.iloc[-self.sequence_length:][['close', 'volume', 'returns']].values
        seq_reshaped = seq.reshape(-1, seq.shape[-1])
        seq_normalized = self.scaler.transform(seq_reshaped)
        seq_normalized = seq_normalized.reshape(1, self.sequence_length, seq.shape[1])
        
        prediction = self.model.predict(seq_normalized, verbose=0)[0][0]
        return np.array([prediction])
    
    def save(self, filepath):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath.replace('.pkl', '.h5'))
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'sequence_length': self.sequence_length
            }, f)
    
    def load(self, filepath):
        """Load model"""
        self.model = keras.models.load_model(filepath.replace('.pkl', '.h5'))
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.sequence_length = data['sequence_length']

