"""Simplified Temporal Fusion Transformer (TFT)-like model.

This is a lightweight, pragmatic implementation that captures
key TFT ideas (variable selection + temporal attention) without
pulling in a heavy external dependency. It's intended as a
drop-in experimental model; for production-grade TFT use a
well-tested library or a full implementation.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Any, Dict
from loguru import logger
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, LayerNormalization, MultiHeadAttention, Concatenate


class TFTPredictor:
    """Simplified TFT-style predictor.

    Accepts sequences of features (sequence_length, n_features) and
    produces a single-step forecast. Not a full TFT implementation,
    but contains encoder LSTM + multi-head attention to capture
    temporal and cross-feature relationships.
    """

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.sequence_length = None
        self.n_features = None

    def build_model(self, sequence_length: int, n_features: int) -> tf.keras.Model:
        inp = Input(shape=(sequence_length, n_features), name='observed_input')

        # Encoder
        x = LSTM(128, return_sequences=True)(inp)
        x = Dropout(0.2)(x)
        x = LayerNormalization()(x)

        # Attention
        attn = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        attn = LayerNormalization()(attn)

        # Pool and dense
        pooled = tf.reduce_mean(attn, axis=1)
        out = Dense(64, activation='relu')(pooled)
        out = Dropout(0.2)(out)
        out = Dense(1, name='price_output')(out)

        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return model

    def prepare_sequences(self, df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        # Reuse feature engineering pathways externally; here we expect df to be feature-engineered
        feature_columns = [c for c in df.columns if c not in ['timestamp']]
        X = []
        y = []
        arr = df[feature_columns].values
        for i in range(sequence_length, len(arr)):
            X.append(arr[i-sequence_length:i])
            y.append(df['close'].values[i])

        return np.array(X), np.array(y)

    def train(self, df: pd.DataFrame, sequence_length: int = 60, epochs: int = 20, batch_size: int = 64) -> bool:
        try:
            X, y = self.prepare_sequences(df, sequence_length=sequence_length)
            if len(X) == 0:
                logger.error('No sequences for TFT training')
                return False

            self.sequence_length = sequence_length
            self.n_features = X.shape[2]

            self.model = self.build_model(self.sequence_length, self.n_features)

            self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
            self.is_trained = True
            logger.info('TFT-like model trained')
            return True
        except Exception as e:
            logger.error(f'TFT training failed: {e}')
            return False

    def predict(self, df: pd.DataFrame) -> Tuple[float, Tuple[float, float]]:
        try:
            if not self.is_trained:
                logger.error('TFT model not trained')
                return 0.0, (0.0, 0.0)

            X, _ = self.prepare_sequences(df, sequence_length=self.sequence_length)
            if len(X) == 0:
                return 0.0, (0.0, 0.0)

            pred = self.model.predict(X[-1:], verbose=0)[0][0]
            margin = abs(pred) * 0.05
            return float(pred), (pred - margin, pred + margin)

        except Exception as e:
            logger.error(f'TFT prediction failed: {e}')
            return 0.0, (0.0, 0.0)

    def update_model(self, new_data: pd.DataFrame) -> bool:
        # For simplicity retrain on combined data
        try:
            return self.train(new_data, sequence_length=self.sequence_length, epochs=5)
        except Exception as e:
            logger.error(f'TFT update failed: {e}')
            return False

    def save_model(self, filepath: str) -> bool:
        try:
            if not self.is_trained:
                return False
            self.model.save(filepath.replace('.joblib', '.tf'))
            return True
        except Exception as e:
            logger.error(f'Failed to save TFT model: {e}')
            return False

    def load_model(self, filepath: str) -> bool:
        try:
            self.model = tf.keras.models.load_model(filepath.replace('.joblib', '.tf'))
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f'Failed to load TFT model: {e}')
            return False

    def get_model_info(self) -> Dict[str, Any]:
        return {'model_type': 'TFT-like', 'is_trained': self.is_trained}
