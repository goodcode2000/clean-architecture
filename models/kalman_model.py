"""Simple 1D Kalman filter predictor for BTC price smoothing and short-term prediction.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from loguru import logger
import joblib
import os

# Add parent directory to path for imports if necessary
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config


class KalmanPredictor:
    """A compact Kalman filter implementation for 1D price prediction.

    State: price (we use a very simple constant model). The filter learns
    process and measurement variances from training data and maintains the
    last state and covariance to provide one-step-ahead predictions.
    """

    def __init__(self):
        self.is_trained = False
        self.last_state = None  # x (price estimate)
        self.last_cov = None    # P (estimate covariance)
        self.process_var = 1e-3  # Q
        self.measure_var = 1.0   # R
        self.last_training_data = None

    def _run_filter(self, prices: np.ndarray):
        """Run Kalman filter over a sequence of prices and return final x,P."""
        # Initialize
        x = float(prices[0])
        P = np.var(prices) if len(prices) > 1 else 1.0

        Q = float(self.process_var)
        R = float(self.measure_var)

        for z in prices[1:]:
            # Predict step (identity state transition)
            x_pred = x
            P_pred = P + Q

            # Update step
            K = P_pred / (P_pred + R)  # Kalman gain
            x = x_pred + K * (float(z) - x_pred)
            P = (1 - K) * P_pred

        return x, P

    def train(self, df: pd.DataFrame) -> bool:
        """Train the Kalman filter by estimating noise variances from data.

        Args:
            df: DataFrame with a 'close' column of prices

        Returns:
            True if training succeeded
        """
        try:
            if df is None or 'close' not in df.columns or len(df) < 5:
                logger.error("Insufficient data for Kalman training")
                return False

            prices = df['close'].dropna().astype(float).values
            if len(prices) < 5:
                logger.error("Not enough numeric price points for Kalman training")
                return False

            # Estimate measurement noise as variance of small-scale residuals
            diffs = np.diff(prices)
            self.process_var = max(np.var(diffs) * 0.01, 1e-6)

            # Measurement noise estimate from short-term volatility
            self.measure_var = max(np.var(prices - np.mean(prices)), 1e-2)

            # Run filter to initialize last state and covariance
            x, P = self._run_filter(prices)

            self.last_state = x
            self.last_cov = P
            self.last_training_data = prices[-2000:]
            self.is_trained = True

            logger.info(f"Kalman trained: last_state={self.last_state:.4f}, P={self.last_cov:.6f}, Q={self.process_var:.6f}, R={self.measure_var:.6f}")
            return True

        except Exception as e:
            logger.error(f"Kalman training failed: {e}")
            return False

    def predict(self, df: pd.DataFrame = None) -> Tuple[float, Tuple[float, float]]:
        """Provide a one-step-ahead prediction.

        If a DataFrame is provided, the filter will incorporate its last
        observations before predicting; otherwise it uses the stored state.
        """
        try:
            if not self.is_trained:
                logger.error("Kalman model not trained")
                return 0.0, (0.0, 0.0)

            if df is not None and 'close' in df.columns and len(df) > 0:
                prices = df['close'].dropna().astype(float).values
                # Update internal state using provided prices
                x, P = self._run_filter(np.concatenate([self.last_training_data[-50:] if self.last_training_data is not None else prices, prices]))
                self.last_state = x
                self.last_cov = P

            # One-step predict: with identity model x_pred = x
            x_pred = float(self.last_state)
            P_pred = float(self.last_cov + self.process_var)

            std = np.sqrt(P_pred + self.measure_var)
            lower = x_pred - 1.96 * std
            upper = x_pred + 1.96 * std

            logger.debug(f"Kalman prediction: {x_pred:.4f} [{lower:.4f}, {upper:.4f}]")
            return x_pred, (lower, upper)

        except Exception as e:
            logger.error(f"Kalman prediction failed: {e}")
            return 0.0, (0.0, 0.0)

    def update_model(self, new_data: pd.DataFrame) -> bool:
        """Incrementally update the filter with new observations."""
        try:
            if not self.is_trained:
                return self.train(new_data)

            if new_data is None or 'close' not in new_data.columns:
                logger.warning("No new data for Kalman update")
                return True

            new_prices = new_data['close'].dropna().astype(float).values
            if len(new_prices) == 0:
                return True

            # Run filter on combined recent history
            history = np.concatenate([self.last_training_data[-500:] if self.last_training_data is not None else new_prices, new_prices])
            x, P = self._run_filter(history)

            self.last_state = x
            self.last_cov = P
            self.last_training_data = history[-2000:]

            logger.debug(f"Kalman updated: last_state={self.last_state:.4f}, P={self.last_cov:.6f}")
            return True

        except Exception as e:
            logger.error(f"Kalman update failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_type': 'Kalman',
            'is_trained': self.is_trained,
            'last_state': float(self.last_state) if self.last_state is not None else None,
            'last_cov': float(self.last_cov) if self.last_cov is not None else None,
            'process_var': float(self.process_var),
            'measure_var': float(self.measure_var)
        }

    def save_model(self, filepath: str) -> bool:
        try:
            model_data = {
                'is_trained': self.is_trained,
                'last_state': self.last_state,
                'last_cov': self.last_cov,
                'process_var': self.process_var,
                'measure_var': self.measure_var,
                'last_training_data': self.last_training_data
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Kalman model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save Kalman model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        try:
            if not os.path.exists(filepath):
                logger.error(f"Kalman model file not found: {filepath}")
                return False
            model_data = joblib.load(filepath)
            self.is_trained = model_data.get('is_trained', False)
            self.last_state = model_data.get('last_state')
            self.last_cov = model_data.get('last_cov')
            self.process_var = model_data.get('process_var', self.process_var)
            self.measure_var = model_data.get('measure_var', self.measure_var)
            self.last_training_data = model_data.get('last_training_data')
            logger.info(f"Kalman model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load Kalman model: {e}")
            return False
