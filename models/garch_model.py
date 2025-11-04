"""GARCH model wrapper for volatility-aware BTC price forecasting.

This model uses the `arch` package when available to fit a GARCH(1,1)
on returns. As GARCH models volatility rather than price directly, the
predict method maps a short-term predicted return (using a simple mean
estimate) plus the forecast volatility into a price prediction so it
can participate in the ensemble.
"""
import os
import numpy as np
import pandas as pd
from loguru import logger
from typing import Tuple, Any, Dict
import joblib

try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False


class GARCHPredictor:
    """Simple GARCH(1,1) predictor wrapper.

    Notes:
    - If `arch` is not installed, falls back to a simple rolling-volatility
      estimator so the pipeline remains usable.
    - This predictor returns a price forecast (single-step) to be
      compatible with the ensemble interface.
    """

    def __init__(self):
        self.model = None
        self.fitted = None
        self.is_trained = False
        self.training_score = None

    def train(self, df: pd.DataFrame) -> bool:
        """Train GARCH model on returns.

        Args:
            df: DataFrame containing a 'close' column

        Returns:
            True if training succeeded
        """
        try:
            returns = df['close'].pct_change().dropna()

            if len(returns) < 50:
                logger.error("Insufficient data to train GARCH model")
                return False

            if HAS_ARCH:
                am = arch_model(returns * 100.0, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
                self.fitted = am.fit(disp='off')
                self.is_trained = True
                logger.info("GARCH model trained using arch package")
            else:
                # Fallback: store rolling std and mean
                self.fitted = {
                    'rolling_std': returns.rolling(window=20).std().iloc[-1],
                    'mean_return': returns.mean()
                }
                self.is_trained = True
                logger.info("GARCH fallback trained (rolling std/mean)")

            return True

        except Exception as e:
            logger.error(f"GARCH training failed: {e}")
            return False

    def predict(self, df: pd.DataFrame) -> Tuple[float, Tuple[float, float]]:
        """Predict next price using the GARCH/volatility forecast.

        Returns:
            (predicted_price, (lower, upper))
        """
        try:
            if not self.is_trained:
                logger.error("GARCH model not trained")
                return 0.0, (0.0, 0.0)

            last_close = df['close'].iloc[-1]
            returns = df['close'].pct_change().dropna()

            if HAS_ARCH and self.fitted is not None:
                # Forecast one-step ahead variance
                forecast = self.fitted.forecast(horizon=1, reindex=False)
                # arch returns variance in percent-squared if we scaled by 100 earlier
                var = float(forecast.variance.values[-1, 0]) / (100.0 ** 2)
                vol = np.sqrt(max(var, 0.0))
                # Use simple mean return as location estimate
                mu = returns.mean()
            else:
                mu = float(self.fitted.get('mean_return', returns.mean()))
                vol = float(self.fitted.get('rolling_std', returns.std()))

            # Predict price as last_close * (1 + mu)
            pred_return = mu
            pred_price = float(last_close * (1.0 + pred_return))

            # Build a confidence interval using volatility
            lower = float(pred_price * (1.0 - 1.96 * vol))
            upper = float(pred_price * (1.0 + 1.96 * vol))

            return pred_price, (lower, upper)

        except Exception as e:
            logger.error(f"GARCH prediction failed: {e}")
            return 0.0, (0.0, 0.0)

    def update_model(self, new_data: pd.DataFrame) -> bool:
        # For simplicity, retrain on the latest window
        try:
            return self.train(new_data)
        except Exception as e:
            logger.error(f"GARCH update failed: {e}")
            return False

    def save_model(self, filepath: str) -> bool:
        try:
            if not self.is_trained:
                return False
            joblib.dump(self.fitted, filepath)
            logger.info(f"GARCH model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save GARCH model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        try:
            if not os.path.exists(filepath):
                return False
            self.fitted = joblib.load(filepath)
            self.is_trained = True
            logger.info(f"GARCH model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load GARCH model: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_type': 'GARCH',
            'is_trained': self.is_trained
        }
