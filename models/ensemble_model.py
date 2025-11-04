"""Enhanced ensemble model with dynamic weighting based on performance and market conditions.

This ensemble uses an adaptive weighting system that adjusts model contributions
based on recent performance and market conditions. It includes volatility-aware
GARCH modeling and experimental models like TFT and RL agents.
"""
import os
import sys
from typing import Dict, Tuple, Any, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
import joblib

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from models.garch_model import GARCHPredictor
from models.svr_model import SVRPredictor
from models.random_forest_model import RandomForestPredictor
from models.lightgbm_model import LightGBMPredictor
from models.lstm_model import LSTMPredictor
from models.tft_model import TFTPredictor
from models.rl_agent import RLTradingAgent
from services.feature_engineering import FeatureEngineer
from services.preprocessing import DataPreprocessor
from services.weight_adjustment import DynamicWeightAdjuster


class EnsemblePredictor:
    """Enhanced ensemble model with dynamic weighting."""

    def __init__(self):
        # Initialize individual models
        self.models = {
            'garch': GARCHPredictor(),
            'svr': SVRPredictor(),
            'random_forest': RandomForestPredictor(),
            'lightgbm': LightGBMPredictor(),
            'lstm': LSTMPredictor(),
            'tft': TFTPredictor(),
            'rl_agent': RLTradingAgent()
        }

        # Initialize dynamic weight adjuster
        self.weight_adjuster = DynamicWeightAdjuster(
            base_weights=Config.ENSEMBLE_WEIGHTS,
            window_size=100,  # Consider last 100 predictions
            min_weight=0.05,  # Minimum 5% weight per model
            max_weight=0.40   # Maximum 40% weight per model
        )

        # Current weights start at base values
        self.weights = self.weight_adjuster.get_current_weights()

        # Performance tracking
        self.predictions_history: List[Dict[str, float]] = []
        self.actual_prices: List[float] = []
        self.volatility_history: List[float] = []
        
        # Feature engineering and preprocessing
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = DataPreprocessor()

        # Training state
        self.is_trained = False
        self.training_scores = {}
        self.model_contributions = {}
        self.last_training_data = None

        # Volatility detection
        self.volatility_threshold = 0.05  # 5% price change threshold
        self.rapid_movement_detected = False

    def prepare_data_for_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare and return a dictionary of data suitable for each model."""
        try:
            logger.info("Preparing data for ensemble models...")

            features_df = self.feature_engineer.create_all_features(df)
            if len(features_df) == 0:
                logger.error("Feature engineering failed")
                return {}

            clean_df = self.preprocessor.clean_data_for_training(features_df)

            prepared_data = {
                'raw_df': df,
                'features_df': features_df,
                'clean_df': clean_df,
                # Keep the key name `ets_data` for compatibility but it is used by GARCH now
                'ets_data': clean_df,
                'sklearn_data': clean_df,
                'lstm_data': clean_df
            }

            logger.info(f"Data prepared: {len(clean_df)} samples, {len(clean_df.columns)} features")
            return prepared_data

        except Exception as e:
            logger.error(f"Failed to prepare ensemble data: {e}")
            return {}

    def train_individual_models(self, prepared_data: Dict[str, Any]) -> Dict[str, bool]:
        training_results = {}
        try:
            # Train GARCH (replaces ETS)
            logger.info("Training GARCH model (replaces ETS)...")
            training_results['garch'] = self.models['garch'].train(prepared_data['ets_data'])

            # SVR
            logger.info("Training SVR model...")
            training_results['svr'] = self.models['svr'].train(prepared_data['sklearn_data'], optimize_params=False)

            # Random Forest
            logger.info("Training Random Forest model...")
            training_results['random_forest'] = self.models['random_forest'].train(prepared_data['sklearn_data'], optimize_params=False)

            # LightGBM
            logger.info("Training LightGBM model...")
            training_results['lightgbm'] = self.models['lightgbm'].train(prepared_data['sklearn_data'], optimize_params=False)

            # LSTM
            logger.info("Training LSTM model...")
            training_results['lstm'] = self.models['lstm'].train(prepared_data['lstm_data'])

            # TFT (experimental)
            try:
                logger.info("Training TFT-like model (optional)...")
                training_results['tft'] = self.models['tft'].train(prepared_data['lstm_data'])
            except Exception as e:
                logger.warning(f"Failed to train TFT model: {e}")
                training_results['tft'] = False

            # Kalman
            logger.info("Training Kalman model (lightweight)...")
            training_results['kalman'] = self.models['kalman'].train(prepared_data['sklearn_data'])

            successful_models = [name for name, success in training_results.items() if success]
            failed_models = [name for name, success in training_results.items() if not success]

            logger.info(f"Successfully trained models: {successful_models}")
            if failed_models:
                logger.warning(f"Failed to train models: {failed_models}")

            return training_results

        except Exception as e:
            logger.error(f"Failed to train individual models: {e}")
            return {name: False for name in self.models.keys()}

    def train(self, df: pd.DataFrame) -> bool:
        try:
            logger.info("Training ensemble model...")
            prepared_data = self.prepare_data_for_models(df)
            if not prepared_data:
                return False

            self.last_training_data = prepared_data['clean_df'].copy()
            training_results = self.train_individual_models(prepared_data)

            successful_count = sum(training_results.values())
            if successful_count < 3:
                logger.error(f"Only {successful_count} models trained successfully. Need at least 3.")
                return False

            self.adjust_weights_for_failed_models(training_results)
            self.calculate_training_scores(prepared_data['clean_df'])

            self.is_trained = True
            logger.info(f"Ensemble model trained successfully with {successful_count} models")
            logger.info(f"Adjusted weights: {self.weights}")
            return True

        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            return False

    def adjust_weights_for_failed_models(self, training_results: Dict[str, bool]):
        try:
            failed_models = [name for name, success in training_results.items() if not success]
            for model_name in failed_models:
                self.weights[model_name] = 0.0

            total_weight = sum(self.weights.values())
            if total_weight > 0:
                for model_name in self.weights:
                    self.weights[model_name] /= total_weight

            logger.info(f"Weights adjusted for failed models: {failed_models}")
        except Exception as e:
            logger.error(f"Failed to adjust weights: {e}")

    def predict(self, df: pd.DataFrame) -> Tuple[float, Tuple[float, float]]:
        """Make ensemble prediction with dynamic weighting."""
        try:
            if not self.is_trained:
                logger.error("Ensemble model not trained")
                return 0.0, (0.0, 0.0)

            prepared_data = self.prepare_data_for_models(df)
            if not prepared_data:
                return 0.0, (0.0, 0.0)

            predictions = {}
            confidence_intervals = {}

            # Calculate current market volatility for weight adjustment
            if len(df) >= 20:
                returns = df['close'].pct_change().dropna()
                current_vol = float(returns.std() * np.sqrt(252))  # Annualized
                self.volatility_history.append(current_vol)
            else:
                current_vol = 0.0
                self.volatility_history.append(current_vol)

            # Get predictions from each model
            for model_name, model in self.models.items():
                try:
                    # Route data types to models that need sequences vs. tabular
                    if model_name == 'garch':
                        pred, conf = model.predict(prepared_data['ets_data'])
                    elif model_name in ['lstm', 'tft']:
                        pred, conf = model.predict(prepared_data['lstm_data'])
                    else:
                        pred, conf = model.predict(prepared_data['sklearn_data'])

                    predictions[model_name] = pred
                    confidence_intervals[model_name] = conf

                except Exception as e:
                    logger.warning(f"Prediction failed for {model_name}: {e}")
                    predictions[model_name] = 0.0
                    confidence_intervals[model_name] = (0.0, 0.0)

            if not predictions:
                logger.error("No model predictions available")
                return 0.0, (0.0, 0.0)

            # Store predictions for performance tracking
            self.predictions_history.append(predictions)

            # Get current dynamic weights
            self.weights = self.weight_adjuster.get_current_weights()

            # Calculate weighted prediction
            ensemble_prediction = 0.0
            total_weight = 0.0
            model_contributions = {}

            for model_name, pred in predictions.items():
                weight = float(self.weights.get(model_name, 0.0))
                contribution = weight * pred
                model_contributions[model_name] = contribution
                ensemble_prediction += contribution
                total_weight += weight

            if total_weight > 0:
                ensemble_prediction /= total_weight

            # Calculate confidence interval based on model disagreement
            pred_values = list(predictions.values())
            pred_std = np.std(pred_values) if len(pred_values) > 1 else 0.0
            
            if pred_std > 0:
                ensemble_lower = ensemble_prediction - 2 * pred_std  # 95% confidence interval
                ensemble_upper = ensemble_prediction + 2 * pred_std
            else:
                margin = abs(ensemble_prediction) * 0.05
                ensemble_lower = ensemble_prediction - margin
                ensemble_upper = ensemble_prediction + margin

            # Log detailed prediction information
            logger.info(f"Current market volatility: {current_vol:.4f}")
            logger.info(f"Model weights: {self.weights}")
            logger.info(f"Model contributions: {model_contributions}")
            logger.info(f"Ensemble prediction: {ensemble_prediction:.2f} [{ensemble_lower:.2f}, {ensemble_upper:.2f}]")

            self.model_contributions = predictions.copy()
            self.detect_rapid_movements(df, ensemble_prediction)

            logger.debug(f"Ensemble prediction: {ensemble_prediction:.2f} [{ensemble_lower:.2f}, {ensemble_upper:.2f}]")
            return float(ensemble_prediction), (float(ensemble_lower), float(ensemble_upper))

        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return 0.0, (0.0, 0.0)

    def detect_rapid_movements(self, df: pd.DataFrame, predicted_price: float):
        try:
            if len(df) < 2:
                return
            current_price = df['close'].iloc[-1]
            previous_price = df['close'].iloc[-2]
            price_change = abs(predicted_price - current_price) / max(current_price, 1e-9)
            recent_change = abs(current_price - previous_price) / max(previous_price, 1e-9)
            if price_change > self.volatility_threshold or recent_change > self.volatility_threshold:
                self.rapid_movement_detected = True
                movement_type = 'rise' if predicted_price > current_price else 'fall'
                logger.warning(f"Rapid {movement_type} detected: {price_change*100:.2f}% predicted change")
            else:
                self.rapid_movement_detected = False
        except Exception as e:
            logger.error(f"Failed to detect rapid movements: {e}")

    def calculate_training_scores(self, df: pd.DataFrame):
        try:
            for model_name, model in self.models.items():
                if hasattr(model, 'training_score') and model.training_score:
                    self.training_scores[model_name] = model.training_score
            logger.debug(f"Training scores collected: {list(self.training_scores.keys())}")
        except Exception as e:
            logger.error(f"Failed to calculate training scores: {e}")

    def update_models(self, new_data: pd.DataFrame) -> bool:
        try:
            if not self.is_trained:
                return self.train(new_data)

            prepared_data = self.prepare_data_for_models(new_data)
            if not prepared_data:
                return False

            update_results = {}
            for model_name, model in self.models.items():
                weight = float(self.weights.get(model_name, 0.0))
                if weight <= 0.0:
                    continue
                try:
                    if model_name == 'garch':
                        update_results[model_name] = model.update_model(prepared_data['ets_data'])
                    elif model_name in ['lstm', 'tft']:
                        update_results[model_name] = model.update_model(prepared_data['lstm_data'])
                    else:
                        update_results[model_name] = model.update_model(prepared_data['sklearn_data'])
                except Exception as e:
                    logger.warning(f"Update failed for {model_name}: {e}")
                    update_results[model_name] = False

            successful_updates = sum(1 for v in update_results.values() if v)
            logger.info(f"Successfully updated {successful_updates} models")
            return successful_updates > 0

        except Exception as e:
            logger.error(f"Ensemble update failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        info = {
            'model_type': 'Ensemble',
            'is_trained': self.is_trained,
            'weights': self.weights.copy(),
            'individual_models': {},
            'rapid_movement_detected': self.rapid_movement_detected,
            'last_contributions': self.model_contributions.copy()
        }
        for model_name, model in self.models.items():
            if hasattr(model, 'get_model_info'):
                info['individual_models'][model_name] = model.get_model_info()
        return info

    def save_ensemble(self, filepath: str) -> bool:
        try:
            if not self.is_trained:
                logger.error("Cannot save untrained ensemble")
                return False

            models_dir = os.path.dirname(filepath)
            saved_models = {}
            for model_name, model in self.models.items():
                if float(self.weights.get(model_name, 0.0)) <= 0.0:
                    continue
                model_path = os.path.join(models_dir, f"{model_name}_model.joblib")
                try:
                    if model.save_model(model_path):
                        saved_models[model_name] = model_path
                except Exception:
                    logger.warning(f"Model {model_name} does not implement save_model or failed to save")

            ensemble_data = {
                'weights': self.weights,
                'is_trained': self.is_trained,
                'training_scores': self.training_scores,
                'saved_models': saved_models,
                'volatility_threshold': self.volatility_threshold
            }
            joblib.dump(ensemble_data, filepath)
            logger.info(f"Ensemble saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save ensemble: {e}")
            return False

    def load_ensemble(self, filepath: str) -> bool:
        try:
            if not os.path.exists(filepath):
                logger.error(f"Ensemble file not found: {filepath}")
                return False
            ensemble_data = joblib.load(filepath)
            self.weights = ensemble_data.get('weights', self.weights)
            self.is_trained = ensemble_data.get('is_trained', False)
            self.training_scores = ensemble_data.get('training_scores', {})
            self.volatility_threshold = ensemble_data.get('volatility_threshold', 0.05)

            saved_models = ensemble_data.get('saved_models', {})
            for model_name, model_path in saved_models.items():
                if model_name in self.models:
                    try:
                        if not self.models[model_name].load_model(model_path):
                            logger.warning(f"Failed to load {model_name} model")
                            self.weights[model_name] = 0.0
                    except Exception:
                        logger.warning(f"Failed to load {model_name} model")
                        self.weights[model_name] = 0.0

            logger.info(f"Ensemble loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")
            return False
"""Ensemble model combining ETS, SVR, Random Forest, LightGBM, and LSTM for BTC prediction."""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, List, Optional
from loguru import logger
import joblib
import os
import sys
from datetime import datetime
import threading
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from models.ets_model import ETSPredictor
from models.svr_model import SVRPredictor
from models.random_forest_model import RandomForestPredictor
from models.lightgbm_model import LightGBMPredictor
from models.lstm_model import LSTMPredictor
from models.kalman_model import KalmanPredictor
from services.feature_engineering import FeatureEngineer
from services.preprocessing import DataPreprocessor

class EnsemblePredictor:
    """Ensemble model combining multiple ML algorithms for BTC price prediction."""
    
    """Ensemble model combining multiple predictors for BTC price forecasting.
    This file replaces the older ETS-based ensemble with a volatility-aware
    GARCH model and adds an experimental TFT-like model. The ensemble
    keeps the same external interface used elsewhere in the codebase.
    """
    import os
    import sys
    from typing import Dict, Tuple, Any
    from datetime import datetime

    import numpy as np
    import pandas as pd
    from loguru import logger
    import joblib

    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.config import Config
    from models.garch_model import GARCHPredictor
    from models.svr_model import SVRPredictor
    from models.random_forest_model import RandomForestPredictor
    from models.lightgbm_model import LightGBMPredictor
    from models.lstm_model import LSTMPredictor
    from models.kalman_model import KalmanPredictor
    from models.tft_model import TFTPredictor
    from services.feature_engineering import FeatureEngineer
    from services.preprocessing import DataPreprocessor


    class EnsemblePredictor:
        """Ensemble model combining multiple ML algorithms for BTC price prediction."""

        def __init__(self):
            # Initialize individual models
            self.models = {
                'garch': GARCHPredictor(),
                'svr': SVRPredictor(),
                'kalman': KalmanPredictor(),
                'random_forest': RandomForestPredictor(),
                'lightgbm': LightGBMPredictor(),
                'lstm': LSTMPredictor(),
                'tft': TFTPredictor()
            }

            # Ensemble weights from config (copy to avoid mutating the class-level config)
            self.weights = Config.ENSEMBLE_WEIGHTS.copy()

            # Feature engineering and preprocessing
            self.feature_engineer = FeatureEngineer()
            self.preprocessor = DataPreprocessor()

            # Training state
            self.is_trained = False
            self.training_scores = {}
            self.model_contributions = {}
            self.last_training_data = None

            # Volatility detection
            self.volatility_threshold = 0.05  # 5% price change threshold
            self.rapid_movement_detected = False

        def prepare_data_for_models(self, df: pd.DataFrame) -> Dict[str, Any]:
            """Prepare and return a dictionary of data suitable for each model."""
            try:
                logger.info("Preparing data for ensemble models...")

                features_df = self.feature_engineer.create_all_features(df)
                if len(features_df) == 0:
                    logger.error("Feature engineering failed")
                    return {}

                clean_df = self.preprocessor.clean_data_for_training(features_df)

                prepared_data = {
                    'raw_df': df,
                    'features_df': features_df,
                    'clean_df': clean_df,
                    # Keep the key name `ets_data` for compatibility but it is used by GARCH now
                    'ets_data': clean_df,
                    'sklearn_data': clean_df,
                    'lstm_data': clean_df
                }

                logger.info(f"Data prepared: {len(clean_df)} samples, {len(clean_df.columns)} features")
                return prepared_data

            except Exception as e:
                logger.error(f"Failed to prepare ensemble data: {e}")
                return {}

        def train_individual_models(self, prepared_data: Dict[str, Any]) -> Dict[str, bool]:
            training_results = {}
            try:
                # Train GARCH (replaces ETS)
                logger.info("Training GARCH model (replaces ETS)...")
                training_results['garch'] = self.models['garch'].train(prepared_data['ets_data'])

                # SVR
                logger.info("Training SVR model...")
                training_results['svr'] = self.models['svr'].train(prepared_data['sklearn_data'], optimize_params=False)

                # Random Forest
                logger.info("Training Random Forest model...")
                training_results['random_forest'] = self.models['random_forest'].train(prepared_data['sklearn_data'], optimize_params=False)

                # LightGBM
                logger.info("Training LightGBM model...")
                training_results['lightgbm'] = self.models['lightgbm'].train(prepared_data['sklearn_data'], optimize_params=False)

                # LSTM
                logger.info("Training LSTM model...")
                training_results['lstm'] = self.models['lstm'].train(prepared_data['lstm_data'])

                # TFT (experimental)
                try:
                    logger.info("Training TFT-like model (optional)...")
                    training_results['tft'] = self.models['tft'].train(prepared_data['lstm_data'])
                except Exception as e:
                    logger.warning(f"Failed to train TFT model: {e}")
                    training_results['tft'] = False

                # Kalman
                logger.info("Training Kalman model (lightweight)...")
                training_results['kalman'] = self.models['kalman'].train(prepared_data['sklearn_data'])

                successful_models = [name for name, success in training_results.items() if success]
                failed_models = [name for name, success in training_results.items() if not success]

                logger.info(f"Successfully trained models: {successful_models}")
                if failed_models:
                    logger.warning(f"Failed to train models: {failed_models}")

                return training_results

            except Exception as e:
                logger.error(f"Failed to train individual models: {e}")
                return {name: False for name in self.models.keys()}

        def train(self, df: pd.DataFrame) -> bool:
            try:
                logger.info("Training ensemble model...")
                prepared_data = self.prepare_data_for_models(df)
                if not prepared_data:
                    return False

                self.last_training_data = prepared_data['clean_df'].copy()
                training_results = self.train_individual_models(prepared_data)

                successful_count = sum(training_results.values())
                if successful_count < 3:
                    logger.error(f"Only {successful_count} models trained successfully. Need at least 3.")
                    return False

                self.adjust_weights_for_failed_models(training_results)
                self.calculate_training_scores(prepared_data['clean_df'])

                self.is_trained = True
                logger.info(f"Ensemble model trained successfully with {successful_count} models")
                logger.info(f"Adjusted weights: {self.weights}")
                return True

            except Exception as e:
                logger.error(f"Ensemble training failed: {e}")
                return False

        def adjust_weights_for_failed_models(self, training_results: Dict[str, bool]):
            try:
                failed_models = [name for name, success in training_results.items() if not success]
                for model_name in failed_models:
                    self.weights[model_name] = 0.0

                total_weight = sum(self.weights.values())
                if total_weight > 0:
                    for model_name in self.weights:
                        self.weights[model_name] /= total_weight

                logger.info(f"Weights adjusted for failed models: {failed_models}")
            except Exception as e:
                logger.error(f"Failed to adjust weights: {e}")

        def predict(self, df: pd.DataFrame) -> Tuple[float, Tuple[float, float]]:
            try:
                if not self.is_trained:
                    logger.error("Ensemble model not trained")
                    return 0.0, (0.0, 0.0)

                prepared_data = self.prepare_data_for_models(df)
                if not prepared_data:
                    return 0.0, (0.0, 0.0)

                predictions = {}
                confidence_intervals = {}

                for model_name, model in self.models.items():
                    weight = float(self.weights.get(model_name, 0.0))
                    if weight <= 0.0:
                        continue

                    try:
                        # Route data types to models that need sequences vs. tabular
                        if model_name == 'garch':
                            pred, conf = model.predict(prepared_data['ets_data'])
                        elif model_name in ['lstm', 'tft']:
                            pred, conf = model.predict(prepared_data['lstm_data'])
                        else:
                            pred, conf = model.predict(prepared_data['sklearn_data'])

                        predictions[model_name] = pred
                        confidence_intervals[model_name] = conf

                    except Exception as e:
                        logger.warning(f"Prediction failed for {model_name}: {e}")
                        predictions[model_name] = 0.0
                        confidence_intervals[model_name] = (0.0, 0.0)

                if not predictions:
                    logger.error("No model predictions available")
                    return 0.0, (0.0, 0.0)

                # Weighted aggregation
                ensemble_prediction = 0.0
                total_weight = 0.0
                for model_name, pred in predictions.items():
                    w = float(self.weights.get(model_name, 0.0))
                    ensemble_prediction += w * pred
                    total_weight += w

                if total_weight > 0:
                    ensemble_prediction /= total_weight

                # Confidence interval as mean of model intervals (weighted could be used)
                lowers = [ci[0] for ci in confidence_intervals.values()]
                uppers = [ci[1] for ci in confidence_intervals.values()]
                if lowers and uppers:
                    ensemble_lower = float(np.mean(lowers))
                    ensemble_upper = float(np.mean(uppers))
                else:
                    margin = abs(ensemble_prediction) * 0.05
                    ensemble_lower = ensemble_prediction - margin
                    ensemble_upper = ensemble_prediction + margin

                self.model_contributions = predictions.copy()
                self.detect_rapid_movements(df, ensemble_prediction)

                logger.debug(f"Ensemble prediction: {ensemble_prediction:.2f} [{ensemble_lower:.2f}, {ensemble_upper:.2f}]")
                return float(ensemble_prediction), (float(ensemble_lower), float(ensemble_upper))

            except Exception as e:
                logger.error(f"Ensemble prediction failed: {e}")
                return 0.0, (0.0, 0.0)

        def detect_rapid_movements(self, df: pd.DataFrame, predicted_price: float):
            try:
                if len(df) < 2:
                    return
                current_price = df['close'].iloc[-1]
                previous_price = df['close'].iloc[-2]
                price_change = abs(predicted_price - current_price) / max(current_price, 1e-9)
                recent_change = abs(current_price - previous_price) / max(previous_price, 1e-9)
                if price_change > self.volatility_threshold or recent_change > self.volatility_threshold:
                    self.rapid_movement_detected = True
                    movement_type = 'rise' if predicted_price > current_price else 'fall'
                    logger.warning(f"Rapid {movement_type} detected: {price_change*100:.2f}% predicted change")
                else:
                    self.rapid_movement_detected = False
            except Exception as e:
                logger.error(f"Failed to detect rapid movements: {e}")

        def calculate_training_scores(self, df: pd.DataFrame):
            try:
                for model_name, model in self.models.items():
                    if hasattr(model, 'training_score') and model.training_score:
                        self.training_scores[model_name] = model.training_score
                logger.debug(f"Training scores collected: {list(self.training_scores.keys())}")
            except Exception as e:
                logger.error(f"Failed to calculate training scores: {e}")

        def update_models(self, new_data: pd.DataFrame) -> bool:
            try:
                if not self.is_trained:
                    return self.train(new_data)

                prepared_data = self.prepare_data_for_models(new_data)
                if not prepared_data:
                    return False

                update_results = {}
                for model_name, model in self.models.items():
                    weight = float(self.weights.get(model_name, 0.0))
                    if weight <= 0.0:
                        continue
                    try:
                        if model_name == 'garch':
                            update_results[model_name] = model.update_model(prepared_data['ets_data'])
                        elif model_name in ['lstm', 'tft']:
                            update_results[model_name] = model.update_model(prepared_data['lstm_data'])
                        else:
                            update_results[model_name] = model.update_model(prepared_data['sklearn_data'])
                    except Exception as e:
                        logger.warning(f"Update failed for {model_name}: {e}")
                        update_results[model_name] = False

                successful_updates = sum(1 for v in update_results.values() if v)
                logger.info(f"Successfully updated {successful_updates} models")
                return successful_updates > 0

            except Exception as e:
                logger.error(f"Ensemble update failed: {e}")
                return False

        def get_model_info(self) -> Dict[str, Any]:
            info = {
                'model_type': 'Ensemble',
                'is_trained': self.is_trained,
                'weights': self.weights.copy(),
                'individual_models': {},
                'rapid_movement_detected': self.rapid_movement_detected,
                'last_contributions': self.model_contributions.copy()
            }
            for model_name, model in self.models.items():
                if hasattr(model, 'get_model_info'):
                    info['individual_models'][model_name] = model.get_model_info()
            return info

        def save_ensemble(self, filepath: str) -> bool:
            try:
                if not self.is_trained:
                    logger.error("Cannot save untrained ensemble")
                    return False

                models_dir = os.path.dirname(filepath)
                saved_models = {}
                for model_name, model in self.models.items():
                    if float(self.weights.get(model_name, 0.0)) <= 0.0:
                        continue
                    model_path = os.path.join(models_dir, f"{model_name}_model.joblib")
                    try:
                        if model.save_model(model_path):
                            saved_models[model_name] = model_path
                    except Exception:
                        logger.warning(f"Model {model_name} does not implement save_model or failed to save")

                ensemble_data = {
                    'weights': self.weights,
                    'is_trained': self.is_trained,
                    'training_scores': self.training_scores,
                    'saved_models': saved_models,
                    'volatility_threshold': self.volatility_threshold
                }
                joblib.dump(ensemble_data, filepath)
                logger.info(f"Ensemble saved to {filepath}")
                return True
            except Exception as e:
                logger.error(f"Failed to save ensemble: {e}")
                return False

        def load_ensemble(self, filepath: str) -> bool:
            try:
                if not os.path.exists(filepath):
                    logger.error(f"Ensemble file not found: {filepath}")
                    return False
                ensemble_data = joblib.load(filepath)
                self.weights = ensemble_data.get('weights', self.weights)
                self.is_trained = ensemble_data.get('is_trained', False)
                self.training_scores = ensemble_data.get('training_scores', {})
                self.volatility_threshold = ensemble_data.get('volatility_threshold', 0.05)

                saved_models = ensemble_data.get('saved_models', {})
                for model_name, model_path in saved_models.items():
                    if model_name in self.models:
                        try:
                            if not self.models[model_name].load_model(model_path):
                                logger.warning(f"Failed to load {model_name} model")
                                self.weights[model_name] = 0.0
                        except Exception:
                            logger.warning(f"Failed to load {model_name} model")
                            self.weights[model_name] = 0.0

                logger.info(f"Ensemble loaded from {filepath}")
                return True
            except Exception as e:
                logger.error(f"Failed to load ensemble: {e}")
                return False