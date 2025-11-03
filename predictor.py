"""
Ensemble predictor with offset tracking and correction
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from config import MODEL_DIR, PREDICTIONS_FILE, ENSEMBLE_WEIGHTS, INTERVAL_MINUTES
from ensemble_models import (
    ETSModelWrapper, GARCHModelWrapper, LightGBMModel,
    TCNModel, CNNModel
)
from feature_engineering import FeatureEngineer


class EnsemblePredictor:
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.models = {
            'ets': ETSModelWrapper(),
            'garch': GARCHModelWrapper(),
            'lightgbm': LightGBMModel(),
            'tcn': TCNModel(sequence_length=60),
            'cnn': CNNModel(sequence_length=30)
        }
        self.weights = ENSEMBLE_WEIGHTS
        self.load_models()
        self.residual_model = LightGBMModel()
        
        # Initialize predictions file
        os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
        if not os.path.exists(PREDICTIONS_FILE):
            pd.DataFrame(columns=[
                'timestamp', 'prediction_target_time', 'real_price', 'predicted_price', 'offset',
                'rapid_change_predicted', 'model_ets', 'model_garch',
                'model_lightgbm', 'model_tcn', 'model_cnn'
            ]).to_csv(PREDICTIONS_FILE, index=False)
    
    def load_models(self):
        """Load all trained models"""
        for model_name, model in self.models.items():
            filepath = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
            try:
                if os.path.exists(filepath):
                    model.load(filepath)
                else:
                    print(f"Warning: {model_name} model not found")
            except Exception as e:
                print(f"Warning: Failed to load {model_name}: {e}")
        # Load residual model
        try:
            res_path = os.path.join(MODEL_DIR, "residual_model.pkl")
            if os.path.exists(res_path):
                self.residual_model.load(res_path)
        except Exception as e:
            print(f"Warning: Failed to load residual model: {e}")

    def compute_online_weights(self):
        """Compute online error-based weights from recent history"""
        if not os.path.exists(PREDICTIONS_FILE):
            return self.weights
        try:
            df = pd.read_csv(PREDICTIONS_FILE)
            if len(df) < 20:
                return self.weights
            # Align realized future price: the next row's real_price is the outcome of previous prediction
            df = df.copy()
            df['real_future'] = df['real_price'].shift(-1)
            recent = df.tail(150)
            model_cols = ['model_ets','model_garch','model_lightgbm','model_tcn','model_cnn']
            maes = {}
            for col in model_cols:
                if col in recent.columns:
                    err = (recent[col] - recent['real_future']).abs()
                    err = err.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(err) >= 10:
                        maes[col] = err.mean()
            if not maes:
                return self.weights
            # Inverse error weighting with floor to avoid explosion
            inv = {k: 1.0 / max(v, 1e-6) for k, v in maes.items()}
            # Map to model keys
            w = {
                'ets': inv.get('model_ets', 0.0),
                'garch': inv.get('model_garch', 0.0),
                'lightgbm': inv.get('model_lightgbm', 0.0),
                'tcn': inv.get('model_tcn', 0.0),
                'cnn': inv.get('model_cnn', 0.0)
            }
            total = sum(w.values())
            if total == 0:
                return self.weights
            return {k: v/total for k, v in w.items()}
        except Exception:
            return self.weights
    
    def predict(self, data):
        """
        Make ensemble prediction
        
        Args:
            data: DataFrame with recent data including features
            
        Returns:
            dict with predictions from all models and ensemble
        """
        predictions = {}
        
        # ETS prediction
        try:
            pred_ets = self.models['ets'].predict(steps=1)[0]
            predictions['ets'] = pred_ets
        except:
            predictions['ets'] = data['close'].iloc[-1]
        
        # GARCH prediction
        try:
            returns = data['returns'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(returns) > 0:
                pred_garch_vol = self.models['garch'].predict(steps=1)[0]
                last_price = data['close'].iloc[-1]
                predictions['garch'] = last_price * (1 + pred_garch_vol)
            else:
                predictions['garch'] = data['close'].iloc[-1]
        except:
            predictions['garch'] = data['close'].iloc[-1]
        
        # LightGBM prediction
        try:
            feature_cols = self.feature_engineer.get_feature_columns()
            available_features = [f for f in feature_cols if f in data.columns]
            
            if available_features and hasattr(self.models['lightgbm'], 'feature_cols'):
                X = data[available_features].iloc[-1:].values
                if np.isfinite(X).all():
                    pred_lgb = self.models['lightgbm'].predict(X)[0]
                    predictions['lightgbm'] = pred_lgb
                else:
                    predictions['lightgbm'] = data['close'].iloc[-1]
            else:
                predictions['lightgbm'] = data['close'].iloc[-1]
        except Exception as e:
            predictions['lightgbm'] = data['close'].iloc[-1]
        
        # TCN prediction
        try:
            if len(data) >= 60:
                pred_tcn = self.models['tcn'].predict(data)[0]
                predictions['tcn'] = pred_tcn
            else:
                predictions['tcn'] = data['close'].iloc[-1]
        except:
            predictions['tcn'] = data['close'].iloc[-1]
        
        # CNN prediction
        try:
            if len(data) >= 30:
                pred_cnn = self.models['cnn'].predict(data)[0]
                predictions['cnn'] = pred_cnn
            else:
                predictions['cnn'] = data['close'].iloc[-1]
        except:
            predictions['cnn'] = data['close'].iloc[-1]
        
        # Weighted ensemble with online error-based weights
        dynamic_weights = self.compute_online_weights()
        ensemble_pred = 0.0
        total_weight = 0.0
        for model_name, weight in dynamic_weights.items():
            if model_name in predictions:
                ensemble_pred += predictions[model_name] * weight
                total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        else:
            ensemble_pred = data['close'].iloc[-1]
        
        # Residual correction if model available
        try:
            feature_cols = self.feature_engineer.get_feature_columns()
            available_features = [f for f in feature_cols if f in data.columns]
            if hasattr(self.residual_model, 'model') and self.residual_model.model is not None and available_features:
                X_last = data[available_features].iloc[-1:].values
                if np.isfinite(X_last).all():
                    residual_pred = float(self.residual_model.predict(X_last)[0])
                    # Apply conservative residual correction (shrink)
                    ensemble_pred = ensemble_pred + 0.6 * residual_pred
        except Exception:
            pass

        # Clamp overly large jumps to reduce extreme offsets (based on recent volatility)
        try:
            recent_vol = float(data['realized_volatility_short'].iloc[-1]) if 'realized_volatility_short' in data.columns else float(data['volatility'].iloc[-1])
            price_now = float(data['close'].iloc[-1])
            max_jump = max(50.0, 3.0 * recent_vol * price_now)
            delta = ensemble_pred - price_now
            if abs(delta) > max_jump:
                ensemble_pred = price_now + np.sign(delta) * max_jump
        except Exception:
            pass

        predictions['ensemble'] = ensemble_pred
        
        # Predict rapid change (if price change > 2%)
        current_price = data['close'].iloc[-1]
        price_change_pct = (ensemble_pred - current_price) / current_price * 100
        rapid_change = abs(price_change_pct) > 2.0
        
        predictions['rapid_change'] = rapid_change
        predictions['price_change_pct'] = price_change_pct
        
        return predictions
    
    def save_prediction(self, real_price, predicted_price, predictions_dict):
        """Save prediction to file for offset tracking"""
        offset = abs(real_price - predicted_price)
        prediction_time = datetime.now()
        prediction_target_time = prediction_time + timedelta(minutes=5)
        
        new_row = {
            'timestamp': prediction_time,
            'prediction_target_time': prediction_target_time,
            'real_price': real_price,
            'predicted_price': predicted_price,
            'offset': offset,
            'rapid_change_predicted': predictions_dict.get('rapid_change', False),
            'model_ets': predictions_dict.get('ets', 0),
            'model_garch': predictions_dict.get('garch', 0),
            'model_lightgbm': predictions_dict.get('lightgbm', 0),
            'model_tcn': predictions_dict.get('tcn', 0),
            'model_cnn': predictions_dict.get('cnn', 0)
        }
        
        df = pd.read_csv(PREDICTIONS_FILE) if os.path.exists(PREDICTIONS_FILE) else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(PREDICTIONS_FILE, index=False)
    
    def get_offset_statistics(self):
        """Get statistics about prediction offset"""
        if not os.path.exists(PREDICTIONS_FILE):
            return None
        
        df = pd.read_csv(PREDICTIONS_FILE)
        if len(df) == 0:
            return None
        
        return {
            'mean_offset': df['offset'].mean(),
            'median_offset': df['offset'].median(),
            'std_offset': df['offset'].std(),
            'min_offset': df['offset'].min(),
            'max_offset': df['offset'].max(),
            'count': len(df)
        }
    
    def apply_offset_correction(self, prediction, offset_stats):
        """
        Apply offset correction based on historical errors
        This helps reduce systematic bias
        """
        if offset_stats is None:
            return prediction
        
        # Simple correction: subtract mean offset if positive bias detected
        # This is a basic approach; can be enhanced with more sophisticated methods
        mean_offset = offset_stats['mean_offset']
        if mean_offset > 0:
            # If we typically over-predict, reduce prediction slightly
            correction_factor = 1 - (mean_offset / prediction) * 0.1  # Small correction
            return prediction * correction_factor
        
        return prediction

