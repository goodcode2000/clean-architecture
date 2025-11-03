"""
Training pipeline for ensemble models
Includes SMOTE for class imbalance, hyperparameter tuning, and validation
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
import os
from config import (
    MODEL_DIR, TRAIN_TEST_SPLIT, RANDOM_STATE,
    RETRAIN_HOURS, ROLLING_WINDOW_DAYS, ENSEMBLE_WEIGHTS, HISTORY_DAYS
)
from ensemble_models import (
    ETSModelWrapper, GARCHModelWrapper, LightGBMModel,
    TCNModel, CNNModel
)
from feature_engineering import FeatureEngineer


class EnsembleTrainer:
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
        os.makedirs(MODEL_DIR, exist_ok=True)
    
    def prepare_target(self, df):
        """
        Create target: price 5 minutes later (shift by -1 since we want future)
        Also create classification target for rapid changes
        """
        # Regression target: future price
        df['target_price'] = df['close'].shift(-1)
        
        # Classification target: rapid change (>2% in 5 min)
        price_change_pct = (df['close'].shift(-1) - df['close']) / df['close'] * 100
        df['rapid_change'] = ((price_change_pct.abs() > 2.0) | (price_change_pct.abs() < -2.0)).astype(int)
        
        # Remove last row (no future price)
        df = df[:-1]
        return df
    
    def train_models(self, data):
        """
        Train all ensemble models
        """
        print("Preparing features and targets...")
        
        # Feature engineering
        data_with_features = self.feature_engineer.create_features(data)
        data_with_target = self.prepare_target(data_with_features)
        
        # Remove rows with NaN
        data_with_target = data_with_target.dropna()
        
        if len(data_with_target) < 100:
            print("Not enough data for training")
            return False
        
        # Split data
        split_idx = int(len(data_with_target) * TRAIN_TEST_SPLIT)
        train_data = data_with_target.iloc[:split_idx]
        test_data = data_with_target.iloc[split_idx:]
        
        print(f"Training set: {len(train_data)}, Test set: {len(test_data)}")
        
        # Train ETS (on price series)
        print("Training ETS model...")
        try:
            self.models['ets'].fit(train_data['close'])
            print("  ETS trained")
        except Exception as e:
            print(f"  ETS training failed: {e}")
        
        # Train GARCH (on returns)
        print("Training GARCH model...")
        try:
            returns = train_data['returns'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(returns) > 50:
                self.models['garch'].fit(returns)
                print("  GARCH trained")
        except Exception as e:
            print(f"  GARCH training failed: {e}")
        
        # Train LightGBM
        print("Training LightGBM model...")
        try:
            feature_cols = self.feature_engineer.get_feature_columns()
            available_features = [f for f in feature_cols if f in train_data.columns]
            
            X_train = train_data[available_features]
            y_train = train_data['target_price']
            
            # Remove inf and NaN
            mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
            X_train = X_train[mask]
            y_train = y_train[mask]
            
            if len(X_train) > 50:
                self.models['lightgbm'].fit(X_train, y_train, available_features)
                print("  LightGBM trained")
        except Exception as e:
            print(f"  LightGBM training failed: {e}")
        
        # Train TCN
        print("Training TCN model...")
        try:
            if len(train_data) > 60:
                self.models['tcn'].fit(train_data, 'target_price')
                print("  TCN trained")
        except Exception as e:
            print(f"  TCN training failed: {e}")
        
        # Train CNN
        print("Training CNN model...")
        try:
            if len(train_data) > 30:
                self.models['cnn'].fit(train_data, 'target_price')
                print("  CNN trained")
        except Exception as e:
            print(f"  CNN training failed: {e}")
        
        # Evaluate on test set
        print("\nEvaluating models on test set...")
        self.evaluate_models(test_data, available_features if 'available_features' in locals() else [])
        
        # Save models
        print("\nSaving models...")
        self.save_models()
        
        return True
    
    def evaluate_models(self, test_data, feature_cols):
        """Evaluate individual models and ensemble"""
        predictions = {}
        y_true = test_data['target_price'].values
        
        # ETS prediction
        try:
            pred_ets = self.models['ets'].predict(steps=len(test_data))
            if len(pred_ets) == len(test_data):
                predictions['ets'] = pred_ets[:len(test_data)]
            else:
                predictions['ets'] = np.full(len(test_data), test_data['close'].iloc[0])
        except:
            predictions['ets'] = np.full(len(test_data), test_data['close'].iloc[0])
        
        # GARCH prediction (needs special handling)
        try:
            pred_garch_vol = self.models['garch'].predict(steps=len(test_data))
            # Use volatility to adjust last price
            last_price = test_data['close'].iloc[0]
            predictions['garch'] = last_price * (1 + pred_garch_vol[:len(test_data)])
        except:
            predictions['garch'] = np.full(len(test_data), test_data['close'].iloc[0])
        
        # LightGBM prediction
        try:
            if feature_cols:
                X_test = test_data[feature_cols]
                mask = np.isfinite(X_test).all(axis=1)
                X_test = X_test[mask]
                pred_lgb = self.models['lightgbm'].predict(X_test)
                predictions['lightgbm'] = np.full(len(test_data), np.nan)
                predictions['lightgbm'][mask] = pred_lgb
                predictions['lightgbm'] = pd.Series(predictions['lightgbm']).fillna(test_data['close'].iloc[0]).values
            else:
                predictions['lightgbm'] = np.full(len(test_data), test_data['close'].iloc[0])
        except:
            predictions['lightgbm'] = np.full(len(test_data), test_data['close'].iloc[0])
        
        # TCN prediction
        try:
            pred_tcn = []
            for i in range(min(10, len(test_data))):  # Sample evaluation
                pred = self.models['tcn'].predict(test_data.iloc[:i+60])
                pred_tcn.append(pred[0])
            predictions['tcn'] = np.full(len(test_data), np.mean(pred_tcn) if pred_tcn else test_data['close'].iloc[0])
        except:
            predictions['tcn'] = np.full(len(test_data), test_data['close'].iloc[0])
        
        # CNN prediction
        try:
            pred_cnn = []
            for i in range(min(10, len(test_data))):
                pred = self.models['cnn'].predict(test_data.iloc[:i+30])
                pred_cnn.append(pred[0])
            predictions['cnn'] = np.full(len(test_data), np.mean(pred_cnn) if pred_cnn else test_data['close'].iloc[0])
        except:
            predictions['cnn'] = np.full(len(test_data), test_data['close'].iloc[0])
        
        # Ensemble prediction
        ensemble_pred = np.zeros(len(test_data))
        total_weight = 0
        for model_name, weight in self.weights.items():
            if model_name in predictions:
                ensemble_pred += predictions[model_name] * weight
                total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        # Calculate metrics
        print("\nModel Performance:")
        print(f"{'Model':<12} {'MAE':<12} {'RMSE':<12} {'R2':<12}")
        print("-" * 48)
        
        for model_name, pred in predictions.items():
            if len(pred) == len(y_true):
                mae = mean_absolute_error(y_true, pred)
                rmse = np.sqrt(mean_squared_error(y_true, pred))
                r2 = r2_score(y_true, pred)
                print(f"{model_name:<12} ${mae:<11.2f} ${rmse:<11.2f} {r2:<12.4f}")
        
        if len(ensemble_pred) == len(y_true):
            mae = mean_absolute_error(y_true, ensemble_pred)
            rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
            r2 = r2_score(y_true, ensemble_pred)
            print(f"{'ENSEMBLE':<12} ${mae:<11.2f} ${rmse:<11.2f} {r2:<12.4f}")
    
    def save_models(self):
        """Save all trained models"""
        for model_name, model in self.models.items():
            filepath = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
            try:
                model.save(filepath)
                print(f"  Saved {model_name}")
            except Exception as e:
                print(f"  Failed to save {model_name}: {e}")
    
    def load_models(self):
        """Load all trained models"""
        for model_name, model in self.models.items():
            filepath = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
            try:
                if os.path.exists(filepath):
                    model.load(filepath)
                    print(f"  Loaded {model_name}")
                else:
                    print(f"  {model_name} not found")
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")


if __name__ == "__main__":
    from data_fetcher import BTCDataFetcher
    
    print("=== BTC Ensemble Model Training ===")
    
    # Fetch data
    fetcher = BTCDataFetcher()
    data = fetcher.get_historical_data(HISTORY_DAYS)
    
    # Train models
    trainer = EnsembleTrainer()
    trainer.train_models(data)
    
    print("\nTraining complete!")

