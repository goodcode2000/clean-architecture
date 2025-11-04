"""Smoke tests for the enhanced LSTM model with multi-source data."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models.lstm_model import LSTMPredictor
from config.config import Config
from services.feature_engineering import FeatureEngineer

def generate_synthetic_data(n_samples: int = 1000):
    """Generate synthetic data for testing."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=100),
        periods=n_samples,
        freq='1H'
    )
    
    # Generate synthetic price data
    df = pd.DataFrame({
        'open': np.random.normal(100, 10, n_samples),
        'high': np.random.normal(105, 12, n_samples),
        'low': np.random.normal(95, 8, n_samples),
        'close': np.random.normal(102, 11, n_samples),
        'volume': np.random.lognormal(10, 1, n_samples),
    }, index=dates)
    
    # Ensure high > low and proper OHLC relationships
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    # Add synthetic sentiment data
    df['sentiment_score'] = np.sin(np.linspace(0, 8*np.pi, n_samples)) * 0.5 + np.random.normal(0, 0.1, n_samples)
    df['sentiment_magnitude'] = np.abs(df['sentiment_score']) + np.random.normal(0, 0.05, n_samples)
    
    # Add synthetic market depth data
    df['bid_sum'] = np.random.lognormal(10, 0.5, n_samples)
    df['ask_sum'] = np.random.lognormal(10, 0.5, n_samples)
    df['imbalance'] = df['bid_sum'] - df['ask_sum']
    
    # Add synthetic derivatives data
    df['funding_rate'] = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 0.001 + np.random.normal(0, 0.0001, n_samples)
    df['open_interest'] = np.random.lognormal(12, 0.3, n_samples)
    
    return df

def test_lstm_model_training():
    """Smoke test for LSTM model training with all feature groups."""
    # Generate synthetic data
    df = generate_synthetic_data(1000)
    
    # Initialize model
    model = LSTMPredictor()
    
    # Train model
    success = model.train(df, validation_split=0.2)
    
    # Verify training success
    assert success, "LSTM model training failed"
    assert model.is_trained, "LSTM model should be marked as trained"
    assert model.model is not None, "LSTM model should be initialized"

def test_lstm_prediction_shape():
    """Test LSTM model prediction output shape and bounds."""
    # Generate synthetic data
    df = generate_synthetic_data(1000)
    
    # Initialize and train model
    model = LSTMPredictor()
    model.train(df, validation_split=0.2)
    
    # Make prediction
    pred_value, (lower, upper) = model.predict(df.iloc[-100:])
    
    # Verify prediction
    assert isinstance(pred_value, float), "Prediction should be a float"
    assert lower <= pred_value <= upper, "Prediction should be within confidence bounds"
    assert not np.isnan(pred_value), "Prediction should not be NaN"
    assert not np.isnan(lower) and not np.isnan(upper), "Confidence bounds should not be NaN"

def test_feature_engineering():
    """Test feature engineering with all data sources."""
    # Generate synthetic data
    df = generate_synthetic_data(1000)
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Compute features
    start_time = df.index.min()
    end_time = df.index.max()
    features = engineer.prepare_features(start_time, end_time)
    
    # Verify features
    assert isinstance(features, dict), "Features should be returned as dictionary"
    assert all(group in features for group in Config.FEATURE_GROUPS), "All feature groups should be present"
    assert all(len(v) > 0 for v in features.values()), "All feature arrays should be non-empty"

def test_model_with_missing_features():
    """Test model behavior with missing feature groups."""
    # Generate minimal data
    df = generate_synthetic_data(1000)
    
    # Remove some feature groups
    for col in ['sentiment_score', 'sentiment_magnitude']:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Train model
    model = LSTMPredictor()
    success = model.train(df, validation_split=0.2)
    
    # Verify model adapts to missing features
    assert success, "Model should train successfully even with missing features"
    
    # Test prediction
    pred_value, bounds = model.predict(df.iloc[-100:])
    assert not np.isnan(pred_value), "Should predict even with missing features"

if __name__ == '__main__':
    pytest.main([__file__])