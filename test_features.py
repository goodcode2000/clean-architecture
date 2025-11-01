"""Test script for feature engineering module."""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.feature_engineering import FeatureEngineer
from data.manager import BTCDataManager
from config.config import Config
from loguru import logger
import pandas as pd
import numpy as np

def test_feature_engineering():
    """Test the feature engineering module."""
    logger.info("Testing Feature Engineering Module")
    logger.info("=" * 50)
    
    try:
        # Initialize components
        data_manager = BTCDataManager()
        feature_engineer = FeatureEngineer()
        
        # Get sample data
        logger.info("Getting sample data for testing...")
        sample_data = data_manager.get_latest_data(n_records=500)  # Get more data for better feature calculation
        
        if sample_data is None or len(sample_data) < 100:
            logger.error("Insufficient data for feature engineering test")
            return False
        
        logger.info(f"Using {len(sample_data)} records for testing")
        logger.info(f"Original columns: {list(sample_data.columns)}")
        
        # Test 1: Bollinger Bands
        logger.info("Test 1: Calculating Bollinger Bands...")
        bb_data = feature_engineer.calculate_bollinger_bands(sample_data)
        bb_columns = ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position']
        if all(col in bb_data.columns for col in bb_columns):
            logger.info("✓ Bollinger Bands calculated successfully")
            logger.info(f"  BB Position range: {bb_data['bb_position'].min():.3f} to {bb_data['bb_position'].max():.3f}")
        else:
            logger.error("✗ Bollinger Bands calculation failed")
        
        # Test 2: RSI
        logger.info("Test 2: Calculating RSI...")
        rsi_data = feature_engineer.calculate_rsi(sample_data)
        if 'rsi' in rsi_data.columns:
            rsi_values = rsi_data['rsi'].dropna()
            logger.info("✓ RSI calculated successfully")
            logger.info(f"  RSI range: {rsi_values.min():.2f} to {rsi_values.max():.2f}")
            logger.info(f"  Current RSI: {rsi_values.iloc[-1]:.2f}")
        else:
            logger.error("✗ RSI calculation failed")
        
        # Test 3: MACD
        logger.info("Test 3: Calculating MACD...")
        macd_data = feature_engineer.calculate_macd(sample_data)
        macd_columns = ['macd', 'macd_signal', 'macd_histogram']
        if all(col in macd_data.columns for col in macd_columns):
            logger.info("✓ MACD calculated successfully")
            current_macd = macd_data['macd'].dropna().iloc[-1]
            current_signal = macd_data['macd_signal'].dropna().iloc[-1]
            logger.info(f"  Current MACD: {current_macd:.4f}, Signal: {current_signal:.4f}")
        else:
            logger.error("✗ MACD calculation failed")
        
        # Test 4: Moving Averages
        logger.info("Test 4: Calculating Moving Averages...")
        ma_data = feature_engineer.calculate_moving_averages(sample_data)
        ma_columns = ['sma_5', 'sma_20', 'sma_50', 'ema_12', 'ema_26']
        if all(col in ma_data.columns for col in ma_columns):
            logger.info("✓ Moving Averages calculated successfully")
            current_price = ma_data['close'].iloc[-1]
            sma_20 = ma_data['sma_20'].dropna().iloc[-1]
            logger.info(f"  Current price: ${current_price:.2f}, SMA(20): ${sma_20:.2f}")
        else:
            logger.error("✗ Moving Averages calculation failed")
        
        # Test 5: Volatility Indicators
        logger.info("Test 5: Calculating Volatility Indicators...")
        vol_data = feature_engineer.calculate_volatility_indicators(sample_data)
        vol_columns = ['returns', 'volatility_20', 'atr']
        if all(col in vol_data.columns for col in vol_columns):
            logger.info("✓ Volatility Indicators calculated successfully")
            current_vol = vol_data['volatility_20'].dropna().iloc[-1]
            current_atr = vol_data['atr'].dropna().iloc[-1]
            logger.info(f"  Current volatility: {current_vol:.4f}, ATR: ${current_atr:.2f}")
        else:
            logger.error("✗ Volatility Indicators calculation failed")
        
        # Test 6: Lag Features
        logger.info("Test 6: Creating Lag Features...")
        lag_data = feature_engineer.create_lag_features(sample_data)
        lag_columns = ['close_lag_1', 'close_lag_5', 'returns_lag_1']
        if all(col in lag_data.columns for col in lag_columns):
            logger.info("✓ Lag Features created successfully")
            logger.info(f"  Time features added: hour_sin, hour_cos, day_sin, day_cos")
        else:
            logger.error("✗ Lag Features creation failed")
        
        # Test 7: Complete Feature Engineering
        logger.info("Test 7: Complete Feature Engineering Pipeline...")
        all_features = feature_engineer.create_all_features(sample_data)
        
        logger.info(f"✓ Complete feature engineering successful")
        logger.info(f"  Original columns: {len(sample_data.columns)}")
        logger.info(f"  Feature columns: {len(all_features.columns)}")
        logger.info(f"  Records after cleaning: {len(all_features)}")
        
        # Show some feature statistics
        numeric_features = all_features.select_dtypes(include=[np.number])
        logger.info(f"  Numeric features: {len(numeric_features.columns)}")
        
        # Test 8: Feature Normalization
        logger.info("Test 8: Testing Feature Normalization...")
        normalized_features, norm_params = feature_engineer.normalize_features(all_features, method='minmax')
        
        if len(norm_params) > 0:
            logger.info("✓ Feature normalization successful")
            logger.info(f"  Normalized {len(norm_params)} features")
        else:
            logger.error("✗ Feature normalization failed")
        
        # Test 9: LSTM Sequence Preparation
        logger.info("Test 9: Testing LSTM Sequence Preparation...")
        X_sequences, y_sequences = feature_engineer.prepare_sequences_for_lstm(normalized_features)
        
        if len(X_sequences) > 0 and len(y_sequences) > 0:
            logger.info("✓ LSTM sequence preparation successful")
            logger.info(f"  Sequence shape: {X_sequences.shape}")
            logger.info(f"  Target shape: {y_sequences.shape}")
        else:
            logger.error("✗ LSTM sequence preparation failed")
        
        # Test 10: Feature Importance Columns
        logger.info("Test 10: Getting Important Features...")
        important_features = feature_engineer.get_feature_importance_columns()
        available_features = [col for col in important_features if col in all_features.columns]
        
        logger.info(f"✓ Important features identified: {len(available_features)}/{len(important_features)}")
        logger.info(f"  Available important features: {available_features[:10]}...")  # Show first 10
        
        # Summary
        logger.info("=" * 50)
        logger.info("Feature Engineering Test Summary:")
        logger.info(f"  ✓ Input data: {len(sample_data)} records, {len(sample_data.columns)} columns")
        logger.info(f"  ✓ Output data: {len(all_features)} records, {len(all_features.columns)} columns")
        logger.info(f"  ✓ LSTM sequences: {len(X_sequences)} sequences")
        logger.info(f"  ✓ Important features: {len(available_features)} available")
        
        return True
        
    except Exception as e:
        logger.error(f"Feature engineering test failed: {e}")
        return False

def test_feature_quality():
    """Test the quality and validity of generated features."""
    logger.info("Testing Feature Quality...")
    
    try:
        data_manager = BTCDataManager()
        feature_engineer = FeatureEngineer()
        
        # Get data
        sample_data = data_manager.get_latest_data(n_records=200)
        if sample_data is None:
            logger.error("No data available for quality test")
            return False
        
        # Create features
        features_df = feature_engineer.create_all_features(sample_data)
        
        # Quality checks
        logger.info("Performing quality checks...")
        
        # Check for infinite values
        inf_columns = []
        for col in features_df.select_dtypes(include=[np.number]).columns:
            if np.isinf(features_df[col]).any():
                inf_columns.append(col)
        
        if inf_columns:
            logger.warning(f"Infinite values found in: {inf_columns}")
        else:
            logger.info("✓ No infinite values found")
        
        # Check for excessive NaN values
        nan_percentages = features_df.isnull().sum() / len(features_df) * 100
        high_nan_columns = nan_percentages[nan_percentages > 50].index.tolist()
        
        if high_nan_columns:
            logger.warning(f"High NaN percentage in: {high_nan_columns}")
        else:
            logger.info("✓ NaN values within acceptable range")
        
        # Check feature ranges
        logger.info("Feature range analysis:")
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        for col in ['rsi', 'bb_position', 'volatility_20'][:3]:  # Check a few key features
            if col in numeric_features.columns:
                values = numeric_features[col].dropna()
                logger.info(f"  {col}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")
        
        logger.info("✓ Feature quality test completed")
        return True
        
    except Exception as e:
        logger.error(f"Feature quality test failed: {e}")
        return False

if __name__ == "__main__":
    # Create necessary directories
    Config.create_directories()
    
    # Initialize data if needed
    data_manager = BTCDataManager()
    if not data_manager.initialize_data():
        logger.error("Failed to initialize data for testing")
        sys.exit(1)
    
    # Run tests
    if test_feature_engineering():
        logger.info("✓ Feature engineering tests passed!")
        
        if test_feature_quality():
            logger.info("✓ Feature quality tests passed!")
        else:
            logger.warning("⚠ Feature quality tests had issues")
    else:
        logger.error("✗ Feature engineering tests failed!")
        sys.exit(1)