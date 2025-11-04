"""Test script for BTC data collection and storage system."""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.manager import BTCDataManager
from config.config import Config
from loguru import logger
import time

def test_data_system():
    """Test the complete data system."""
    logger.info("Testing BTC Data Collection System")
    logger.info("=" * 50)
    
    try:
        # Initialize data manager
        data_manager = BTCDataManager()
        
        # Test 1: Get current price
        logger.info("Test 1: Getting current BTC price...")
        current_price = data_manager.get_current_price()
        if current_price:
            logger.info(f"✓ Current BTC price: ${current_price:,.2f}")
        else:
            logger.error("✗ Failed to get current price")
            return False
        
        # Test 2: Initialize historical data (small sample for testing)
        logger.info("Test 2: Initializing historical data...")
        if data_manager.initialize_data():
            logger.info("✓ Historical data initialized successfully")
        else:
            logger.error("✗ Failed to initialize historical data")
            return False
        
        # Test 3: Get latest data
        logger.info("Test 3: Getting latest data records...")
        latest_data = data_manager.get_latest_data(n_records=10)
        if latest_data is not None and len(latest_data) > 0:
            logger.info(f"✓ Retrieved {len(latest_data)} latest records")
            logger.info(f"  Latest timestamp: {latest_data['timestamp'].max()}")
            logger.info(f"  Latest price: ${latest_data['close'].iloc[-1]:,.2f}")
        else:
            logger.error("✗ Failed to get latest data")
            return False
        
        # Test 4: Data validation
        logger.info("Test 4: Testing data validation...")
        is_valid, issues = data_manager.storage.validate_price_data(latest_data)
        if is_valid:
            logger.info("✓ Data validation passed")
        else:
            logger.warning(f"⚠ Data validation issues: {issues}")
        
        # Test 5: Get training data
        logger.info("Test 5: Getting data for training...")
        training_data = data_manager.get_data_for_training(lookback_days=7)  # Just 7 days for testing
        if training_data is not None and len(training_data) > 0:
            logger.info(f"✓ Training data prepared: {len(training_data)} records")
        else:
            logger.error("✗ Failed to prepare training data")
            return False
        
        # Test 6: System status
        logger.info("Test 6: Getting system status...")
        status = data_manager.get_system_status()
        logger.info("✓ System Status:")
        for key, value in status.items():
            if key != 'error':
                logger.info(f"  {key}: {value}")
        
        # Test 7: Save a test prediction
        logger.info("Test 7: Testing prediction storage...")
        test_prediction = current_price * 1.01  # Predict 1% increase
        confidence = (current_price * 0.99, current_price * 1.03)
        model_contributions = {
            'garch': current_price * 1.005,
            'svr': current_price * 1.01,
            'random_forest': current_price * 1.015,
            'lightgbm': current_price * 1.008,
            'lstm': current_price * 1.012
        }
        
        if data_manager.save_prediction(test_prediction, confidence, model_contributions):
            logger.info("✓ Test prediction saved successfully")
        else:
            logger.error("✗ Failed to save test prediction")
        
        logger.info("=" * 50)
        logger.info("✓ All data system tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data system test failed: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            data_manager.cleanup()
        except:
            pass

def test_real_time_updates():
    """Test real-time data updates (run for a short period)."""
    logger.info("Testing real-time data updates...")
    
    data_manager = BTCDataManager()
    
    try:
        # Initialize data
        if not data_manager.initialize_data():
            logger.error("Failed to initialize data for real-time test")
            return False
        
        # Start automatic updates
        data_manager.start_automatic_updates()
        
        # Monitor for 2 minutes
        logger.info("Monitoring updates for 2 minutes...")
        for i in range(4):  # 4 * 30 seconds = 2 minutes
            time.sleep(30)
            current_price = data_manager.get_current_price()
            logger.info(f"Update check {i+1}: Current price = ${current_price:,.2f}")
        
        # Stop updates
        data_manager.stop_automatic_updates()
        logger.info("✓ Real-time update test completed")
        return True
        
    except Exception as e:
        logger.error(f"Real-time update test failed: {e}")
        return False
    
    finally:
        data_manager.cleanup()

if __name__ == "__main__":
    # Create necessary directories
    Config.create_directories()
    
    # Run basic tests
    if test_data_system():
        logger.info("Basic tests passed!")
        
        # Ask user if they want to test real-time updates
        test_realtime = input("Test real-time updates? (y/n): ").lower().strip()
        if test_realtime == 'y':
            test_real_time_updates()
    else:
        logger.error("Basic tests failed!")
        sys.exit(1)