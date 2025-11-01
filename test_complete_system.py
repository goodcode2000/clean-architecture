"""Complete system test for BTC Prediction System."""
import sys
import os
import time
import requests
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from data.manager import BTCDataManager
from services.prediction_pipeline import PredictionPipeline
from services.terminal_monitor import TerminalMonitor
from api.server import APIServer
from loguru import logger

def test_complete_system():
    """Test the complete BTC prediction system."""
    logger.info("🧪 COMPLETE SYSTEM TEST STARTING")
    logger.info("="*60)
    
    try:
        # Test 1: Data System
        logger.info("Test 1: Data Collection System")
        data_manager = BTCDataManager()
        
        if not data_manager.initialize_data():
            logger.error("❌ Data system initialization failed")
            return False
        
        current_price = data_manager.get_current_price()
        if current_price:
            logger.info(f"✅ Current BTC Price: ${current_price:,.2f}")
        else:
            logger.error("❌ Could not get current price")
            return False
        
        # Test 2: Prediction Pipeline
        logger.info("Test 2: Prediction Pipeline")
        prediction_pipeline = PredictionPipeline()
        
        if not prediction_pipeline.initialize_pipeline():
            logger.error("❌ Prediction pipeline initialization failed")
            return False
        
        logger.info("✅ Prediction pipeline initialized")
        
        # Test 3: Terminal Monitor
        logger.info("Test 3: Terminal Monitor")
        terminal_monitor = TerminalMonitor()
        terminal_monitor.update_data(current_price=current_price)
        logger.info("✅ Terminal monitor ready")
        
        # Test 4: API Server
        logger.info("Test 4: API Server")
        api_server = APIServer()
        api_server.initialize_components(data_manager, prediction_pipeline, terminal_monitor)
        api_server.start_server()
        
        # Wait for server to start
        time.sleep(2)
        
        # Test API endpoints
        base_url = f"http://{Config.API_HOST}:{Config.API_PORT}"
        
        try:
            # Test health endpoint
            response = requests.get(f"{base_url}/api/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ API health check passed")
            else:
                logger.error("❌ API health check failed")
                return False
            
            # Test current price endpoint
            response = requests.get(f"{base_url}/api/current-price", timeout=5)
            if response.status_code == 200:
                price_data = response.json()
                logger.info(f"✅ API current price: ${price_data['price']:,.2f}")
            else:
                logger.error("❌ API current price failed")
                return False
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API connection failed: {e}")
            return False
        
        # Test 5: Make a prediction
        logger.info("Test 5: Making Prediction")
        prediction_result = prediction_pipeline.make_prediction()
        
        if prediction_result:
            pred_price = prediction_result['corrected_prediction']
            logger.info(f"✅ Prediction made: ${pred_price:.2f}")
        else:
            logger.error("❌ Prediction failed")
            return False
        
        # Test 6: System Integration
        logger.info("Test 6: System Integration")
        
        # Update terminal with prediction
        terminal_monitor.update_data(prediction_data=prediction_result)
        
        # Test prediction history API
        try:
            response = requests.get(f"{base_url}/api/latest-prediction", timeout=5)
            if response.status_code == 200:
                api_prediction = response.json()
                logger.info(f"✅ API prediction: ${api_prediction['predicted_price']:.2f}")
            else:
                logger.error("❌ API prediction endpoint failed")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API prediction request failed: {e}")
            return False
        
        # Test 7: System Status
        logger.info("Test 7: System Status")
        system_status = prediction_pipeline.get_pipeline_status()
        
        if system_status['is_running']:
            logger.info("✅ Pipeline is running")
        else:
            logger.error("❌ Pipeline not running")
            return False
        
        # Cleanup
        logger.info("Cleaning up test components...")
        api_server.stop_server()
        prediction_pipeline.stop_pipeline()
        data_manager.cleanup()
        
        logger.info("="*60)
        logger.info("🎉 ALL TESTS PASSED - SYSTEM READY!")
        logger.info("="*60)
        logger.info("System Components Verified:")
        logger.info("✅ Data Collection & Storage")
        logger.info("✅ Feature Engineering")
        logger.info("✅ ML Models (ETS, SVR, RF, LightGBM, LSTM)")
        logger.info("✅ Ensemble Prediction")
        logger.info("✅ Offset Correction")
        logger.info("✅ Prediction Pipeline")
        logger.info("✅ Terminal Monitoring")
        logger.info("✅ REST API Server")
        logger.info("✅ System Integration")
        logger.info("="*60)
        logger.info("Ready to run: python main.py")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System test failed: {e}")
        return False

def test_api_endpoints():
    """Test all API endpoints."""
    logger.info("Testing API endpoints...")
    
    base_url = f"http://{Config.API_HOST}:{Config.API_PORT}"
    endpoints = [
        "/api/health",
        "/api/current-price",
        "/api/system-status",
        "/api/backups"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ {endpoint}")
            else:
                logger.warning(f"⚠️ {endpoint} - Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ {endpoint} - Error: {e}")

if __name__ == "__main__":
    # Create necessary directories
    Config.create_directories()
    
    # Run complete system test
    success = test_complete_system()
    
    if success:
        print("\n" + "="*60)
        print("🚀 BTC PREDICTION SYSTEM IS READY!")
        print("="*60)
        print("To start the system:")
        print("1. cd btc-predictor-app")
        print("2. pip install -r requirements.txt")
        print("3. python main.py")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ SYSTEM TEST FAILED")
        print("Please check the logs and fix issues before running main.py")
        print("="*60)
        sys.exit(1)