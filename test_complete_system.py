#!/usr/bin/env python3
"""
Complete System Test for BTC Predictor
Tests all major components to ensure system is ready to run
"""
import sys
import os
import time
import requests
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from config.config import Config
        print("✅ Config imported successfully")
        
        from data.collector import BTCDataCollector
        print("✅ Data collector imported successfully")
        
        from data.manager import BTCDataManager
        print("✅ Data manager imported successfully")
        
        from models.ensemble_model import EnsemblePredictor
        print("✅ Ensemble model imported successfully")
        
        from services.prediction_pipeline import PredictionPipeline
        print("✅ Prediction pipeline imported successfully")
        
        from api.server import APIServer
        print("✅ API server imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration settings"""
    print("\nTesting configuration...")
    
    try:
        from config.config import Config
        
        # Test basic config values
        assert Config.API_PORT == 5000, f"Expected port 5000, got {Config.API_PORT}"
        assert Config.DATA_INTERVAL_MINUTES == 5, "Data interval should be 5 minutes"
        assert Config.PREDICTION_HORIZON_MINUTES == 5, "Prediction horizon should be 5 minutes"
        
        print(f"✅ API Port: {Config.API_PORT}")
        print(f"✅ Data Interval: {Config.DATA_INTERVAL_MINUTES} minutes")
        print(f"✅ Prediction Horizon: {Config.PREDICTION_HORIZON_MINUTES} minutes")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_data_collection():
    """Test data collection functionality"""
    print("\nTesting data collection...")
    
    try:
        from data.collector import BTCDataCollector
        
        collector = BTCDataCollector()
        
        # Test current price fetch
        current_price = collector.get_current_price()
        
        if current_price and current_price > 0:
            print(f"✅ Current BTC price: ${current_price:,.2f}")
            return True
        else:
            print("❌ Failed to fetch current BTC price")
            return False
            
    except Exception as e:
        print(f"❌ Data collection test failed: {e}")
        return False

def test_directories():
    """Test that required directories can be created"""
    print("\nTesting directory creation...")
    
    try:
        from config.config import Config
        
        Config.create_directories()
        
        required_dirs = [
            Config.DATA_DIR,
            Config.MODELS_DIR,
            Config.LOGS_DIR,
            "models/ensemble",
            "models/individual"
        ]
        
        for directory in required_dirs:
            if os.path.exists(directory):
                print(f"✅ Directory exists: {directory}")
            else:
                print(f"❌ Directory missing: {directory}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Directory test failed: {e}")
        return False

def test_simple_predictor():
    """Test the simple predictor as a fallback"""
    print("\nTesting simple predictor...")
    
    try:
        from simple_predictor import SimpleBTCPredictor
        
        predictor = SimpleBTCPredictor()
        
        # Test price fetch
        if predictor.fetch_btc_price():
            print(f"✅ Simple predictor can fetch price: ${predictor.current_price:,.2f}")
            return True
        else:
            print("❌ Simple predictor failed to fetch price")
            return False
            
    except Exception as e:
        print(f"❌ Simple predictor test failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints (if server is running)"""
    print("\nTesting API endpoints...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        
        if response.status_code == 200:
            print("✅ API server is running and responding")
            return True
        else:
            print(f"❌ API server returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("⚠️  API server not running (this is normal for initial test)")
        return True  # This is expected if server isn't started yet
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("🧪 BTC PREDICTOR SYSTEM TEST")
    print("="*60)
    print(f"Test started at: {datetime.now()}")
    print()
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("Directory Test", test_directories),
        ("Data Collection Test", test_data_collection),
        ("Simple Predictor Test", test_simple_predictor),
        ("API Endpoints Test", test_api_endpoints),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready to start.")
        print("\nTo start the system:")
        print("1. Simple version: python simple_predictor.py")
        print("2. Full version: python main.py")
    elif passed >= total - 1:
        print("\n⚠️  Most tests passed. System should work with minor issues.")
        print("Try starting with: python simple_predictor.py")
    else:
        print("\n❌ Multiple test failures. Check dependencies and configuration.")
        print("Try installing requirements: pip install -r requirements.txt")
    
    print("="*60)

if __name__ == "__main__":
    main()