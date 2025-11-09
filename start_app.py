"""
Startup script for BTC Price Prediction App
Uses: LSTM, LightGBM, Random Forest, and Kalman Filter
"""
import sys
import os
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>")
logger.add("logs/btc_predictor.log", rotation="100 MB", retention="7 days", level="DEBUG")

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from services.prediction_pipeline import PredictionPipeline

def main():
    """Main entry point for the BTC prediction application."""
    try:
        logger.info("=" * 80)
        logger.info("BTC Price Prediction System Starting")
        logger.info("Models: LSTM, LightGBM, Random Forest, Kalman Filter")
        logger.info("=" * 80)
        
        # Create necessary directories
        Config.create_directories()
        logger.info("Directories created/verified")
        
        # Initialize prediction pipeline
        logger.info("Initializing prediction pipeline...")
        pipeline = PredictionPipeline()
        
        # Start the pipeline
        logger.info("Starting automated prediction pipeline...")
        pipeline.start_pipeline()
        
        logger.info("=" * 80)
        logger.info("System is now running!")
        logger.info(f"- Predictions every {Config.DATA_INTERVAL_MINUTES} minutes")
        logger.info(f"- Model retraining every {Config.RETRAIN_INTERVAL_HOURS} hours")
        logger.info("- Press Ctrl+C to stop")
        logger.info("=" * 80)
        
        # Keep the main thread alive
        import time
        while True:
            time.sleep(60)
            
            # Print status every 5 minutes
            status = pipeline.get_pipeline_status()
            if status.get('is_running'):
                logger.info(f"Status: Running | Predictions: {status['metrics']['successful_predictions']} | Last: {status.get('last_prediction_time', 'N/A')}")
        
    except KeyboardInterrupt:
        logger.info("\nShutdown signal received...")
        if 'pipeline' in locals():
            pipeline.stop_pipeline()
        logger.info("Application stopped successfully")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
