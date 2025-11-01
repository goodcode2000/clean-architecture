"""Main entry point for BTC Predictor application."""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from loguru import logger

def setup_logging():
    """Configure logging for the application."""
    Config.create_directories()
    
    logger.remove()  # Remove default handler
    logger.add(
        f"{Config.LOGS_DIR}/btc_predictor.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
    )

def main():
    """Main application entry point."""
    setup_logging()
    logger.info("Starting BTC Predictor Application")
    
    try:
        # Create necessary directories
        Config.create_directories()
        logger.info("Project directories created successfully")
        
        # Initialize core components
        from data.manager import BTCDataManager
        from services.prediction_pipeline import PredictionPipeline
        from services.terminal_monitor import TerminalMonitor
        from api.server import APIServer
        
        logger.info("Initializing BTC Prediction System components...")
        
        # Initialize data manager
        data_manager = BTCDataManager()
        if not data_manager.initialize_data():
            logger.error("Failed to initialize data system")
            sys.exit(1)
        
        # Initialize prediction pipeline
        prediction_pipeline = PredictionPipeline()
        
        # Initialize terminal monitor
        terminal_monitor = TerminalMonitor()
        
        # Initialize API server
        api_server = APIServer()
        api_server.initialize_components(data_manager, prediction_pipeline, terminal_monitor)
        
        logger.info("All components initialized successfully")
        
        # Get initial price to verify system
        current_price = data_manager.get_current_price()
        if current_price:
            logger.info(f"Current BTC Price: ${current_price:,.2f}")
            terminal_monitor.update_data(current_price=current_price)
        else:
            logger.warning("Could not retrieve current BTC price")
        
        # Start all services
        logger.info("Starting BTC Prediction System services...")
        
        # Start prediction pipeline (includes data updates)
        prediction_pipeline.start_pipeline()
        
        # Start terminal monitoring
        terminal_monitor.start_monitoring()
        
        # Start API server
        api_server.start_server()
        
        logger.info("="*60)
        logger.info("🚀 BTC PREDICTION SYSTEM FULLY OPERATIONAL 🚀")
        logger.info("="*60)
        logger.info("✅ Data Collection: Running (5-minute updates)")
        logger.info("✅ ML Models: Training and predicting")
        logger.info("✅ Terminal Monitor: Real-time display active")
        logger.info(f"✅ API Server: http://{Config.API_HOST}:{Config.API_PORT}")
        logger.info("✅ Offset Correction: Learning from predictions")
        logger.info("="*60)
        logger.info("Press Ctrl+C to stop the system")
        
        # Keep the application running
        try:
            while True:
                time.sleep(30)  # Check every 30 seconds
                
                # Update terminal monitor with latest data
                current_price = data_manager.get_current_price()
                if current_price:
                    terminal_monitor.update_data(current_price=current_price)
                
                # Get latest prediction if available
                if len(prediction_pipeline.prediction_history) > 0:
                    latest_prediction = prediction_pipeline.prediction_history[-1]
                    terminal_monitor.update_data(prediction_data=latest_prediction)
                
                # Update system status
                system_status = prediction_pipeline.get_pipeline_status()
                terminal_monitor.update_data(system_status=system_status)
                
        except KeyboardInterrupt:
            logger.info("Shutting down BTC Prediction System...")
            
            # Stop all services gracefully
            terminal_monitor.stop_monitoring()
            prediction_pipeline.stop_pipeline()
            api_server.stop_server()
            data_manager.cleanup()
            
            logger.info("BTC Prediction System stopped successfully")
            logger.info("Thank you for using BTC Predictor! 🎯")
        
    except Exception as e:
        logger.error(f"Failed to start BTC Predictor: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()