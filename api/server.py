"""REST API server for BTC prediction system."""
from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from loguru import logger
import threading
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from data.manager import BTCDataManager
from services.prediction_pipeline import PredictionPipeline
from services.terminal_monitor import TerminalMonitor
from services.backup_recovery import BackupRecoverySystem

class APIServer:
    """REST API server for accessing BTC prediction data."""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for browser access
        
        # Core components
        self.data_manager = None
        self.prediction_pipeline = None
        self.terminal_monitor = None
        self.backup_system = BackupRecoverySystem()
        
        # Server state
        self.is_running = False
        self.server_thread = None
        
        # Setup routes
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes."""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            })
        
        @self.app.route('/api/current-price', methods=['GET'])
        def get_current_price():
            """Get current BTC price."""
            try:
                if not self.data_manager:
                    return jsonify({'error': 'Data manager not initialized'}), 500
                
                current_price = self.data_manager.get_current_price()
                
                if current_price is None:
                    return jsonify({'error': 'Price data not available'}), 503
                
                return jsonify({
                    'price': current_price,
                    'timestamp': datetime.now().isoformat(),
                    'currency': 'USD'
                })
                
            except Exception as e:
                logger.error(f"Current price API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/latest-prediction', methods=['GET'])
        def get_latest_prediction():
            """Get most recent 5-minute prediction."""
            try:
                if not self.prediction_pipeline:
                    return jsonify({'error': 'Prediction pipeline not initialized'}), 500
                
                if len(self.prediction_pipeline.prediction_history) == 0:
                    return jsonify({'error': 'No predictions available'}), 404
                
                latest_prediction = self.prediction_pipeline.prediction_history[-1]
                
                # Format response
                response = {
                    'timestamp': latest_prediction['timestamp'].isoformat(),
                    'predicted_price': latest_prediction['corrected_prediction'],
                    'raw_prediction': latest_prediction['raw_prediction'],
                    'confidence_interval': {
                        'lower': latest_prediction['confidence_interval'][0],
                        'upper': latest_prediction['confidence_interval'][1]
                    },
                    'current_price': latest_prediction['current_price'],
                    'rapid_movement_detected': latest_prediction['rapid_movement_detected'],
                    'market_volatility': latest_prediction['market_volatility'],
                    'model_contributions': latest_prediction['corrected_contributions']
                }
                
                # Add accuracy if available
                if 'actual_price' in latest_prediction:
                    response['actual_price'] = latest_prediction['actual_price']
                    response['error'] = latest_prediction['error']
                    response['error_percentage'] = latest_prediction['error_percentage']
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Latest prediction API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/prediction-history', methods=['GET'])
        def get_prediction_history():
            """Get historical predictions with filtering."""
            try:
                if not self.prediction_pipeline:
                    return jsonify({'error': 'Prediction pipeline not initialized'}), 500
                
                # Get query parameters
                hours = request.args.get('hours', default=24, type=int)
                limit = request.args.get('limit', default=100, type=int)
                
                # Filter predictions
                cutoff_time = datetime.now() - timedelta(hours=hours)
                filtered_predictions = [
                    p for p in self.prediction_pipeline.prediction_history
                    if p['timestamp'] >= cutoff_time
                ]
                
                # Apply limit
                if limit > 0:
                    filtered_predictions = filtered_predictions[-limit:]
                
                # Format response
                predictions = []
                for pred in filtered_predictions:
                    pred_data = {
                        'timestamp': pred['timestamp'].isoformat(),
                        'predicted_price': pred['corrected_prediction'],
                        'current_price': pred['current_price'],
                        'confidence_interval': {
                            'lower': pred['confidence_interval'][0],
                            'upper': pred['confidence_interval'][1]
                        },
                        'rapid_movement_detected': pred['rapid_movement_detected']
                    }
                    
                    # Add accuracy if available
                    if 'actual_price' in pred:
                        pred_data['actual_price'] = pred['actual_price']
                        pred_data['error_percentage'] = pred['error_percentage']
                    
                    predictions.append(pred_data)
                
                return jsonify({
                    'predictions': predictions,
                    'count': len(predictions),
                    'hours_requested': hours,
                    'limit_applied': limit
                })
                
            except Exception as e:
                logger.error(f"Prediction history API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/model-performance', methods=['GET'])
        def get_model_performance():
            """Get individual model metrics."""
            try:
                if not self.prediction_pipeline:
                    return jsonify({'error': 'Prediction pipeline not initialized'}), 500
                
                # Get performance metrics from offset correction system
                accuracy_metrics = self.prediction_pipeline.offset_correction.calculate_accuracy_metrics()
                
                # Get model info from ensemble
                model_info = self.prediction_pipeline.ensemble_model.get_model_info()
                
                response = {
                    'ensemble_info': {
                        'is_trained': model_info['is_trained'],
                        'weights': model_info['weights'],
                        'rapid_movement_detected': model_info['rapid_movement_detected']
                    },
                    'individual_models': model_info['individual_models'],
                    'accuracy_metrics': accuracy_metrics,
                    'last_updated': datetime.now().isoformat()
                }
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Model performance API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/system-status', methods=['GET'])
        def get_system_status():
            """Get comprehensive system status."""
            try:
                status = {
                    'api_server': {
                        'status': 'running',
                        'timestamp': datetime.now().isoformat()
                    }
                }
                
                # Pipeline status
                if self.prediction_pipeline:
                    status['prediction_pipeline'] = self.prediction_pipeline.get_pipeline_status()
                
                # Data manager status
                if self.data_manager:
                    status['data_manager'] = self.data_manager.get_system_status()
                
                # Terminal monitor status
                if self.terminal_monitor:
                    status['terminal_monitor'] = self.terminal_monitor.get_monitor_status()
                
                return jsonify(status)
                
            except Exception as e:
                logger.error(f"System status API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/retrain-models', methods=['POST'])
        def retrain_models():
            """Trigger manual model retraining."""
            try:
                if not self.prediction_pipeline:
                    return jsonify({'error': 'Prediction pipeline not initialized'}), 500
                
                # Trigger retraining in background
                def retrain_task():
                    success = self.prediction_pipeline.train_models(force_retrain=True)
                    logger.info(f"Manual retraining completed: {'success' if success else 'failed'}")
                
                retrain_thread = threading.Thread(target=retrain_task, daemon=True)
                retrain_thread.start()
                
                return jsonify({
                    'message': 'Model retraining initiated',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Retrain models API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/historical-data', methods=['GET'])
        def get_historical_data():
            """Get historical price data."""
            try:
                if not self.data_manager:
                    return jsonify({'error': 'Data manager not initialized'}), 500
                
                # Get query parameters
                hours = request.args.get('hours', default=5, type=int)
                
                # Calculate number of records (5-minute intervals)
                records_needed = (hours * 60) // 5
                
                # Get historical data
                historical_data = self.data_manager.get_latest_data(n_records=records_needed)
                
                if historical_data is None or len(historical_data) == 0:
                    return jsonify({'error': 'No historical data available'}), 404
                
                # Format response
                data_points = []
                for _, row in historical_data.iterrows():
                    data_points.append({
                        'timestamp': row['timestamp'].isoformat(),
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    })
                
                return jsonify({
                    'data': data_points,
                    'count': len(data_points),
                    'hours_requested': hours,
                    'interval_minutes': 5
                })
                
            except Exception as e:
                logger.error(f"Historical data API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/backup', methods=['POST'])
        def create_backup():
            """Create system backup."""
            try:
                success = self.backup_system.create_backup()
                
                if success:
                    return jsonify({
                        'message': 'Backup created successfully',
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({'error': 'Backup creation failed'}), 500
                    
            except Exception as e:
                logger.error(f"Backup API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/backups', methods=['GET'])
        def list_backups():
            """List available backups."""
            try:
                backups = self.backup_system.list_backups()
                
                return jsonify({
                    'backups': backups,
                    'count': len(backups)
                })
                
            except Exception as e:
                logger.error(f"List backups API error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def initialize_components(self, data_manager: BTCDataManager, 
                            prediction_pipeline: PredictionPipeline,
                            terminal_monitor: TerminalMonitor):
        """
        Initialize API server with core components.
        
        Args:
            data_manager: Data manager instance
            prediction_pipeline: Prediction pipeline instance
            terminal_monitor: Terminal monitor instance
        """
        self.data_manager = data_manager
        self.prediction_pipeline = prediction_pipeline
        self.terminal_monitor = terminal_monitor
        
        logger.info("API server components initialized")
    
    def start_server(self, host: str = None, port: int = None, debug: bool = False):
        """
        Start the API server.
        
        Args:
            host: Host address (default from config)
            port: Port number (default from config)
            debug: Enable debug mode
        """
        if self.is_running:
            logger.warning("API server already running")
            return
        
        host = host or Config.API_HOST
        port = port or Config.API_PORT
        
        logger.info(f"Starting API server on {host}:{port}")
        
        def run_server():
            try:
                self.app.run(
                    host=host,
                    port=port,
                    debug=debug,
                    use_reloader=False,  # Disable reloader to avoid issues
                    threaded=True
                )
            except Exception as e:
                logger.error(f"API server error: {e}")
                self.is_running = False
        
        self.is_running = True
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # Give server time to start
        time.sleep(1)
        
        logger.info(f"API server started at http://{host}:{port}")
        logger.info("Available endpoints:")
        logger.info("  GET  /api/health - Health check")
        logger.info("  GET  /api/current-price - Current BTC price")
        logger.info("  GET  /api/latest-prediction - Latest prediction")
        logger.info("  GET  /api/prediction-history - Prediction history")
        logger.info("  GET  /api/model-performance - Model metrics")
        logger.info("  GET  /api/system-status - System status")
        logger.info("  POST /api/retrain-models - Trigger retraining")
        logger.info("  GET  /api/historical-data - Historical price data")
        logger.info("  POST /api/backup - Create backup")
        logger.info("  GET  /api/backups - List backups")
    
    def stop_server(self):
        """Stop the API server."""
        if not self.is_running:
            logger.warning("API server not running")
            return
        
        logger.info("Stopping API server...")
        self.is_running = False
        
        # Note: Flask development server doesn't have a clean shutdown method
        # In production, you would use a proper WSGI server like Gunicorn
        
        logger.info("API server stopped")
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        Get API server information.
        
        Returns:
            Dictionary with server info
        """
        return {
            'is_running': self.is_running,
            'host': Config.API_HOST,
            'port': Config.API_PORT,
            'endpoints_count': len(self.app.url_map._rules),
            'components_initialized': all([
                self.data_manager is not None,
                self.prediction_pipeline is not None,
                self.terminal_monitor is not None
            ])
        }