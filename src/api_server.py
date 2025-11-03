"""
FastAPI Server for BTC Prediction Engine
Provides REST endpoints and WebSocket support for real-time data streaming
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from .prediction_generator import PredictionGenerator
from .data_storage import DataStorage
from .data_collector import BTCDataCollector
from .config import Config

logger = logging.getLogger(__name__)

class BTCPredictionAPI:
    def __init__(self, config: Config):
        self.config = config
        self.app = FastAPI(title="BTC Price Prediction API")
        self._setup_middleware()
        self._setup_routes()
        self.prediction_generator = None

    def _setup_middleware(self):
        """Configure CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Configure API routes"""
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy"}
        
        @self.app.get("/predict")
        async def get_prediction():
            if not self.prediction_generator:
                raise HTTPException(status_code=503, detail="Prediction service not initialized")
            try:
                prediction = await self.prediction_generator.get_latest_prediction()
                return prediction
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    if self.prediction_generator:
                        prediction = await self.prediction_generator.get_latest_prediction()
                        await websocket.send_json(prediction)
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")

    def get_app(self):
        """Get the FastAPI application instance"""
        return self.app

# Pydantic models for API
class PredictionResponse(BaseModel):
    timestamp: str
    current_price: float
    predicted_price: float
    confidence_score: float
    prediction_horizon: int
    rapid_change_detected: bool = False
    rapid_change_magnitude: float = 0.0
    rapid_change_direction: str = "stable"
    model_contributions: Dict[str, float] = {}

class HistoricalDataResponse(BaseModel):
    timestamp: str
    open_price: float

async def initialize(self):
    """Initialize the API server"""
    logger.info("Initializing BTC Prediction API server...")
    try:
        self.prediction_generator = PredictionGenerator(self.config)
        await self.prediction_generator.initialize()
        logger.info("BTC Prediction API server initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize API server: {e}")
        raise

async def shutdown(self):
    """Shutdown the API server"""
    logger.info("Shutting down BTC Prediction API server...")
    if self.prediction_generator:
        await self.prediction_generator.cleanup()
    logger.info("BTC Prediction API server shutdown complete")

def create_api_server(config: Config) -> BTCPredictionAPI:
    """Create and configure the API server"""
    api = BTCPredictionAPI(config)
    
    @api.app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    @api.app.get("/predict")
    async def get_prediction():
        if not api.prediction_generator:
            raise HTTPException(status_code=503, detail="Prediction service not initialized")
        try:
            prediction = await api.prediction_generator.get_latest_prediction()
            return prediction
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @api.app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                if api.prediction_generator:
                    prediction = await api.prediction_generator.get_latest_prediction()
                    await websocket.send_json(prediction)
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
            
    return api
    high_price: float
    low_price: float
    close_price: float
    volume: float

class PerformanceResponse(BaseModel):
    model_name: str
    timestamp: str
    mae: float
    rmse: float
    accuracy_within_threshold: float
    rapid_change_detection_rate: float

class SystemStatusResponse(BaseModel):
    status: str
    uptime: str
    prediction_count: int
    current_accuracy: float
    last_prediction_time: Optional[str]
    models_trained: int

class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
        
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
            
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

class BTCPredictionAPI:
    """Main API server class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.app = FastAPI(
            title="BTC Price Prediction API",
            description="Real-time Bitcoin price prediction using ensemble machine learning",
            version="1.0.0"
        )
        
        # Initialize components
        self.prediction_generator = PredictionGenerator(config)
        self.data_storage = DataStorage(config)
        self.data_collector = BTCDataCollector(config)
        self.websocket_manager = WebSocketManager()
        
        # Server state
        self.start_time = datetime.now()
        self.is_running = False
        self.background_tasks_running = False
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize server on startup"""
            await self.initialize()
            
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown"""
            await self.shutdown()
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {"message": "BTC Price Prediction API", "status": "running", "version": "1.0.0"}
        
        @self.app.get("/api/current-price")
        async def get_current_price():
            """Get current BTC price"""
            try:
                # Get latest price from data storage
                recent_data = await self.data_storage.get_recent_price_data(hours=1)
                
                if not recent_data:
                    raise HTTPException(status_code=404, detail="No recent price data available")
                
                latest_price = recent_data[-1]
                
                return {
                    "timestamp": latest_price.timestamp.isoformat(),
                    "price": latest_price.close_price,
                    "open": latest_price.open_price,
                    "high": latest_price.high_price,
                    "low": latest_price.low_price,
                    "volume": latest_price.volume
                }
                
            except Exception as e:
                logger.error(f"Current price endpoint failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/prediction", response_model=PredictionResponse)
        async def get_prediction():
            """Get current 5-minute price prediction"""
            try:
                if not self.prediction_generator.is_initialized:
                    raise HTTPException(status_code=503, detail="Prediction generator not initialized")
                
                # Generate new prediction
                prediction = await self.prediction_generator.generate_prediction()
                
                if not prediction:
                    raise HTTPException(status_code=500, detail="Failed to generate prediction")
                
                response = PredictionResponse(
                    timestamp=prediction.timestamp.isoformat(),
                    current_price=prediction.current_price,
                    predicted_price=prediction.predicted_price,
                    confidence_score=prediction.confidence_score,
                    prediction_horizon=prediction.prediction_horizon,
                    rapid_change_detected=getattr(prediction, 'rapid_change_detected', False),
                    rapid_change_magnitude=getattr(prediction, 'rapid_change_magnitude', 0.0),
                    rapid_change_direction=getattr(prediction, 'rapid_change_direction', 'stable'),
                    model_contributions=prediction.model_contributions
                )
                
                return response
                
            except Exception as e:
                logger.error(f"Prediction endpoint failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/history")
        async def get_history(hours: int = 5):
            """Get historical price data"""
            try:
                if hours > 24:
                    hours = 24  # Limit to 24 hours
                
                historical_data = await self.data_storage.get_recent_price_data(hours=hours)
                
                if not historical_data:
                    raise HTTPException(status_code=404, detail="No historical data available")
                
                response_data = []
                for data_point in historical_data:
                    response_data.append(HistoricalDataResponse(
                        timestamp=data_point.timestamp.isoformat(),
                        open_price=data_point.open_price,
                        high_price=data_point.high_price,
                        low_price=data_point.low_price,
                        close_price=data_point.close_price,
                        volume=data_point.volume
                    ))
                
                return response_data
                
            except Exception as e:
                logger.error(f"History endpoint failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/predictions")
        async def get_recent_predictions(count: int = 100):
            """Get recent predictions"""
            try:
                if count > 1000:
                    count = 1000  # Limit to 1000 predictions
                
                predictions = self.prediction_generator.get_recent_predictions(count)
                
                return {
                    "predictions": predictions,
                    "count": len(predictions)
                }
                
            except Exception as e:
                logger.error(f"Recent predictions endpoint failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/performance")
        async def get_performance():
            """Get model performance metrics"""
            try:
                stats = self.prediction_generator.get_prediction_stats()
                
                return {
                    "current_accuracy": stats.get('current_accuracy', 0.0),
                    "prediction_count": stats.get('prediction_count', 0),
                    "recent_performance": stats.get('recent_performance', {}),
                    "ensemble_info": stats.get('ensemble_info', {}),
                    "rapid_change_threshold": stats.get('rapid_change_threshold', 50.0)
                }
                
            except Exception as e:
                logger.error(f"Performance endpoint failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/status", response_model=SystemStatusResponse)
        async def get_system_status():
            """Get system status"""
            try:
                uptime = datetime.now() - self.start_time
                stats = self.prediction_generator.get_prediction_stats()
                
                # Count trained models
                models_trained = 0
                if 'ensemble_info' in stats:
                    models_trained = stats['ensemble_info'].get('base_models_trained', 0)
                
                return SystemStatusResponse(
                    status="running" if self.is_running else "stopped",
                    uptime=str(uptime),
                    prediction_count=stats.get('prediction_count', 0),
                    current_accuracy=stats.get('current_accuracy', 0.0),
                    last_prediction_time=stats.get('last_prediction_time'),
                    models_trained=models_trained
                )
                
            except Exception as e:
                logger.error(f"Status endpoint failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/diagnostics")
        async def get_diagnostics():
            """Get detailed system diagnostics"""
            try:
                diagnostics = self.prediction_generator.get_model_diagnostics()
                
                # Add server diagnostics
                diagnostics['server'] = {
                    'uptime': str(datetime.now() - self.start_time),
                    'is_running': self.is_running,
                    'background_tasks_running': self.background_tasks_running,
                    'websocket_connections': len(self.websocket_manager.active_connections)
                }
                
                return diagnostics
                
            except Exception as e:
                logger.error(f"Diagnostics endpoint failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/retrain")
        async def trigger_retrain(background_tasks: BackgroundTasks):
            """Trigger model retraining"""
            try:
                background_tasks.add_task(self._retrain_models)
                
                return {
                    "message": "Model retraining started",
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Retrain endpoint failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await self.websocket_manager.connect(websocket)
            
            try:
                while True:
                    # Keep connection alive and handle client messages
                    data = await websocket.receive_text()
                    
                    # Echo back or handle specific commands
                    message = json.loads(data)
                    if message.get('type') == 'ping':
                        await websocket.send_text(json.dumps({
                            'type': 'pong',
                            'timestamp': datetime.now().isoformat()
                        }))
                    
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.websocket_manager.disconnect(websocket)
    
    async def initialize(self):
        """Initialize the API server"""
        try:
            logger.info("Initializing BTC Prediction API server...")
            
            # Initialize prediction generator
            success = await self.prediction_generator.initialize()
            
            if not success:
                logger.error("Failed to initialize prediction generator")
                return False
            
            # Start background tasks
            asyncio.create_task(self._background_prediction_loop())
            asyncio.create_task(self._background_data_collection())
            asyncio.create_task(self._background_retraining_schedule())
            
            self.is_running = True
            self.background_tasks_running = True
            
            logger.info("BTC Prediction API server initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"API server initialization failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the API server"""
        try:
            logger.info("Shutting down BTC Prediction API server...")
            
            self.is_running = False
            self.background_tasks_running = False
            
            # Cleanup
            await self.prediction_generator.cleanup_cache()
            
            logger.info("BTC Prediction API server shutdown complete")
            
        except Exception as e:
            logger.error(f"API server shutdown failed: {e}")
    
    async def _background_prediction_loop(self):
        """Background task for generating predictions every 5 minutes"""
        logger.info("Starting background prediction loop...")
        
        while self.background_tasks_running:
            try:
                # Generate prediction
                prediction = await self.prediction_generator.generate_prediction()
                
                if prediction:
                    # Broadcast to WebSocket clients
                    prediction_data = {
                        'type': 'prediction',
                        'timestamp': prediction.timestamp.isoformat(),
                        'current_price': prediction.current_price,
                        'predicted_price': prediction.predicted_price,
                        'confidence_score': prediction.confidence_score,
                        'rapid_change_detected': getattr(prediction, 'rapid_change_detected', False),
                        'rapid_change_magnitude': getattr(prediction, 'rapid_change_magnitude', 0.0),
                        'rapid_change_direction': getattr(prediction, 'rapid_change_direction', 'stable')
                    }
                    
                    await self.websocket_manager.broadcast(prediction_data)
                
                # Wait for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Background prediction loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _background_data_collection(self):
        """Background task for collecting BTC data"""
        logger.info("Starting background data collection...")
        
        while self.background_tasks_running:
            try:
                # Collect current price data
                async with self.data_collector as collector:
                    price_data = await collector.fetch_current_price()
                    
                    if price_data:
                        # Store the data
                        await self.data_storage.store_price_data(price_data)
                        
                        # Broadcast current price to WebSocket clients
                        price_update = {
                            'type': 'price_update',
                            'timestamp': datetime.now().isoformat(),
                            'price': price_data['close_price'],
                            'volume': price_data['volume']
                        }
                        
                        await self.websocket_manager.broadcast(price_update)
                
                # Wait for 1 minute (more frequent than predictions)
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Background data collection error: {e}")
                await asyncio.sleep(30)  # Wait 30 seconds before retry
    
    async def _background_retraining_schedule(self):
        """Background task for scheduled model retraining"""
        logger.info("Starting background retraining scheduler...")
        
        while self.background_tasks_running:
            try:
                # Wait for 24 hours
                await asyncio.sleep(24 * 60 * 60)
                
                if self.background_tasks_running:
                    logger.info("Starting scheduled model retraining...")
                    await self._retrain_models()
                
            except Exception as e:
                logger.error(f"Background retraining scheduler error: {e}")
                await asyncio.sleep(60 * 60)  # Wait 1 hour before retry
    
    async def _retrain_models(self):
        """Retrain models with recent data"""
        try:
            logger.info("Starting model retraining...")
            
            success = await self.prediction_generator.retrain_models()
            
            if success:
                logger.info("Model retraining completed successfully")
                
                # Broadcast retraining completion
                retrain_message = {
                    'type': 'retrain_complete',
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
                
                await self.websocket_manager.broadcast(retrain_message)
            else:
                logger.error("Model retraining failed")
                
        except Exception as e:
            logger.error(f"Model retraining error: {e}")

    def get_app(self):
        """Return the underlying FastAPI application instance."""
        return self.app
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API server"""
        logger.info(f"Starting BTC Prediction API server on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )

# Factory function to create API server
def create_api_server(config: Config = None) -> BTCPredictionAPI:
    """Create and return API server instance"""
    if config is None:
        config = Config()
    
    return BTCPredictionAPI(config)