#!/usr/bin/env python3
"""
BTC Price Prediction Engine
Main entry point for the prediction system running on Ubuntu VPS
"""

import asyncio
import logging
import argparse
import sys
from datetime import datetime
from src.prediction_generator import PredictionGenerator
from src.api_server import create_api_server
from src.terminal_output import create_terminal_output
from src.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/prediction_engine.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PredictionEngine:
    def __init__(self, mode: str = "full"):
        self.config = Config()
        self.mode = mode  # "full", "api", "terminal"
        
        # Initialize components based on mode
        self.prediction_generator = PredictionGenerator(self.config)
        
        if mode in ["full", "api"]:
            self.api_server = create_api_server(self.config)
        else:
            self.api_server = None
            
        if mode in ["full", "terminal"]:
            self.terminal_output = create_terminal_output(self.config)
        else:
            self.terminal_output = None
        
        self.running = False
        
    async def start(self):
        """Start the prediction engine"""
        logger.info(f"Starting BTC Prediction Engine in {self.mode} mode...")
        self.running = True
        
        try:
            # Initialize prediction generator
            success = await self.prediction_generator.initialize()
            if not success:
                logger.error("Failed to initialize prediction generator")
                return False
            
            # Start components based on mode
            if self.mode == "full":
                # Start both API server and terminal output
                await self._start_full_mode()
            elif self.mode == "api":
                # Start only API server
                await self._start_api_mode()
            elif self.mode == "terminal":
                # Start only terminal output
                await self._start_terminal_mode()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start prediction engine: {e}")
            return False
    
    async def _start_full_mode(self):
        """Start full mode with both API and terminal"""
        logger.info("Starting full mode (API + Terminal)")
        
        # Start API server in background
        if self.api_server:
            api_task = asyncio.create_task(self._run_api_server())
        
        # Start terminal output (this will block until stopped)
        if self.terminal_output:
            await self.terminal_output.start(self.prediction_generator)
    
    async def _start_api_mode(self):
        """Start API-only mode"""
        logger.info("Starting API-only mode")
        
        if self.api_server:
            await self._run_api_server()
    
    async def _start_terminal_mode(self):
        """Start terminal-only mode"""
        logger.info("Starting terminal-only mode")
        
        if self.terminal_output:
            await self.terminal_output.start(self.prediction_generator)
    
    async def _run_api_server(self):
        """Run the API server"""
        try:
            # Initialize API server
            await self.api_server.initialize()
            
            # Keep running until stopped
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"API server error: {e}")
    
    async def stop(self):
        """Stop the prediction engine"""
        logger.info("Stopping BTC Prediction Engine...")
        self.running = False
        
        # Stop components
        if self.api_server:
            await self.api_server.shutdown()
        
        if self.terminal_output:
            self.terminal_output.stop()

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='BTC Price Prediction Engine')
    parser.add_argument(
        '--mode', 
        choices=['full', 'api', 'terminal'], 
        default='full',
        help='Run mode: full (API + terminal), api (API only), terminal (terminal only)'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='API server host (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='API server port (default: 8000)'
    )
    
    args = parser.parse_args()
    
    # Create and start engine
    engine = PredictionEngine(mode=args.mode)
    
    try:
        if args.mode == "api":
            # For API-only mode, use uvicorn directly
            logger.info(f"Starting API server on {args.host}:{args.port}")
            engine.api_server.run(host=args.host, port=args.port)
        else:
            # For other modes, use async startup
            await engine.start()
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())