"""
Terminal Output System for Real vs Predicted Price Comparison
Displays current real price and 5-minute predictions on terminal
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import sys
from colorama import init, Fore, Back, Style
from .prediction_generator import PredictionGenerator
from .data_storage import DataStorage
from .config import Config

# Initialize colorama for cross-platform colored output
init(autoreset=True)

logger = logging.getLogger(__name__)

class TerminalDisplay:
    """Terminal display manager for BTC price predictions"""
    
    def __init__(self, config: Config):
        self.config = config
        self.prediction_generator = None
        self.data_storage = DataStorage(config)
        
        # Display state
        self.is_running = False
        self.display_mode = "startup"  # "startup", "predictions"
        self.last_update_time = None
        
        # Price tracking
        self.current_price = 0.0
        self.predicted_price = 0.0
        self.prediction_confidence = 0.0
        self.price_history = []
        self.prediction_history = []
        
        # Display settings
        self.update_interval = 5  # seconds
        self.show_startup_only = True  # Initially show only real prices
        
    def set_prediction_generator(self, prediction_generator: PredictionGenerator):
        """Set the prediction generator reference"""
        self.prediction_generator = prediction_generator
        
    async def start_display(self):
        """Start the terminal display"""
        try:
            logger.info("Starting terminal display...")
            
            self.is_running = True
            
            # Clear screen and show header
            self._clear_screen()
            self._show_header()
            
            # Main display loop
            while self.is_running:
                await self._update_display()
                await asyncio.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            logger.info("Terminal display interrupted by user")
        except Exception as e:
            logger.error(f"Terminal display error: {e}")
        finally:
            self.is_running = False
    
    async def _update_display(self):
        """Update the terminal display"""
        try:
            # Get current price data
            await self._fetch_current_data()
            
            # Check if we should start showing predictions
            if self.show_startup_only and self.prediction_generator and self.prediction_generator.is_initialized:
                # Wait for first prediction cycle to complete
                if self.prediction_generator.prediction_count > 0:
                    self.show_startup_only = False
                    self.display_mode = "predictions"
                    self._clear_screen()
                    self._show_header()
            
            # Update display based on mode
            if self.display_mode == "startup":
                self._show_startup_display()
            else:
                self._show_prediction_display()
                
            self.last_update_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Display update failed: {e}")
    
    async def _fetch_current_data(self):
        """Fetch current price and prediction data"""
        try:
            # Get current price
            recent_data = await self.data_storage.get_recent_price_data(hours=1)
            if recent_data:
                self.current_price = recent_data[-1].close_price
                
                # Update price history
                self.price_history.append({
                    'timestamp': recent_data[-1].timestamp,
                    'price': self.current_price
                })
                
                # Keep only last 20 entries
                if len(self.price_history) > 20:
                    self.price_history = self.price_history[-20:]
            
            # Get prediction data if available
            if self.prediction_generator and not self.show_startup_only:
                recent_predictions = self.prediction_generator.get_recent_predictions(1)
                if recent_predictions:
                    latest_pred = recent_predictions[0]
                    self.predicted_price = latest_pred.get('predicted_price', 0.0)
                    self.prediction_confidence = latest_pred.get('confidence_score', 0.0)
                    
                    # Update prediction history
                    self.prediction_history.append({
                        'timestamp': datetime.fromisoformat(latest_pred['timestamp']),
                        'predicted_price': self.predicted_price,
                        'current_price': latest_pred.get('current_price', 0.0),
                        'confidence': self.prediction_confidence
                    })
                    
                    # Keep only last 10 entries
                    if len(self.prediction_history) > 10:
                        self.prediction_history = self.prediction_history[-10:]
                        
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
    
    def _clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _show_header(self):
        """Show the application header"""
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}")
        print(f"{Fore.CYAN}{Style.BRIGHT}           BTC PRICE PREDICTION ENGINE - TERMINAL DISPLAY")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}")
        print()
        
        # Show current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Fore.WHITE}Current Time: {Style.BRIGHT}{current_time}")
        print()
    
    def _show_startup_display(self):
        """Show startup display with only real prices"""
        print(f"{Fore.YELLOW}{Style.BRIGHT}STARTUP MODE - Displaying Real Prices Only")
        print(f"{Fore.WHITE}Waiting for first prediction cycle to complete...")
        print()
        
        # Current price display
        if self.current_price > 0:
            print(f"{Fore.GREEN}{Style.BRIGHT}Current BTC Price:")
            print(f"{Fore.GREEN}{Style.BRIGHT}${self.current_price:,.2f}")
        else:
            print(f"{Fore.RED}No price data available")
        
        print()
        
        # Show recent price history
        if self.price_history:
            print(f"{Fore.CYAN}Recent Price History:")
            print(f"{Fore.CYAN}{'-'*50}")
            
            for i, entry in enumerate(self.price_history[-5:]):  # Last 5 entries
                timestamp = entry['timestamp'].strftime("%H:%M:%S")
                price = entry['price']
                
                # Color based on price change
                if i > 0:
                    prev_price = self.price_history[-(5-i+1)]['price'] if len(self.price_history) >= (5-i+1) else price
                    if price > prev_price:
                        color = Fore.GREEN
                        arrow = "↑"
                    elif price < prev_price:
                        color = Fore.RED
                        arrow = "↓"
                    else:
                        color = Fore.WHITE
                        arrow = "→"
                else:
                    color = Fore.WHITE
                    arrow = "→"
                
                print(f"{color}{timestamp} {arrow} ${price:,.2f}")
        
        print()
        
        # Status information
        if self.prediction_generator:
            if self.prediction_generator.is_initialized:
                print(f"{Fore.YELLOW}Status: Prediction engine initialized, waiting for first prediction...")
            else:
                print(f"{Fore.YELLOW}Status: Initializing prediction engine...")
        else:
            print(f"{Fore.RED}Status: Prediction engine not available")
    
    def _show_prediction_display(self):
        """Show full prediction display with real vs predicted prices"""
        print(f"{Fore.GREEN}{Style.BRIGHT}PREDICTION MODE - Real vs Predicted Prices")
        print()
        
        # Current price and prediction side by side
        print(f"{Fore.CYAN}{Style.BRIGHT}{'CURRENT PRICE':<25} {'5-MIN PREDICTION':<25} {'DIFFERENCE':<15}")
        print(f"{Fore.CYAN}{'-'*65}")
        
        if self.current_price > 0 and self.predicted_price > 0:
            difference = self.predicted_price - self.current_price
            diff_pct = (difference / self.current_price) * 100 if self.current_price > 0 else 0
            
            # Color coding for difference
            if abs(difference) <= 25:
                diff_color = Fore.GREEN  # Good prediction (within $25)
            elif abs(difference) <= 50:
                diff_color = Fore.YELLOW  # Moderate prediction (within $50)
            else:
                diff_color = Fore.RED  # Poor prediction (>$50)
            
            # Direction arrow
            if difference > 0:
                arrow = "↑"
                direction_color = Fore.GREEN
            elif difference < 0:
                arrow = "↓"
                direction_color = Fore.RED
            else:
                arrow = "→"
                direction_color = Fore.WHITE
            
            print(f"{Fore.WHITE}${self.current_price:,.2f}{'':<15} "
                  f"{Fore.CYAN}${self.predicted_price:,.2f}{'':<15} "
                  f"{diff_color}{direction_color}{arrow} ${abs(difference):,.2f} ({diff_pct:+.2f}%)")
            
            # Confidence display
            confidence_bar = self._create_confidence_bar(self.prediction_confidence)
            print(f"{Fore.WHITE}Confidence: {confidence_bar} {self.prediction_confidence:.1%}")
            
        else:
            print(f"{Fore.RED}No prediction data available")
        
        print()
        
        # Prediction accuracy summary
        if self.prediction_generator:
            stats = self.prediction_generator.get_prediction_stats()
            
            print(f"{Fore.CYAN}{Style.BRIGHT}PERFORMANCE SUMMARY")
            print(f"{Fore.CYAN}{'-'*40}")
            print(f"{Fore.WHITE}Total Predictions: {stats.get('prediction_count', 0)}")
            print(f"{Fore.WHITE}Current Accuracy: {stats.get('current_accuracy', 0.0):.1%}")
            
            if 'recent_performance' in stats:
                recent = stats['recent_performance']
                avg_error = recent.get('avg_error', 0.0)
                rapid_changes = recent.get('rapid_changes_detected', 0)
                
                # Color code average error
                if avg_error <= 25:
                    error_color = Fore.GREEN
                elif avg_error <= 50:
                    error_color = Fore.YELLOW
                else:
                    error_color = Fore.RED
                
                print(f"{Fore.WHITE}Average Error: {error_color}${avg_error:.2f}")
                print(f"{Fore.WHITE}Rapid Changes Detected: {Fore.YELLOW}{rapid_changes}")
        
        print()
        
        # Recent predictions table
        if self.prediction_history:
            print(f"{Fore.CYAN}{Style.BRIGHT}RECENT PREDICTIONS")
            print(f"{Fore.CYAN}{'-'*70}")
            print(f"{Fore.WHITE}{'Time':<8} {'Real':<10} {'Predicted':<10} {'Error':<8} {'Accuracy':<8}")
            print(f"{Fore.CYAN}{'-'*70}")
            
            for entry in self.prediction_history[-5:]:  # Last 5 predictions
                timestamp = entry['timestamp'].strftime("%H:%M")
                real_price = entry['current_price']
                pred_price = entry['predicted_price']
                error = abs(pred_price - real_price)
                accuracy = max(0, 1 - error / real_price) if real_price > 0 else 0
                
                # Color code accuracy
                if accuracy >= 0.95:  # >95% accuracy
                    acc_color = Fore.GREEN
                elif accuracy >= 0.90:  # >90% accuracy
                    acc_color = Fore.YELLOW
                else:
                    acc_color = Fore.RED
                
                print(f"{Fore.WHITE}{timestamp:<8} "
                      f"${real_price:,.0f}{'':<4} "
                      f"${pred_price:,.0f}{'':<4} "
                      f"${error:,.0f}{'':<4} "
                      f"{acc_color}{accuracy:.1%}")
        
        print()
        
        # Model contributions
        if self.prediction_generator and hasattr(self.prediction_generator.ensemble_model, 'model_weights'):
            weights = self.prediction_generator.ensemble_model.model_weights
            
            print(f"{Fore.CYAN}{Style.BRIGHT}MODEL CONTRIBUTIONS")
            print(f"{Fore.CYAN}{'-'*40}")
            
            for model_name, weight in weights.items():
                bar_length = int(weight * 20)  # Scale to 20 characters
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"{Fore.WHITE}{model_name:<10} {Fore.CYAN}{bar} {weight:.1%}")
        
        print()
        
        # Last update time
        if self.last_update_time:
            update_time = self.last_update_time.strftime("%H:%M:%S")
            print(f"{Fore.WHITE}Last Update: {update_time}")
    
    def _create_confidence_bar(self, confidence: float) -> str:
        """Create a visual confidence bar"""
        bar_length = 20
        filled_length = int(confidence * bar_length)
        
        # Color based on confidence level
        if confidence >= 0.8:
            color = Fore.GREEN
        elif confidence >= 0.6:
            color = Fore.YELLOW
        else:
            color = Fore.RED
        
        bar = color + "█" * filled_length + Fore.WHITE + "░" * (bar_length - filled_length)
        return bar
    
    def stop_display(self):
        """Stop the terminal display"""
        self.is_running = False
        logger.info("Terminal display stopped")
    
    def show_startup_message(self):
        """Show initial startup message"""
        self._clear_screen()
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}")
        print(f"{Fore.CYAN}{Style.BRIGHT}           BTC PRICE PREDICTION ENGINE")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}")
        print()
        print(f"{Fore.YELLOW}Initializing system...")
        print(f"{Fore.WHITE}• Loading configuration")
        print(f"{Fore.WHITE}• Connecting to data sources")
        print(f"{Fore.WHITE}• Training ensemble models")
        print(f"{Fore.WHITE}• Starting prediction engine")
        print()
        print(f"{Fore.GREEN}At startup, only real BTC prices will be displayed.")
        print(f"{Fore.GREEN}Predictions will appear after the first prediction cycle completes.")
        print()
        print(f"{Fore.CYAN}Press Ctrl+C to stop the application")
        print()

class TerminalOutputManager:
    """Manager for terminal output system"""
    
    def __init__(self, config: Config):
        self.config = config
        self.terminal_display = TerminalDisplay(config)
        self.is_running = False
        
    async def start(self, prediction_generator: PredictionGenerator):
        """Start the terminal output system"""
        try:
            # Set prediction generator reference
            self.terminal_display.set_prediction_generator(prediction_generator)
            
            # Show startup message
            self.terminal_display.show_startup_message()
            
            # Wait a moment for user to read
            await asyncio.sleep(3)
            
            # Start display
            self.is_running = True
            await self.terminal_display.start_display()
            
        except Exception as e:
            logger.error(f"Terminal output manager failed: {e}")
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the terminal output system"""
        self.is_running = False
        self.terminal_display.stop_display()
        
        # Show goodbye message
        print(f"\n{Fore.CYAN}{Style.BRIGHT}BTC Prediction Engine stopped.")
        print(f"{Fore.WHITE}Thank you for using the system!")
        print()

# Factory function
def create_terminal_output(config: Config = None) -> TerminalOutputManager:
    """Create terminal output manager"""
    if config is None:
        config = Config()
    
    return TerminalOutputManager(config)