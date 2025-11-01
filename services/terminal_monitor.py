"""Terminal output and monitoring system for BTC price predictions."""
import os
import sys
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger
import colorama
from colorama import Fore, Back, Style

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

# Initialize colorama for cross-platform colored output
colorama.init()

class TerminalMonitor:
    """Terminal-based monitoring and display system for BTC predictions."""
    
    def __init__(self):
        self.is_running = False
        self.display_thread = None
        self.update_interval = 30  # Update display every 30 seconds
        self.startup_mode = True  # Show only real price until first prediction
        
        # Display data
        self.current_price = None
        self.last_prediction = None
        self.prediction_history = []
        self.system_status = {}
        self.model_performance = {}
        
        # Display settings
        self.show_detailed_info = True
        self.show_model_contributions = True
        self.max_history_display = 10
        
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def format_price(self, price: float) -> str:
        """Format price with color coding."""
        if price is None:
            return "N/A"
        
        return f"${price:,.2f}"
    
    def format_change(self, current: float, previous: float) -> str:
        """Format price change with color coding."""
        if current is None or previous is None:
            return "N/A"
        
        change = current - previous
        change_pct = (change / previous) * 100 if previous != 0 else 0
        
        if change > 0:
            color = Fore.GREEN
            symbol = "↑"
        elif change < 0:
            color = Fore.RED
            symbol = "↓"
        else:
            color = Fore.YELLOW
            symbol = "→"
        
        return f"{color}{symbol} ${change:+.2f} ({change_pct:+.2f}%){Style.RESET_ALL}"
    
    def format_accuracy(self, accuracy: float) -> str:
        """Format accuracy with color coding."""
        if accuracy is None:
            return "N/A"
        
        if accuracy >= 90:
            color = Fore.GREEN
        elif accuracy >= 70:
            color = Fore.YELLOW
        else:
            color = Fore.RED
        
        return f"{color}{accuracy:.1f}%{Style.RESET_ALL}"
    
    def format_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp for display."""
        if timestamp is None:
            return "N/A"
        
        return timestamp.strftime("%H:%M:%S")
    
    def display_header(self):
        """Display the header section."""
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{'BTC PRICE PREDICTION SYSTEM':^80}")
        print(f"{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        print()
    
    def display_current_price(self):
        """Display current BTC price section."""
        print(f"{Fore.YELLOW}{'CURRENT BTC PRICE':^80}{Style.RESET_ALL}")
        print(f"{'-'*80}")
        
        if self.current_price is not None:
            # Get previous price for change calculation
            previous_price = None
            if len(self.prediction_history) > 0:
                previous_price = self.prediction_history[-1].get('current_price')
            
            price_str = self.format_price(self.current_price)
            change_str = self.format_change(self.current_price, previous_price) if previous_price else "N/A"
            
            print(f"Price: {Fore.WHITE}{price_str}{Style.RESET_ALL}")
            print(f"Change: {change_str}")
        else:
            print(f"{Fore.RED}Price data not available{Style.RESET_ALL}")
        
        print()
    
    def display_prediction(self):
        """Display prediction section."""
        if self.startup_mode:
            print(f"{Fore.CYAN}{'PREDICTION SYSTEM INITIALIZING...':^80}{Style.RESET_ALL}")
            print(f"{'-'*80}")
            print(f"{Fore.YELLOW}Waiting for first prediction...{Style.RESET_ALL}")
            print()
            return
        
        print(f"{Fore.GREEN}{'LATEST PREDICTION (5 MIN AHEAD)':^80}{Style.RESET_ALL}")
        print(f"{'-'*80}")
        
        if self.last_prediction:
            pred = self.last_prediction
            
            # Prediction details
            pred_price = self.format_price(pred.get('corrected_prediction'))
            raw_price = self.format_price(pred.get('raw_prediction'))
            confidence = pred.get('confidence_interval', (0, 0))
            conf_str = f"[{self.format_price(confidence[0])} - {self.format_price(confidence[1])}]"
            
            print(f"Predicted Price: {Fore.WHITE}{pred_price}{Style.RESET_ALL}")
            print(f"Raw Prediction: {raw_price}")
            print(f"Confidence Range: {conf_str}")
            print(f"Prediction Time: {self.format_timestamp(pred.get('timestamp'))}")
            
            # Rapid movement detection
            if pred.get('rapid_movement_detected'):
                print(f"{Fore.RED}⚠ RAPID MOVEMENT DETECTED ⚠{Style.RESET_ALL}")
            
            # Market volatility
            volatility = pred.get('market_volatility', 'unknown')
            vol_color = Fore.RED if volatility == 'high' else Fore.YELLOW if volatility == 'normal' else Fore.GREEN
            print(f"Market Volatility: {vol_color}{volatility.upper()}{Style.RESET_ALL}")
            
            # Accuracy if available
            if 'actual_price' in pred:
                actual = pred['actual_price']
                predicted = pred['corrected_prediction']
                error_pct = abs((predicted - actual) / actual) * 100
                accuracy = 100 - error_pct
                
                print(f"Actual Price: {self.format_price(actual)}")
                print(f"Accuracy: {self.format_accuracy(accuracy)}")
        else:
            print(f"{Fore.RED}No prediction available{Style.RESET_ALL}")
        
        print()
    
    def display_model_contributions(self):
        """Display individual model contributions."""
        if not self.show_model_contributions or not self.last_prediction:
            return
        
        print(f"{Fore.MAGENTA}{'MODEL CONTRIBUTIONS':^80}{Style.RESET_ALL}")
        print(f"{'-'*80}")
        
        contributions = self.last_prediction.get('corrected_contributions', {})
        
        if contributions:
            for model_name, contribution in contributions.items():
                model_display = model_name.replace('_', ' ').title()
                contrib_str = self.format_price(contribution)
                print(f"{model_display:15}: {contrib_str}")
        else:
            print(f"{Fore.RED}Model contributions not available{Style.RESET_ALL}")
        
        print()
    
    def display_recent_history(self):
        """Display recent prediction history."""
        if len(self.prediction_history) == 0:
            return
        
        print(f"{Fore.BLUE}{'RECENT PREDICTIONS':^80}{Style.RESET_ALL}")
        print(f"{'-'*80}")
        
        # Table header
        print(f"{'Time':<8} {'Predicted':<12} {'Actual':<12} {'Error':<8} {'Accuracy':<8}")
        print(f"{'-'*60}")
        
        # Show recent predictions
        recent_predictions = self.prediction_history[-self.max_history_display:]
        
        for pred in recent_predictions:
            time_str = self.format_timestamp(pred.get('timestamp'))
            pred_str = f"${pred.get('corrected_prediction', 0):.2f}"
            
            if 'actual_price' in pred:
                actual_str = f"${pred['actual_price']:.2f}"
                error = pred.get('error', 0)
                error_str = f"{error:+.2f}"
                
                accuracy = 100 - abs(pred.get('error_percentage', 100))
                acc_str = f"{accuracy:.1f}%"
                
                # Color code accuracy
                if accuracy >= 90:
                    acc_str = f"{Fore.GREEN}{acc_str}{Style.RESET_ALL}"
                elif accuracy >= 70:
                    acc_str = f"{Fore.YELLOW}{acc_str}{Style.RESET_ALL}"
                else:
                    acc_str = f"{Fore.RED}{acc_str}{Style.RESET_ALL}"
            else:
                actual_str = "Pending"
                error_str = "N/A"
                acc_str = "N/A"
            
            print(f"{time_str:<8} {pred_str:<12} {actual_str:<12} {error_str:<8} {acc_str:<8}")
        
        print()
    
    def display_system_status(self):
        """Display system status information."""
        if not self.show_detailed_info:
            return
        
        print(f"{Fore.CYAN}{'SYSTEM STATUS':^80}{Style.RESET_ALL}")
        print(f"{'-'*80}")
        
        if self.system_status:
            # Pipeline status
            pipeline_running = self.system_status.get('is_running', False)
            status_color = Fore.GREEN if pipeline_running else Fore.RED
            status_text = "RUNNING" if pipeline_running else "STOPPED"
            print(f"Pipeline Status: {status_color}{status_text}{Style.RESET_ALL}")
            
            # Last prediction time
            last_pred_time = self.system_status.get('last_prediction_time')
            if last_pred_time:
                print(f"Last Prediction: {self.format_timestamp(last_pred_time)}")
            
            # Metrics
            metrics = self.system_status.get('metrics', {})
            total_preds = metrics.get('total_predictions', 0)
            successful_preds = metrics.get('successful_predictions', 0)
            success_rate = (successful_preds / total_preds * 100) if total_preds > 0 else 0
            
            print(f"Total Predictions: {total_preds}")
            print(f"Success Rate: {self.format_accuracy(success_rate)}")
            
            # Data status
            data_status = self.system_status.get('data_status', {})
            current_price_status = data_status.get('current_price')
            if current_price_status:
                print(f"Data Connection: {Fore.GREEN}ACTIVE{Style.RESET_ALL}")
            else:
                print(f"Data Connection: {Fore.RED}INACTIVE{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}System status not available{Style.RESET_ALL}")
        
        print()
    
    def display_performance_summary(self):
        """Display model performance summary."""
        if not self.model_performance:
            return
        
        print(f"{Fore.WHITE}{'PERFORMANCE SUMMARY':^80}{Style.RESET_ALL}")
        print(f"{'-'*80}")
        
        # Overall accuracy
        if 'ensemble' in self.model_performance:
            ensemble_perf = self.model_performance['ensemble']
            mae = ensemble_perf.get('mae', 0)
            mape = ensemble_perf.get('mape', 0)
            directional_acc = ensemble_perf.get('directional_accuracy', 0)
            
            print(f"Mean Absolute Error: ${mae:.2f}")
            print(f"Mean Absolute Percentage Error: {mape:.2f}%")
            print(f"Directional Accuracy: {self.format_accuracy(directional_acc)}")
        
        print()
    
    def display_footer(self):
        """Display footer with controls."""
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{'Press Ctrl+C to stop monitoring':^80}")
        print(f"{'='*80}{Style.RESET_ALL}")
    
    def update_display(self):
        """Update the terminal display."""
        try:
            self.clear_screen()
            
            # Display all sections
            self.display_header()
            self.display_current_price()
            self.display_prediction()
            
            if not self.startup_mode:
                self.display_model_contributions()
                self.display_recent_history()
                self.display_system_status()
                self.display_performance_summary()
            
            self.display_footer()
            
        except Exception as e:
            logger.error(f"Display update failed: {e}")
    
    def update_data(self, current_price: float = None, prediction_data: Dict = None, 
                   system_status: Dict = None, model_performance: Dict = None):
        """
        Update display data.
        
        Args:
            current_price: Current BTC price
            prediction_data: Latest prediction data
            system_status: System status information
            model_performance: Model performance metrics
        """
        try:
            if current_price is not None:
                self.current_price = current_price
            
            if prediction_data:
                self.last_prediction = prediction_data
                self.prediction_history.append(prediction_data)
                
                # Keep only recent history
                if len(self.prediction_history) > 100:
                    self.prediction_history = self.prediction_history[-100:]
                
                # Exit startup mode after first prediction
                if self.startup_mode:
                    self.startup_mode = False
                    logger.info("Exiting startup mode - predictions now available")
            
            if system_status:
                self.system_status = system_status
            
            if model_performance:
                self.model_performance = model_performance
                
        except Exception as e:
            logger.error(f"Data update failed: {e}")
    
    def start_monitoring(self):
        """Start the terminal monitoring display."""
        if self.is_running:
            logger.warning("Terminal monitor already running")
            return
        
        logger.info("Starting terminal monitoring...")
        
        self.is_running = True
        
        def monitor_loop():
            try:
                while self.is_running:
                    self.update_display()
                    time.sleep(self.update_interval)
            except KeyboardInterrupt:
                logger.info("Terminal monitoring stopped by user")
                self.is_running = False
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                self.is_running = False
        
        self.display_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.display_thread.start()
        
        # Initial display
        self.update_display()
        
        logger.info("Terminal monitoring started")
    
    def stop_monitoring(self):
        """Stop the terminal monitoring display."""
        if not self.is_running:
            logger.warning("Terminal monitor not running")
            return
        
        logger.info("Stopping terminal monitoring...")
        
        self.is_running = False
        
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2)
        
        # Clear screen and show final message
        self.clear_screen()
        print(f"{Fore.CYAN}BTC Prediction System Monitoring Stopped{Style.RESET_ALL}")
        
        logger.info("Terminal monitoring stopped")
    
    def toggle_detailed_info(self):
        """Toggle detailed information display."""
        self.show_detailed_info = not self.show_detailed_info
        logger.info(f"Detailed info display: {'ON' if self.show_detailed_info else 'OFF'}")
    
    def toggle_model_contributions(self):
        """Toggle model contributions display."""
        self.show_model_contributions = not self.show_model_contributions
        logger.info(f"Model contributions display: {'ON' if self.show_model_contributions else 'OFF'}")
    
    def set_update_interval(self, seconds: int):
        """
        Set the display update interval.
        
        Args:
            seconds: Update interval in seconds
        """
        if seconds < 5:
            logger.warning("Update interval too short, minimum is 5 seconds")
            seconds = 5
        
        self.update_interval = seconds
        logger.info(f"Display update interval set to {seconds} seconds")
    
    def get_monitor_status(self) -> Dict[str, Any]:
        """
        Get monitoring system status.
        
        Returns:
            Dictionary with monitor status
        """
        return {
            'is_running': self.is_running,
            'startup_mode': self.startup_mode,
            'update_interval': self.update_interval,
            'show_detailed_info': self.show_detailed_info,
            'show_model_contributions': self.show_model_contributions,
            'prediction_history_count': len(self.prediction_history),
            'current_price': self.current_price,
            'last_prediction_time': self.last_prediction.get('timestamp') if self.last_prediction else None
        }