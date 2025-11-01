"""Backup and recovery system for prediction datasets and models."""
import os
import shutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class BackupRecoverySystem:
    """Handles backup and recovery of prediction datasets and models."""
    
    def __init__(self):
        self.backup_dir = os.path.join(Config.DATA_DIR, 'backups')
        self.max_backups = 10  # Keep last 10 backups
        
        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def create_backup(self) -> bool:
        """Create a backup of all important data."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(self.backup_dir, f'backup_{timestamp}')
            os.makedirs(backup_path, exist_ok=True)
            
            # Backup data files
            data_files = [
                Config.PREDICTIONS_FILE,
                os.path.join(Config.DATA_DIR, 'btc_historical.csv')
            ]
            
            for file_path in data_files:
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path)
                    shutil.copy2(file_path, os.path.join(backup_path, filename))
            
            # Backup models
            models_backup_dir = os.path.join(backup_path, 'models')
            if os.path.exists(Config.MODELS_DIR):
                shutil.copytree(Config.MODELS_DIR, models_backup_dir)
            
            # Create backup manifest
            manifest = {
                'timestamp': timestamp,
                'backup_path': backup_path,
                'files_backed_up': data_files,
                'models_backed_up': os.path.exists(Config.MODELS_DIR)
            }
            
            with open(os.path.join(backup_path, 'manifest.json'), 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
            
            # Cleanup old backups
            self.cleanup_old_backups()
            
            logger.info(f"Backup created successfully: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return False
    
    def cleanup_old_backups(self):
        """Remove old backups to save space."""
        try:
            backup_dirs = [d for d in os.listdir(self.backup_dir) 
                          if d.startswith('backup_') and os.path.isdir(os.path.join(self.backup_dir, d))]
            
            backup_dirs.sort(reverse=True)  # Most recent first
            
            # Remove excess backups
            for old_backup in backup_dirs[self.max_backups:]:
                old_path = os.path.join(self.backup_dir, old_backup)
                shutil.rmtree(old_path)
                logger.info(f"Removed old backup: {old_backup}")
                
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
    
    def restore_backup(self, backup_timestamp: str) -> bool:
        """Restore from a specific backup."""
        try:
            backup_path = os.path.join(self.backup_dir, f'backup_{backup_timestamp}')
            
            if not os.path.exists(backup_path):
                logger.error(f"Backup not found: {backup_timestamp}")
                return False
            
            # Load manifest
            manifest_path = os.path.join(backup_path, 'manifest.json')
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                logger.info(f"Restoring backup from {manifest['timestamp']}")
            
            # Restore data files
            for filename in ['predictions.csv', 'btc_historical.csv']:
                backup_file = os.path.join(backup_path, filename)
                if os.path.exists(backup_file):
                    if filename == 'predictions.csv':
                        target_path = Config.PREDICTIONS_FILE
                    else:
                        target_path = os.path.join(Config.DATA_DIR, filename)
                    
                    # Create target directory if needed
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    shutil.copy2(backup_file, target_path)
                    logger.info(f"Restored {filename}")
            
            # Restore models
            models_backup_path = os.path.join(backup_path, 'models')
            if os.path.exists(models_backup_path):
                if os.path.exists(Config.MODELS_DIR):
                    shutil.rmtree(Config.MODELS_DIR)
                shutil.copytree(models_backup_path, Config.MODELS_DIR)
                logger.info("Restored models")
            
            logger.info(f"Backup restoration completed: {backup_timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Backup restoration failed: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        try:
            backups = []
            
            backup_dirs = [d for d in os.listdir(self.backup_dir) 
                          if d.startswith('backup_') and os.path.isdir(os.path.join(self.backup_dir, d))]
            
            for backup_dir in sorted(backup_dirs, reverse=True):
                backup_path = os.path.join(self.backup_dir, backup_dir)
                manifest_path = os.path.join(backup_path, 'manifest.json')
                
                backup_info = {
                    'timestamp': backup_dir.replace('backup_', ''),
                    'path': backup_path,
                    'size_mb': self.get_directory_size(backup_path) / (1024 * 1024)
                }
                
                if os.path.exists(manifest_path):
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        backup_info.update(manifest)
                
                backups.append(backup_info)
            
            return backups
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []
    
    def get_directory_size(self, path: str) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception as e:
            logger.error(f"Failed to calculate directory size: {e}")
        
        return total_size