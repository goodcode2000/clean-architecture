"""
Feature caching system for performance optimization
"""

import asyncio
import pickle
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import hashlib
from .feature_engineering import FeatureSet
from .config import Config

logger = logging.getLogger(__name__)

class FeatureCache:
    """High-performance feature caching system"""
    
    def __init__(self, config: Config, max_memory_items: int = 1000, 
                 disk_cache_days: int = 7):
        self.config = config
        self.max_memory_items = max_memory_items
        self.disk_cache_days = disk_cache_days
        
        # In-memory cache
        self.memory_cache: Dict[str, FeatureSet] = {}
        self.access_times: Dict[str, datetime] = {}
        
        # Disk cache directory
        self.cache_dir = os.path.join(config.DATA_DIR, 'feature_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'evictions': 0
        }
        
        self._lock = asyncio.Lock()
    
    def _generate_cache_key(self, timestamp: datetime, data_hash: str) -> str:
        """Generate unique cache key"""
        time_str = timestamp.strftime('%Y%m%d_%H%M%S')
        return f"{time_str}_{data_hash[:8]}"
    
    def _hash_data(self, data: Any) -> str:
        """Generate hash for data"""
        data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    async def get(self, timestamp: datetime, data_hash: str) -> Optional[FeatureSet]:
        """Get features from cache"""
        cache_key = self._generate_cache_key(timestamp, data_hash)
        
        async with self._lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                self.access_times[cache_key] = datetime.now()
                self.stats['hits'] += 1
                self.stats['memory_hits'] += 1
                logger.debug(f"Memory cache hit for {cache_key}")
                return self.memory_cache[cache_key]
            
            # Check disk cache
            disk_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(disk_path):
                try:
                    with open(disk_path, 'rb') as f:
                        feature_set = pickle.load(f)
                    
                    # Move to memory cache
                    await self._add_to_memory_cache(cache_key, feature_set)
                    
                    self.stats['hits'] += 1
                    self.stats['disk_hits'] += 1
                    logger.debug(f"Disk cache hit for {cache_key}")
                    return feature_set
                    
                except Exception as e:
                    logger.warning(f"Failed to load from disk cache {cache_key}: {e}")
                    # Remove corrupted file
                    try:
                        os.remove(disk_path)
                    except:
                        pass
            
            # Cache miss
            self.stats['misses'] += 1
            logger.debug(f"Cache miss for {cache_key}")
            return None
    
    async def put(self, timestamp: datetime, data_hash: str, feature_set: FeatureSet):
        """Store features in cache"""
        cache_key = self._generate_cache_key(timestamp, data_hash)
        
        async with self._lock:
            # Add to memory cache
            await self._add_to_memory_cache(cache_key, feature_set)
            
            # Save to disk cache asynchronously
            asyncio.create_task(self._save_to_disk(cache_key, feature_set))
    
    async def _add_to_memory_cache(self, cache_key: str, feature_set: FeatureSet):
        """Add feature set to memory cache with LRU eviction"""
        # Check if we need to evict
        if len(self.memory_cache) >= self.max_memory_items:
            await self._evict_lru()
        
        self.memory_cache[cache_key] = feature_set
        self.access_times[cache_key] = datetime.now()
        
        logger.debug(f"Added {cache_key} to memory cache")
    
    async def _evict_lru(self):
        """Evict least recently used item from memory cache"""
        if not self.access_times:
            return
        
        # Find least recently used item
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from memory cache
        del self.memory_cache[lru_key]
        del self.access_times[lru_key]
        
        self.stats['evictions'] += 1
        logger.debug(f"Evicted {lru_key} from memory cache")
    
    async def _save_to_disk(self, cache_key: str, feature_set: FeatureSet):
        """Save feature set to disk cache"""
        try:
            disk_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            with open(disk_path, 'wb') as f:
                pickle.dump(feature_set, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.debug(f"Saved {cache_key} to disk cache")
            
        except Exception as e:
            logger.warning(f"Failed to save to disk cache {cache_key}: {e}")
    
    async def cleanup_old_cache(self):
        """Clean up old cache files"""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.disk_cache_days)
            removed_count = 0
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    
                    if file_time < cutoff_time:
                        try:
                            os.remove(file_path)
                            removed_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to remove old cache file {filename}: {e}")
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old cache files")
                
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'memory_hits': self.stats['memory_hits'],
            'disk_hits': self.stats['disk_hits'],
            'evictions': self.stats['evictions'],
            'memory_cache_size': len(self.memory_cache),
            'disk_cache_files': len([f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')])
        }
    
    async def clear_cache(self):
        """Clear all cache data"""
        async with self._lock:
            # Clear memory cache
            self.memory_cache.clear()
            self.access_times.clear()
            
            # Clear disk cache
            try:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.pkl'):
                        os.remove(os.path.join(self.cache_dir, filename))
                logger.info("Cache cleared successfully")
            except Exception as e:
                logger.error(f"Failed to clear disk cache: {e}")
            
            # Reset statistics
            self.stats = {
                'hits': 0,
                'misses': 0,
                'memory_hits': 0,
                'disk_hits': 0,
                'evictions': 0
            }

class FeaturePipeline:
    """Feature engineering pipeline with caching"""
    
    def __init__(self, config: Config, feature_engineer, cache_enabled: bool = True):
        self.config = config
        self.feature_engineer = feature_engineer
        self.cache_enabled = cache_enabled
        
        if cache_enabled:
            self.cache = FeatureCache(config)
        else:
            self.cache = None
    
    async def process_features(self, price_data: List, market_data: Optional[List] = None) -> List[FeatureSet]:
        """Process features with caching"""
        if not price_data:
            return []
        
        # Generate data hash for cache key
        data_hash = self._hash_input_data(price_data, market_data)
        
        # Check cache if enabled
        if self.cache_enabled and self.cache:
            cached_features = await self.cache.get(price_data[-1].timestamp, data_hash)
            if cached_features:
                return [cached_features]
        
        # Compute features
        feature_sets = await self.feature_engineer.engineer_features(price_data, market_data)
        
        # Cache results if enabled
        if self.cache_enabled and self.cache and feature_sets:
            await self.cache.put(price_data[-1].timestamp, data_hash, feature_sets[-1])
        
        return feature_sets
    
    def _hash_input_data(self, price_data: List, market_data: Optional[List] = None) -> str:
        """Generate hash for input data"""
        # Create a string representation of the input data
        data_str = ""
        
        # Hash price data
        for pd in price_data[-10:]:  # Use last 10 points for hash
            data_str += f"{pd.timestamp}_{pd.close_price}_{pd.volume}_"
        
        # Hash market data if available
        if market_data:
            for md in market_data[-10:]:
                if md:
                    data_str += f"{md.volume_delta}_{md.funding_rate}_"
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    async def cleanup_cache(self):
        """Clean up old cache data"""
        if self.cache:
            await self.cache.cleanup_old_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache:
            return self.cache.get_cache_stats()
        return {'cache_disabled': True}