import os
import pickle
import hashlib
import time
from typing import Any, Dict, Optional
from src.utils.logger import get_data_logger
from src.utils.registry import registry

logger = get_data_logger()

class CacheManager:
    """Manager for caching and retrieving objects"""
    
    def __init__(self, cache_dir: str = "src/data/cache", max_age_days: int = 7):
        """
        Initialize the cache manager
        
        Args:
            cache_dir: Directory to store cache files
            max_age_days: Maximum age of cache files in days
        """
        self.cache_dir = cache_dir
        self.max_age_seconds = max_age_days * 24 * 60 * 60
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Store in registry
        if not registry.has("cache_manager"):
            registry.set("cache_manager", self)
            logger.info(f"Cache manager initialized with cache directory: {cache_dir}")
    
    def _get_cache_path(self, key: str) -> str:
        """Get the file path for a cache key"""
        # Create a hash of the key
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}.pkl")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get an object from the cache
        
        Args:
            key: Cache key
            default: Default value to return if key not found
            
        Returns:
            Cached object or default
        """
        cache_path = self._get_cache_path(key)
        
        # Check if cache file exists
        if not os.path.exists(cache_path):
            return default
        
        # Check if cache is too old
        if self.max_age_seconds > 0:
            file_age = time.time() - os.path.getmtime(cache_path)
            if file_age > self.max_age_seconds:
                logger.info(f"Cache for {key} is too old ({file_age/86400:.1f} days), removing")
                os.remove(cache_path)
                return default
        
        # Load the cached object
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading cache for {key}: {e}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Store an object in the cache
        
        Args:
            key: Cache key
            value: Object to cache
        """
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.error(f"Error caching object for {key}: {e}")
    
    def clear(self, key: Optional[str] = None) -> None:
        """
        Clear cache entries
        
        Args:
            key: Specific key to clear, or None to clear all
        """
        if key is not None:
            # Clear specific key
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logger.info(f"Cleared cache for {key}")
        else:
            # Clear all cache files
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".pkl"):
                    os.remove(os.path.join(self.cache_dir, filename))
            logger.info("Cleared all cache files")
    
    def cleanup_old_files(self) -> int:
        """
        Remove cache files older than max_age_days
        
        Returns:
            Number of files removed
        """
        if self.max_age_seconds <= 0:
            return 0
            
        count = 0
        current_time = time.time()
        
        for filename in os.listdir(self.cache_dir):
            if not filename.endswith(".pkl"):
                continue
                
            file_path = os.path.join(self.cache_dir, filename)
            file_age = current_time - os.path.getmtime(file_path)
            
            if file_age > self.max_age_seconds:
                os.remove(file_path)
                count += 1
        
        if count > 0:
            logger.info(f"Removed {count} old cache files")
        
        return count
