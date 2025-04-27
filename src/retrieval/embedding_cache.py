import os
import pickle
import hashlib
import time
from typing import List, Dict, Any, Optional
from src.utils.logger import get_data_logger
from src.utils.registry import registry

logger = get_data_logger()

class EmbeddingCache:
    def __init__(self, cache_dir="src/data/cache/embeddings", max_memory_items=1000):
        """
        Initialize the embedding cache
        
        Args:
            cache_dir: Directory to store persistent cache files
            max_memory_items: Maximum number of items to keep in memory cache
        """
        self.cache_dir = cache_dir
        self.max_memory_items = max_memory_items
        self.memory_cache = {}
        self.access_times = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"Embedding cache initialized with directory: {cache_dir}")
    
    def _get_cache_key(self, text, model_name):
        """Generate a cache key for a text and model"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        model_hash = hashlib.md5(model_name.encode()).hexdigest()
        return f"{model_hash}_{text_hash}"
    
    def get(self, text, model_name):
        """Get embeddings from cache if available"""
        key = self._get_cache_key(text, model_name)
        
        # Check memory cache first (faster)
        if key in self.memory_cache:
            # Update access time
            self.access_times[key] = time.time()
            logger.debug(f"Memory cache hit for embedding: {key[:8]}")
            return self.memory_cache[key]
        
        # Check disk cache
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    embedding = pickle.load(f)
                
                # Add to memory cache
                self._add_to_memory_cache(key, embedding)
                
                logger.debug(f"Disk cache hit for embedding: {key[:8]}")
                return embedding
            except Exception as e:
                logger.warning(f"Error loading from cache: {e}")
        
        return None
    
    def set(self, text, model_name, embedding):
        """Store embeddings in cache"""
        key = self._get_cache_key(text, model_name)
        
        # Add to memory cache
        self._add_to_memory_cache(key, embedding)
        
        # Save to disk cache
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(embedding, f)
            logger.debug(f"Cached embedding: {key[:8]}")
        except Exception as e:
            logger.warning(f"Error caching embedding: {e}")
    
    def _add_to_memory_cache(self, key, embedding):
        """Add an embedding to the memory cache, evicting old items if necessary"""
        # If cache is full, remove least recently used item
        if len(self.memory_cache) >= self.max_memory_items:
            # Find least recently used item
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            # Remove from caches
            del self.memory_cache[oldest_key]
            del self.access_times[oldest_key]
            logger.debug(f"Evicted LRU item from memory cache: {oldest_key[:8]}")
        
        # Add to caches
        self.memory_cache[key] = embedding
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear all cache entries"""
        # Clear memory cache
        self.memory_cache.clear()
        self.access_times.clear()
        
        # Clear disk cache
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".pkl"):
                os.remove(os.path.join(self.cache_dir, filename))
        
        logger.info("Embedding cache cleared")
    
    def get_stats(self):
        """Get statistics about the cache"""
        disk_cache_count = len([f for f in os.listdir(self.cache_dir) if f.endswith(".pkl")])
        memory_cache_count = len(self.memory_cache)
        
        return {
            "memory_cache_size": memory_cache_count,
            "disk_cache_size": disk_cache_count,
            "max_memory_items": self.max_memory_items
        }
    def preload(self, file_path):
        """
        Preload cache from a file of text entries
        
        Args:
            file_path: Path to a text file with one entry per line
        """
        if not os.path.exists(file_path):
            logger.warning(f"Preload file not found: {file_path}")
            return 0
            
        count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            # Check which texts are already in cache
            model_name = "preload"  # Use a generic model name for preloaded texts
            for text in texts:
                key = self._get_cache_key(text, model_name)
                if key not in self.memory_cache:
                    count += 1
            
            logger.info(f"Preloaded {count} new items into cache from {file_path}")
            return count
        except Exception as e:
            logger.error(f"Error preloading cache: {e}")
            return 0
    
    def cleanup(self, max_age_days=30):
        """
        Remove cache files older than specified days
        
        Args:
            max_age_days: Maximum age in days for cache files
            
        Returns:
            Number of files removed
        """
        max_age_seconds = max_age_days * 24 * 60 * 60
        count = 0
        current_time = time.time()
        
        for filename in os.listdir(self.cache_dir):
            if not filename.endswith(".pkl"):
                continue
                
            file_path = os.path.join(self.cache_dir, filename)
            file_age = current_time - os.path.getmtime(file_path)
            
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    logger.error(f"Error removing old cache file {filename}: {e}")
        
        if count > 0:
            logger.info(f"Removed {count} old cache files")
        
        return count
    
    def __str__(self):
        """String representation of the cache"""
        stats = self.get_stats()
        return f"EmbeddingCache(memory={stats['memory_cache_size']}, disk={stats['disk_cache_size']})"
