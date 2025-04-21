import os
import pickle
import hashlib
import logging

logger = logging.getLogger("retrieval")

class EmbeddingCache:
    def __init__(self, cache_dir="src/data/cache/embeddings"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, text, model_name):
        """Generate a cache key for a text and model"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        model_hash = hashlib.md5(model_name.encode()).hexdigest()
        return f"{model_hash}_{text_hash}"
    
    def get(self, text, model_name):
        """Get embeddings from cache if available"""
        cache_key = self._get_cache_key(text, model_name)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    embedding = pickle.load(f)
                logger.debug(f"Cache hit for embedding: {cache_key[:8]}")
                return embedding
            except Exception as e:
                logger.warning(f"Error loading from cache: {e}")
        
        return None
    
    def set(self, text, model_name, embedding):
        """Store embeddings in cache"""
        cache_key = self._get_cache_key(text, model_name)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(embedding, f)
            logger.debug(f"Cached embedding: {cache_key[:8]}")
        except Exception as e:
            logger.warning(f"Error caching embedding: {e}")