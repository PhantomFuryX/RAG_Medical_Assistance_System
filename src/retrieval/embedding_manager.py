import torch
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any, Optional
import time
import os
from src.utils.logger import get_data_logger
from src.utils.registry import registry
from src.retrieval.embedding_cache import EmbeddingCache

logger = get_data_logger()

class EmbeddingManager:
    _instance = None
    _model = None
    _model_name = None
    _device = None
    _initialized = False
    
    @classmethod
    def get_instance(cls, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Get or create a singleton instance of the embedding manager"""
        if cls._instance is None:
            cls._instance = cls(model_name)
        elif cls._instance._model_name != model_name:
            logger.warning(f"Requested model {model_name} but instance already exists with model {cls._instance._model_name}")
        
        # Store in registry if not already there
        if not registry.has("embedding_manager"):
            registry.set("embedding_manager", cls._instance)
            
        return cls._instance
    
    def __init__(self, model_name):
        """Initialize the embedding manager with the specified model"""
        if EmbeddingManager._instance is not None:
            raise Exception("This class is a singleton. Use get_instance() instead.")
        
        # Create or get cache from registry
        if registry.has("embedding_cache"):
            self.cache = registry.get("embedding_cache")
            logger.info("Using existing embedding cache from registry")
        else:
            self.cache = EmbeddingCache()
            registry.set("embedding_cache", self.cache)
        
        # Store parameters but don't initialize yet
        self._model_name = model_name
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Embedding manager initialized with model {model_name} (lazy loading) on {self._device}")
        
        # Store batch size for efficient processing
        self._batch_size = 32 if self._device == "cuda" else 8
        
        # Register in registry
        registry.set("embedding_manager", self)
        
        EmbeddingManager._instance = self
    
    def _initialize(self):
        """Initialize the embedding model if not already initialized"""
        if not self._initialized:
            # Check if model already exists in registry
            registry_key = f"embedding_model_{self._model_name}"
            if registry.has(registry_key):
                logger.info(f"Using embedding model {self._model_name} from registry")
                self._model = registry.get(registry_key)
                self._initialized = True
                return
                
            logger.info(f"Loading embedding model: {self._model_name} on {self._device}")
            try:
                self._model = HuggingFaceEmbeddings(
                    model_name=self._model_name,
                    model_kwargs={"device": self._device}
                )
                logger.info(f"Initialized embeddings model: {self._model_name}")
                self._initialized = True
                
                # Store in registry
                registry.set(registry_key, self._model)
            except Exception as e:
                logger.error(f"Error initializing embeddings: {e}")
                self._model = None
    
    def embed_documents(self, texts):
        """Embed multiple documents with caching and batching"""
        if not texts:
            return []
            
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cached_embedding = self.cache.get(text, self._model_name)
            if cached_embedding is not None:
                results.append(cached_embedding)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # If all embeddings were cached, return them
        if not uncached_texts:
            return results
        
        # Initialize model if needed
        if not self._initialized:
            self._initialize()
        
        # Process uncached texts in batches for efficiency
        uncached_embeddings = []
        
        if len(uncached_texts) <= self._batch_size:
            # Small enough for a single batch
            batch_embeddings = self._model.embed_documents(uncached_texts)
            uncached_embeddings.extend(batch_embeddings)
        else:
            # Process in batches
            logger.info(f"Processing {len(uncached_texts)} texts in batches of {self._batch_size}")
            for i in range(0, len(uncached_texts), self._batch_size):
                batch = uncached_texts[i:i + self._batch_size]
                batch_embeddings = self._model.embed_documents(batch)
                uncached_embeddings.extend(batch_embeddings)
        
        # Cache the new embeddings
        for text, embedding in zip(uncached_texts, uncached_embeddings):
            self.cache.set(text, self._model_name, embedding)
        
        # Merge cached and newly computed embeddings
        final_results = [None] * len(texts)
        for i, embedding in enumerate(results):
            final_results[i] = embedding
        
        for i, embedding in zip(uncached_indices, uncached_embeddings):
            final_results[i] = embedding
        
        return final_results
    
    def embed_query(self, query):
        """Embed a single query string"""
        # Check cache first
        cached_embedding = self.cache.get(query, self._model_name)
        if cached_embedding is not None:
            return cached_embedding
        
        # Initialize model if needed
        if not self._initialized:
            self._initialize()
        
        # Compute embedding
        embedding = self._model.embed_query(query)
        
        # Cache the embedding
        self.cache.set(query, self._model_name, embedding)
        
        return embedding
    
    def __str__(self):
        """String representation of the embedding manager"""
        return f"EmbeddingManager(model={self._model_name}, device={self._device}, initialized={self._initialized})"
    
    def __repr__(self):
        """Detailed representation of the embedding manager"""
        cache_stats = self.get_cache_stats()
        return (f"EmbeddingManager(model={self._model_name}, device={self._device}, "
                f"initialized={self._initialized}, batch_size={self._batch_size}, "
                f"memory_cache_size={cache_stats['memory_cache_size']}, "
                f"disk_cache_size={cache_stats['disk_cache_size']})")
    
    def preload_common_embeddings(self, common_texts=None):
        """
        Preload embeddings for commonly used texts to improve response time
        
        Args:
            common_texts: List of common texts to preload, or None to use defaults
        """
        if common_texts is None:
            # Default common medical terms that might be frequently queried
            common_texts = [
                "fever", "headache", "pain", "cough", "cold", "flu", 
                "diabetes", "hypertension", "cancer", "heart disease",
                "What are the symptoms of", "How to treat", "Is it serious",
                "When should I see a doctor", "What causes", "How to prevent"
            ]
        
        logger.info(f"Preloading embeddings for {len(common_texts)} common texts")
        self.embed_documents(common_texts)
        logger.info("Preloading complete")
    
    def warm_up(self):
        """
        Warm up the embedding model by running a test embedding
        This can help reduce latency for the first real request
        """
        if not self._initialized:
            logger.info("Warming up embedding model...")
            self.embed_query("warm up")
            logger.info("Embedding model warmed up")

    def clear_cache(self):
        """Clear the embedding cache"""
        self.cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self):
        """Get statistics about the cache"""
        return self.cache.get_stats()
    
    @property
    def model(self):
        """Get the underlying embedding model, initializing if necessary"""
        if not self._initialized:
            self._initialize()
        return self._model
    
    @property
    def model_name(self):
        """Get the name of the embedding model"""
        return self._model_name
    
    @property
    def device(self):
        """Get the device the model is running on"""
        return self._device
    
    def precompute_common_embeddings(self):
        """Precompute embeddings for common medical terms to speed up queries"""
        common_terms_file = "src/data/common_medical_terms.txt"
        
        if not os.path.exists(common_terms_file):
            logger.warning(f"Common terms file not found: {common_terms_file}")
            return
        
        with open(common_terms_file, 'r') as f:
            common_terms = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Precomputing embeddings for {len(common_terms)} common medical terms")
        
        # Compute embeddings in batches
        self.embed_documents(common_terms)
        
        logger.info("Precomputed embeddings for common terms")