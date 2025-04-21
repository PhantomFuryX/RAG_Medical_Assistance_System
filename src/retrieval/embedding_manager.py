import torch
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.logger import get_logger
from src.retrieval.embedding_cache import EmbeddingCache


# Configure logging
logger = get_logger("retrieval")



class EmbeddingManager:
    _instance = None
    _model = None
    _model_name = None
    _device = None
    _initialized = False
    
    @classmethod
    def get_instance(cls, model_name=None, device=None):
        """Get or create a singleton instance of the embedding manager"""
        if cls._instance is None:
            cls._instance = cls(model_name=model_name, device=device)
        return cls._instance
    
    def __init__(self, model_name):
        if EmbeddingManager._instance is not None:
            raise Exception("This class is a singleton. Use get_instance() instead.")
        self.cache = EmbeddingCache()
        
        # Store parameters but don't initialize yet
        self._model_name = model_name
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Embedding manager initialized with model {model_name} (lazy loading)")
        
        EmbeddingManager._instance = self
    
    def _initialize(self):
        if not self._initialized:
            logger.info(f"Loading embedding model: {self._model_name} on {self._device}")
            try:
                self._model = HuggingFaceEmbeddings(
                    model_name=self._model_name,
                    model_kwargs={"device": self._device}
                )
                logger.info(f"Initialized embeddings model: {self._model_name}")
                self._initialized = True
            except Exception as e:
                logger.error(f"Error initializing embeddings: {e}")
                self._model = None
                
    def embed_documents(self, texts):
        """Embed multiple documents with caching"""
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
        
        # Compute embeddings for uncached texts
        uncached_embeddings = self._model.embed_documents(uncached_texts)
        
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
    @property
    def model(self):
        if not self._initialized:
            self._initialize()
        return self._model