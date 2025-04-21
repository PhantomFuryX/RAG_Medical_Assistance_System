from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import os
import time
import threading
import numpy as np
import faiss
import json
import pickle
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from src.retrieval.embedding_manager import EmbeddingManager
import hashlib
import json
from src.utils.registry import registry

from src.utils.logger import get_logger

# Configure logging
logger = get_logger("retrieval")
class MedicalDocumentRetriever:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 index_path: str = "faiss_medical_index",
                 lazy_loading: bool = True):
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device} for embeddings")
        
        # Store parameters
        # self.embedding_model = embedding_model
        self.embedding_manager = EmbeddingManager.get_instance(embedding_model)
        
        self._embeddings = self.embedding_manager.model
        self.index_path = index_path
        self._embeddings = None
        self.index = None
        self._loading_lock = threading.Lock()
        self._loading_thread = None
        self._loading_complete = threading.Event()
        self._document_hash_file = os.path.join(index_path, "document_hash.json")
        
        # Cache for query results
        self._query_cache = {}
        self._max_cache_size = 100
        
        # Initialize embeddings (this is relatively fast)
        self._init_embeddings()
        
        # Load the index (potentially in background)
        if not lazy_loading:
            self._load_index()
        else:
            logger.info(f"FAISS index will be loaded lazily on first use")
    
    def _init_embeddings(self):
        """Initialize the embeddings model"""
        try:
            self._embeddings = self.embedding_manager.model
            if self._embeddings:
                logger.info(f"Initialized embeddings model from embedding manager")
            else:
                logger.warning("Embedding manager returned None for model")
            # logger.info(f"Initialized embeddings model: {self.embedding_model}")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            self._embeddings = None
    
    def _load_index_in_background(self):
        """Load the index in a background thread"""
        try:
            logger.info(f"Starting background loading of FAISS index from {self.index_path}")
            start_time = time.time()
            
            # Check if the index exists
            if not os.path.exists(os.path.join(self.index_path, "index.faiss")):
                logger.warning(f"Index not found at {self.index_path}")
                self._loading_complete.set()
                return
            
            # Load the index
            self.index = FAISS.load_local(
                self.index_path, 
                self._embeddings, 
                allow_dangerous_deserialization=True
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Completed background loading of FAISS index in {elapsed:.2f} seconds")
            self._loading_complete.set()
        except Exception as e:
            logger.error(f"Error in background loading of FAISS index: {e}")
            self._loading_complete.set()
    
    def _load_index(self):
        """Load the index, potentially waiting if it's already loading"""
        # If already loaded, return immediately
        if self.index is not None:
            return
        
        # If already loading in background, wait for it to complete
        if self._loading_thread is not None and self._loading_thread.is_alive():
            logger.info("Waiting for background index loading to complete...")
            self._loading_complete.wait()
            return
        
        # Acquire lock to prevent multiple threads from loading simultaneously
        with self._loading_lock:
            # Check again in case another thread loaded while waiting for lock
            if self.index is not None:
                return
                
            # Start loading in background
            self._loading_thread = threading.Thread(target=self._load_index_in_background)
            self._loading_thread.daemon = True
            self._loading_thread.start()
            
            # For the first request, wait for loading to complete
            logger.info("Waiting for initial index loading to complete...")
            self._loading_complete.wait()
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for a query with caching"""
        # Check cache first
        cache_key = f"{query}_{k}"
        if cache_key in self._query_cache:
            logger.info(f"Cache hit for query: {query}")
            return self._query_cache[cache_key]
        
        # Ensure index is loaded
        self._load_index()
        
        if self.index is None:
            logger.warning("No index available for retrieval")
            return []
        
        # Perform retrieval
        try:
            results = self.index.similarity_search(query, k=k)
            
            # Cache the results
            if len(self._query_cache) >= self._max_cache_size:
                # Remove oldest item
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[cache_key] = results
            
            return results
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def create_index(self, documents: List[Document]):
        """Create a new optimized FAISS index from the provided documents."""
        if not documents:
            raise ValueError("No documents provided to create the index.")
        
        logger.info(f"Creating new optimized index with {len(documents)} documents")
        
        current_hash = self._get_documents_hash(documents)
    
        # Check if the index exists and has the same document hash
        if os.path.exists(os.path.join(self.index_path, "index.faiss")) and \
        os.path.exists(self._document_hash_file):
            try:
                with open(self._document_hash_file, "r") as f:
                    stored_hash = json.load(f).get("hash")
                    
                if stored_hash == current_hash:
                    logger.info("Documents haven't changed, using existing index")
                    self._load_index()
                    return self.index
            except Exception as e:
                logger.warning(f"Error checking document hash: {e}")
        
        logger.info(f"Creating new optimized index with {len(documents)} documents")
    
        try:
            # Extract text and metadata
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Create embeddings
            embeddings_list = self.embedding_manager.embed_documents(texts)
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings_list).astype('float32')
            
            # Create directory if it doesn't exist
            os.makedirs(self.index_path, exist_ok=True)
            
            # Determine if we should use IVF index (better for larger datasets)
            if len(documents) > 1000:
                # Use IVF index for larger datasets
                dimension = len(embeddings_list[0])
                nlist = min(int(len(documents) ** 0.5), 100)  # Number of clusters
                
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
                
                # Train the index
                index.train(embeddings_array)
                index.add(embeddings_array)
                
                logger.info(f"Created IVF index with {nlist} clusters")
            else:
                # Use flat index for smaller datasets
                index = faiss.IndexFlatL2(len(embeddings_list[0]))
                index.add(embeddings_array)
                logger.info("Created flat index")
            
            # Save the index
            faiss.write_index(index, os.path.join(self.index_path, "index.faiss"))
            
            # Save the texts and metadata separately for faster loading
            with open(os.path.join(self.index_path, "documents.pkl"), "wb") as f:
                pickle.dump({"texts": texts, "metadatas": metadatas}, f)
            
            logger.info(f"Index saved to {self.index_path}")
            try:
                with open(self._document_hash_file, "w") as f:
                    json.dump({"hash": current_hash, "timestamp": time.time()}, f)
            except Exception as e:
                logger.warning(f"Error saving document hash: {e}")
            # Create the FAISS wrapper
            self.index = FAISS(self._embeddings.embed_query, index, texts, metadatas)
            
            return self.index
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def _get_documents_hash(self, documents):
        """Generate a hash of the documents to detect changes"""
    
    # Create a hash based on document content and metadata
        doc_data = []
        for doc in documents:
            doc_data.append({
                "content_hash": hashlib.md5(doc.page_content.encode()).hexdigest(),
                "metadata": doc.metadata
            })
        
        # Generate a hash of the entire document set
        return hashlib.md5(json.dumps(doc_data, sort_keys=True).encode()).hexdigest()
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the existing index or create a new one if it doesn't exist."""
        if not documents:
            logger.warning("No documents provided to add")
            return
            
        # Ensure index is loaded
        self._load_index()
        
        if self.index is None:
            return self.create_index(documents)
        
        logger.info(f"Adding {len(documents)} documents to existing index")
        
        try:
            # Add documents to the index
            self.index.add_documents(documents)
            
            # Save the updated index
            self.index.save_local(self.index_path)
            
            # Clear the query cache as results may have changed
            self._query_cache.clear()
            
            logger.info(f"Index updated and saved to {self.index_path}")
            return self.index
        except Exception as e:
            logger.error(f"Error adding documents to index: {e}")
            raise
    
    def optimize_index(self):
        """Optimize the existing index for better performance."""
        # Ensure index is loaded
        self._load_index()
        
        if self.index is None:
            logger.warning("No index to optimize")
            return
        
        logger.info("Optimizing FAISS index...")
        
        try:
            # Get the current index data
            current_index = self.index
            
            # Extract texts and metadata
            texts = current_index.docstore._dict.values()
            metadatas = [doc.metadata for doc in texts]
            texts = [doc.page_content for doc in texts]
            
            # Create a new optimized index
            return self.create_index([Document(page_content=text, metadata=metadata) 
                                     for text, metadata in zip(texts, metadatas)])
        except Exception as e:
            logger.error(f"Error optimizing index: {e}")
            raise

def initialize_retriever(documents_path: str = "src/data/medical_books", 
                         embeddings_path: str = "src/data/embeddings"):
    """
    Initialize the document retriever with the specified documents and model.
    
    Args:
        documents_path: Path to the documents directory
        embeddings_path: Path to save/load embeddings
    """
    # Create a singleton instance
    global _retriever_instance
    if '_retriever_instance' not in globals():
        # Make sure the directories exist
        os.makedirs(documents_path, exist_ok=True)
        os.makedirs(embeddings_path, exist_ok=True)
        
        # Initialize the retriever with the specified paths
        _retriever_instance = MedicalDocumentRetriever(
            index_path=embeddings_path,
            lazy_loading=False  # Load immediately since this is a background task
        )
        
        # If no index exists, try to create one from documents
        if not os.path.exists(os.path.join(embeddings_path, "index.faiss")):
            try:
                from langchain_core.documents import Document
                from langchain_community.document_loaders import DirectoryLoader, TextLoader
                
                # Try to load documents from the documents directory
                loader = DirectoryLoader(
                    documents_path,
                    glob="**/*.pdf",
                    loader_cls=TextLoader
                )
                documents = loader.load()
                
                if documents:
                    _retriever_instance.create_index(documents)
                    logger.info(f"Created new index with {len(documents)} documents")
                else:
                    logger.warning(f"No documents found in {documents_path}")
            except Exception as e:
                logger.error(f"Error creating index: {str(e)}")
    else:
        logger.info("Retriever already initialized")
# Function to retrieve relevant documents using the class-based retriever
def retrieve_relevant_documents(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Retrieve the most relevant documents for a query using the MedicalDocumentRetriever.
    
    Args:
        query: The user's question
        top_k: Number of documents to retrieve
        
    Returns:
        Dictionary with query, documents, and scores
    """
    # Create a singleton instance or get existing one
    global _retriever_instance
    if '_retriever_instance' not in globals():
        _retriever_instance = MedicalDocumentRetriever(lazy_loading=True)
    
    # retriever = _retriever_instance
    if registry.get("retriever") is None:
        retriever = MedicalDocumentRetriever(lazy_loading=True)
    else:
        # Use the existing retriever from the registry
        logger.info("Using existing retriever from registry")
        retriever = registry.get("retriever")
    
    try:
        # Retrieve documents
        results = retriever.retrieve(query, k=top_k)
        
        # Format the results
        documents = []
        for doc in results:
            documents.append({
                "title": doc.metadata.get("title", "Unknown"),
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown")
            })
        
        return {
            "query": query,
            "documents": documents,
            "scores": [1.0] * len(documents)  # Placeholder scores
        }
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return {"query": query, "documents": [], "scores": []}

# Test Retrieval
if __name__ == "__main__":
    # Create a retriever with lazy loading
    retriever = MedicalDocumentRetriever(lazy_loading=True)
    
    # Test if the index exists
    if os.path.exists(os.path.join(retriever.index_path, "index.faiss")):
        # Test retrieval
        query = "What are the symptoms of pneumonia?"
        print(f"Retrieving documents for query: {query}")
        
        start_time = time.time()
        results = retriever.retrieve(query, k=3)
        elapsed = time.time() - start_time
        
        print(f"Retrieved {len(results)} documents in {elapsed:.2f} seconds")
        
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    else:
        print(f"No index found at {retriever.index_path}. Please run build_index.py first.")
