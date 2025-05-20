from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import torch
import faiss.contrib.torch_utils
import os
import time
import threading
import numpy as np
import faiss
import json
import pickle
from typing import List, Dict, Any, Optional, Union, Tuple
import concurrent.futures
import hashlib
from dotenv import load_dotenv

from src.retrieval.embedding_manager import EmbeddingManager
from src.utils.registry import registry
from src.utils.logger import get_data_logger
from src.utils.settings import settings
# Load environment variables
load_dotenv()

# Configure logging
logger = get_data_logger()

class MedicalDocumentRetriever:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 index_path: str = "faiss_medical_index",
                 lazy_loading: bool = True,
                 retrieval_strategy: str = "hybrid"):
        """
        Initialize the enhanced medical document retriever.
        
        Args:
            embedding_model: The embedding model to use
            index_path: Path to store/load the FAISS index
            lazy_loading: Whether to load the index lazily
            retrieval_strategy: The retrieval strategy to use (hybrid, bm25, semantic, ensemble)
        """
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device} for embeddings")
        
        # Store parameters
        self.embedding_model_name = embedding_model
        self.index_path = index_path
        self.retrieval_strategy = retrieval_strategy
        self._embeddings = None
        self.vector_store = None
        self.retriever = None
        self.bm25_retriever = None
        self._loading_lock = threading.Lock()
        self._loading_thread = None
        self._loading_complete = threading.Event()
        self._document_hash_file = os.path.join(index_path, "document_hash.json")
        
        # Cache for query results
        self._query_cache = {}
        self._max_cache_size = 100
        
        # Check if we already have an instance in registry
        if registry.has("document_retriever"):
            existing_retriever = registry.get("document_retriever")
            if existing_retriever.index_path == index_path:
                # Copy the existing instance's attributes
                logger.info("Using existing retriever from registry")
                self._embeddings = existing_retriever._embeddings
                self.vector_store = existing_retriever.vector_store
                self.retriever = existing_retriever.retriever
                self.bm25_retriever = existing_retriever.bm25_retriever
                self._loading_complete.set()
                return
        
        # Get embedding manager from registry or create new one
        if registry.has("embedding_manager"):
            self.embedding_manager = registry.get("embedding_manager")
        else:
            self.embedding_manager = EmbeddingManager.get_instance(embedding_model)
            registry.set("embedding_manager", self.embedding_manager)
        
        # Initialize embeddings
        self._init_embeddings()
        
        # Load the index (potentially in background)
        if not lazy_loading:
            self._load_index()
        else:
            logger.info(f"FAISS index will be loaded lazily on first use")
        
        # Store in registry
        registry.set("document_retriever", self)
    
    def _init_embeddings(self):
        """Initialize the embeddings model"""
        try:
            # Check if embeddings are already in registry
            if registry.has("embeddings_model"):
                self._embeddings = registry.get("embeddings_model")
                logger.info("Using embeddings model from registry")
            else:
                self._embeddings = self.embedding_manager.model
                if self._embeddings:
                    registry.set("embeddings_model", self._embeddings)
                    logger.info(f"Initialized embeddings model and stored in registry")
                else:
                    logger.warning("Embedding manager returned None for model")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            self._embeddings = None
    
    def _batch_embed_documents(self, texts, batch_size=256):
        """Embed documents in batches to avoid memory issues."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embedding_manager.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        return embeddings
    
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
            
            # Load the vector store
            if self._embeddings is not None:
                self.vector_store = FAISS.load_local(
                    self.index_path, 
                    self._embeddings, 
                    allow_dangerous_deserialization=True
                )
            else:
                logger.error("Embeddings are not initialized")
                self._loading_complete.set()
                return
            
            # Create the appropriate retriever based on strategy
            self._setup_retriever()
            
            elapsed = time.time() - start_time
            logger.info(f"Completed background loading of FAISS index in {elapsed:.2f} seconds")
            
            # Store in registry
            registry.set("faiss_vector_store", self.vector_store)
            
            self._loading_complete.set()
        except Exception as e:
            logger.error(f"Error in background loading of FAISS index: {e}")
            self._loading_complete.set()
    
    def _load_index(self):
        """Load the index, potentially waiting if it's already loading"""
        # If already loaded, return immediately
        if self.vector_store is not None:
            return
        
        # Check if index is in registry
        if registry.has("faiss_vector_store"):
            self.vector_store = registry.get("faiss_vector_store")
            logger.info("Using FAISS vector store from registry")
            self._setup_retriever()
            self._loading_complete.set()
            return
        
        # If already loading in background, wait for it to complete
        if self._loading_thread is not None and self._loading_thread.is_alive():
            logger.info("Waiting for background index loading to complete...")
            self._loading_complete.wait()
            return
        
        # Acquire lock to prevent multiple threads from loading simultaneously
        with self._loading_lock:
            # Check again in case another thread loaded while waiting for lock
            if self.vector_store is not None:
                return
                
            # Start loading in background
            self._loading_thread = threading.Thread(target=self._load_index_in_background)
            self._loading_thread.daemon = True
            self._loading_thread.start()
            
            # For the first request, wait for loading to complete
            logger.info("Waiting for initial index loading to complete...")
            self._loading_complete.wait()
    
    def _setup_retriever(self):
        """Set up the appropriate retriever based on the strategy"""
        if self.vector_store is None:
            logger.warning("Cannot set up retriever: vector store is None")
            return
        
        # Create the base semantic search retriever
        semantic_retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Use Maximum Marginal Relevance for diversity
            search_kwargs={"k": 10, "fetch_k": 20}  # Fetch more candidates, then select diverse subset
        )
        
        # Create BM25 retriever if we have documents
        if hasattr(self.vector_store, "docstore") and hasattr(self.vector_store.docstore, "_dict"):
            docs = list(self.vector_store.docstore._dict.values()) # type: ignore
            texts = [doc.page_content for doc in docs]
            self.bm25_retriever = BM25Retriever.from_texts(texts)
            self.bm25_retriever.k = 10
        
        # Set up the retriever based on strategy
        if self.retrieval_strategy == "semantic":
            # Use semantic search only
            self.retriever = semantic_retriever
            logger.info("Using semantic search retriever")
            
        elif self.retrieval_strategy == "bm25" and self.bm25_retriever:
            # Use BM25 only
            self.retriever = self.bm25_retriever
            logger.info("Using BM25 retriever")
            
        elif self.retrieval_strategy == "ensemble" and self.bm25_retriever:
            # Combine BM25 and semantic search
            self.retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, semantic_retriever],
                weights=[0.5, 0.5]
            )
            logger.info("Using ensemble retriever (BM25 + semantic)")
            
        elif self.retrieval_strategy == "hybrid" and self.bm25_retriever:
            # Use contextual compression with LLM extraction
            try:
                # Try to set up a contextual compression retriever
                llm = ChatOpenAI(temperature=0)
                compressor = LLMChainExtractor.from_llm(llm)
                
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=semantic_retriever
                )
                
                # Create a hybrid retriever that combines BM25 and compressed semantic search
                self.retriever = EnsembleRetriever(
                    retrievers=[self.bm25_retriever, compression_retriever],
                    weights=[0.3, 0.7]
                )
                logger.info("Using hybrid retriever with contextual compression")
            except Exception as e:
                logger.warning(f"Failed to set up hybrid retriever: {e}. Falling back to ensemble.")
                # Fall back to ensemble retriever
                self.retriever = EnsembleRetriever(
                    retrievers=[self.bm25_retriever, semantic_retriever],
                    weights=[0.5, 0.5]
                )
        else:
            # Default to semantic search
            self.retriever = semantic_retriever
            logger.info(f"Using default semantic search retriever (strategy '{self.retrieval_strategy}' not available)")
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for a query with advanced retrieval techniques"""
        # Check cache first
        cache_key = f"{query}_{k}_{self.retrieval_strategy}"
        if cache_key in self._query_cache:
            logger.info(f"Cache hit for query: {query}")
            return self._query_cache[cache_key]
        
        # Check if we're using sharded indices
        shards_dir = os.path.join(self.index_path, "shards")
        shard_metadata_path = os.path.join(self.index_path, "shard_metadata.json")
        
        if os.path.exists(shards_dir) and os.path.exists(shard_metadata_path):
            # We're using sharded indices
            return self._retrieve_from_shards(query, k)
        
        # Ensure index is loaded
        self._load_index()
        
        if self.vector_store is None or self.retriever is None:
            logger.warning("No index or retriever available")
            return []
        
        # Perform retrieval
        try:
            # Use a safer approach that doesn't rely on direct map
            # First try using the retriever directly
            results = []
            try:
                # Try to use query expansion for better results
                expanded_queries = self._expand_query(query)
            
                all_results = []
                # Retrieve documents for each expanded query
                for q in expanded_queries:
                    query_results = self.retriever.get_relevant_documents(q)
                    all_results.extend(query_results)
            
                # Deduplicate results
                seen_contents = set()
                unique_results = []
                for doc in all_results:
                    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        unique_results.append(doc)
            
                # Limit to top k results
                results = unique_results[:k]
            except Exception as e:
                logger.warning(f"Error using retriever: {e}. Falling back to vector store.")
                # Fall back to vector store's similarity search
                if hasattr(self.vector_store, "similarity_search"):
                    results = self.vector_store.similarity_search(query, k=k)
        
            # Cache the results
            if len(self._query_cache) >= self._max_cache_size:
                # Remove oldest item
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[cache_key] = results
        
            return results
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            # Last resort fallback - try a very basic approach
            try:
                if hasattr(self.vector_store, "similarity_search"):
                    return self.vector_store.similarity_search(query, k=k)
                return []
            except:
                return []
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand the query to improve retrieval performance"""
        # Start with the original query
        queries = [query]
        
        # Simple rule-based expansion
        # Add a version without question marks
        if '?' in query:
            queries.append(query.replace('?', ''))
        
        # Add a version with "medical" prefix if not present
        if not any(term in query.lower() for term in ["medical", "health", "clinical", "doctor"]):
            queries.append(f"medical {query}")
        
        # Try to extract key terms
        terms = [word for word in query.split() if len(word) > 3 and word.lower() not in 
                ["what", "when", "where", "which", "who", "whom", "whose", "why", "how", 
                 "does", "is", "are", "was", "were", "will", "would", "should", "could", "have", "has", "had", "been", "being", "this", "that", "these", "those"]]
        
        if len(terms) >= 2:
            # Add a query with just the key terms
            queries.append(" ".join(terms))
        
        # Limit to 3 queries to avoid too much overhead
        return queries[:3]
    
    def create_index(self, documents: List[Document], use_sharding: bool):
        """Create a new optimized FAISS index from the provided documents."""
        if not documents:
            raise ValueError("No documents provided to create the index.")
        
        logger.info(f"Creating new optimized index with {len(documents)} documents")
        
        # Determine if we should use sharding
        if use_sharding is None:
            # Auto-determine based on document count
            use_sharding = len(documents) > 50000  # Threshold for using sharding
        
        if use_sharding:
            # Calculate optimal number of shards based on document count
            num_shards = max(4, min(32, int(len(documents) / 10000)))
            return self.create_sharded_index(documents, num_shards=num_shards)
        
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
                    return self.vector_store
            except Exception as e:
                logger.warning(f"Error checking document hash: {e}")
        
        logger.info(f"Creating new optimized index with {len(documents)} documents")
    
        try:
            # Extract text and metadata
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Create directory if it doesn't exist
            os.makedirs(self.index_path, exist_ok=True)
            
            if self._embeddings is None:
                raise ValueError("Embeddings are not initialized")

            # Create the vector store
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self._embeddings,
            )
            
            # Save the vector store
            self.vector_store.save_local(self.index_path)
            
            # Set up the retriever
            self._setup_retriever()
            
            # Create BM25 retriever
            self.bm25_retriever = BM25Retriever.from_texts(texts)
            self.bm25_retriever.k = 10
            
            # Save BM25 retriever
            with open(os.path.join(self.index_path, "bm25_retriever.pkl"), "wb") as f:
                pickle.dump(self.bm25_retriever, f)
            
            logger.info(f"Index saved to {self.index_path}")
            
            # Save document hash
            try:
                with open(self._document_hash_file, "w") as f:
                    json.dump({"hash": current_hash, "timestamp": time.time()}, f)
            except Exception as e:
                logger.warning(f"Error saving document hash: {e}")
            
            # Store in registry
            registry.set("faiss_vector_store", self.vector_store)
            
            return self.vector_store
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
        
        if self.vector_store is None:
            return self.create_index(documents, use_sharding=False)
        
        logger.info(f"Adding {len(documents)} documents to existing index")
        
        try:
            # Add documents to the vector store
            self.vector_store.add_documents(documents)
            
            # Save the updated vector store
            self.vector_store.save_local(self.index_path)
            
            # Update BM25 retriever
            if self.bm25_retriever:
                texts = [doc.page_content for doc in documents]
                self.bm25_retriever.from_texts(texts)
                
                # Save updated BM25 retriever
                with open(os.path.join(self.index_path, "bm25_retriever.pkl"), "wb") as f:
                    pickle.dump(self.bm25_retriever, f)
            
            # Clear the query cache as results may have changed
            self._query_cache.clear()
            
            # Update registry
            registry.set("faiss_vector_store", self.vector_store)
            
            logger.info(f"Index updated and saved to {self.index_path}")
            return self.vector_store
        except Exception as e:
            logger.error(f"Error adding documents to index: {e}")
            raise
    
    def optimize_index(self):
        """Optimize the existing index for better performance."""
        # Ensure index is loaded
        self._load_index()
        
        if self.vector_store is None:
            logger.warning("No index to optimize")
            return
        
        logger.info("Optimizing FAISS index...")
        
        try:
            # Get the current documents
            docs = list(self.vector_store.docstore._dict.values()) # type: ignore
            
            # Create a new optimized index
            return self.create_index(docs, use_sharding=False)
        except Exception as e:
            logger.error(f"Error optimizing index: {e}")
            raise
    
    def _retrieve_from_shards(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve documents from sharded indices"""
        logger.info(f"Retrieving from sharded indices for query: {query}")
        
        # Load shard metadata
        shard_metadata_path = os.path.join(self.index_path, "shard_metadata.json")
        with open(shard_metadata_path, 'r') as f:
            shard_metadata = json.load(f)
        
        num_shards = shard_metadata.get("num_shards", 0)
        if num_shards == 0:
            logger.warning("No shards found in metadata")
            return []
        
        # We'll retrieve k*2 results from each shard to ensure we have enough candidates
        shard_k = min(k * 2, 100)  # Cap at 100 to avoid excessive memory usage
        
        # Create embedding for the query (only once)
        query_embedding = self.embedding_manager.embed_query(query)
        
        # Expand the query for better retrieval
        expanded_queries = self._expand_query(query)
        expanded_embeddings = [
            self.embedding_manager.embed_query(q) for q in expanded_queries
        ]
        
        # Use ThreadPoolExecutor to query shards in parallel
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_shards, 8)) as executor:
            future_to_shard = {}
            
            # Submit tasks to query each shard for each expanded query
            for shard_id in range(num_shards):
                for q_idx, q_embedding in enumerate(expanded_embeddings):
                    future = executor.submit(
                        self._query_single_shard, 
                        shard_id, 
                        q_embedding, 
                        shard_k // len(expanded_embeddings)
                    )
                    future_to_shard[future] = (shard_id, q_idx)
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_shard):
                shard_id, q_idx = future_to_shard[future]
                try:
                    shard_results = future.result()
                    all_results.extend(shard_results)
                    logger.debug(f"Retrieved {len(shard_results)} results from shard {shard_id} for query {q_idx}")
                except Exception as e:
                    logger.error(f"Error querying shard {shard_id} for query {q_idx}: {e}")
        
        # Deduplicate results
        seen_contents = set()
        unique_results = []
        for doc, score in all_results:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append((doc, score))
        
        # Sort all results by similarity score (ascending for L2 distance)
        unique_results.sort(key=lambda x: x[1])
        
        # Take top-k results
        top_k_results = unique_results[:k]
        
        # Convert to Document objects
        documents = [doc for doc, score in top_k_results]
        
        # Cache the results
        cache_key = f"{query}_{k}_{self.retrieval_strategy}"
        if len(self._query_cache) >= self._max_cache_size:
            # Remove oldest item
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[cache_key] = documents
        
        logger.info(f"Retrieved {len(documents)} documents from {num_shards} shards")
        return documents

    def _query_single_shard(self, shard_id: int, query_embedding: List[float], k: int) -> List[Tuple[Document, float]]:
        """Query a single shard and return (document, score) tuples"""
        shard_path = os.path.join(self.index_path, "shards", f"shard_{shard_id}")
        index_path = os.path.join(shard_path, "index.faiss")
        documents_path = os.path.join(shard_path, "documents.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(documents_path):
            logger.warning(f"Missing files for shard {shard_id}")
            return []
        
        try:
            # Load the index
            index = faiss.read_index(index_path)
            
            # Load documents
            with open(documents_path, 'rb') as f:
                doc_data = pickle.load(f)
                texts = doc_data["texts"]
                metadatas = doc_data["metadatas"]
            
            # Convert query embedding to numpy array
            query_array = np.array([query_embedding]).astype('float32')
            
            # Search the index
            distances, indices = index.search(query_array, k)
            
            # Create Document objects with scores
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(texts):  # -1 means no result
                    doc = Document(
                        page_content=texts[idx],
                        metadata=metadatas[idx] if idx < len(metadatas) else {}
                    )
                    results.append((doc, distances[0][i]))
            
            return results
        except Exception as e:
            logger.error(f"Error querying shard {shard_id}: {e}")
            return []
    
    def create_sharded_index(self, documents: List[Document], num_shards: int = 4):
        """
        Create multiple smaller indices (shards) instead of one large index
        
        Args:
            documents: List of documents to index
            num_shards: Number of shards to create
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            logger.warning("No documents provided to create sharded index")
            return False
        
        logger.info(f"Creating sharded index with {len(documents)} documents in {num_shards} shards")
        
        # Create directory for shards
        shards_dir = os.path.join(self.index_path, "shards")
        os.makedirs(shards_dir, exist_ok=True)
        
        # Calculate documents per shard
        docs_per_shard = len(documents) // num_shards
        if docs_per_shard == 0:
            docs_per_shard = 1
            num_shards = min(num_shards, len(documents))
        
        dimensions = []
        # Process each shard
        for i in range(num_shards):
            start_idx = i * docs_per_shard
            end_idx = start_idx + docs_per_shard if i < num_shards - 1 else len(documents)
            shard_docs = documents[start_idx:end_idx]
            
            logger.info(f"Processing shard {i+1}/{num_shards} with {len(shard_docs)} documents")
            
            # Create shard directory
            shard_path = os.path.join(shards_dir, f"shard_{i}")
            os.makedirs(shard_path, exist_ok=True)
            
            # Extract text and metadata
            texts = [doc.page_content for doc in shard_docs]
            metadatas = [doc.metadata for doc in shard_docs]
            
            # Create embeddings
            embeddings_list = self._batch_embed_documents(texts)
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings_list).astype('float32')
            
            # Create FAISS index for this shard
            dimension = len(embeddings_array[0]) if embeddings_array.size > 0 else 0
            dimensions.append(dimension)
            if dimension == 0:
                logger.error("Embeddings array is empty. Skipping shard.")
                continue
            index = faiss.IndexFlatL2(dimension)
            
            # Add vectors to index
            index.add(embeddings_array, len(embeddings_array))
            
            # Save the index
            faiss.write_index(index, os.path.join(shard_path, "index.faiss"))
            
            # Save the texts and metadata
            with open(os.path.join(shard_path, "documents.pkl"), "wb") as f:
                pickle.dump({"texts": texts, "metadatas": metadatas}, f)
            
            logger.info(f"Completed shard {i+1}/{num_shards}")
        
        # Create shard metadata
        shard_metadata = {
            "num_shards": num_shards,
            "total_documents": len(documents),
            "created_at": time.time(),
            "dimension": dimensions[-1]
        }
        
        with open(os.path.join(self.index_path, "shard_metadata.json"), "w") as f:
            json.dump(shard_metadata, f)
        
        logger.info(f"Created sharded index with {num_shards} shards")
        return True

    def get_retriever(self):
        """Get the underlying retriever for direct use in LangChain chains"""
        self._load_index()
        return self.retriever
    
    def rerank_results(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        Rerank the retrieved documents using a cross-encoder model for better relevance.
        
        Args:
            query: The user query
            documents: The retrieved documents
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of documents
        """
        try:
            from sentence_transformers import CrossEncoder
            
            # Check if we have a cross-encoder in registry
            if registry.has("cross_encoder"):
                model = registry.get("cross_encoder")
            else:
                # Initialize cross-encoder model
                model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                registry.set("cross_encoder", model)
            
            # Prepare document pairs for scoring
            doc_pairs = [[query, doc.page_content] for doc in documents]
            
            # Score document pairs
            scores = model.predict(doc_pairs)
            
            # Create document-score pairs
            doc_score_pairs = list(zip(documents, scores))
            
            # Sort by score in descending order
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k documents
            return [doc for doc, score in doc_score_pairs[:top_k]]
        except ImportError:
            logger.warning("sentence-transformers not available for reranking. Using original order.")
            return documents[:top_k]
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return documents[:top_k]
    
    def search_with_metadata_filter(self, query: str, filter_dict: Dict[str, Any], k: int = 5) -> List[Document]:
        """
        Search documents with metadata filtering
        
        Args:
            query: The search query
            filter_dict: Dictionary of metadata filters
            k: Number of documents to retrieve
            
        Returns:
            List of matching documents
        """
        self._load_index()
        
        if self.vector_store is None:
            logger.warning("No vector store available for metadata search")
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            # Extract just the documents
            documents = [doc for doc, score in results]
            return documents
        except Exception as e:
            logger.error(f"Error during metadata search: {e}")
            return []
    
    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform pure semantic search without hybrid retrieval
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of matching documents
        """
        self._load_index()
        
        if self.vector_store is None:
            logger.warning("No vector store available for semantic search")
            return []
        
        try:
            # Use MMR search for better diversity
            results = self.vector_store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=k*3  # Fetch more candidates for diversity
            )
            return results
        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []
    
    def keyword_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform keyword-based search using BM25
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of matching documents
        """
        self._load_index()
        
        if self.bm25_retriever is None:
            logger.warning("No BM25 retriever available for keyword search")
            return []
        
        try:
            # Set the number of documents to retrieve
            self.bm25_retriever.k = k
            
            # Perform retrieval
            results = self.bm25_retriever.get_relevant_documents(query)
            return results
        except Exception as e:
            logger.error(f"Error during keyword search: {e}")
            return []
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform hybrid search combining semantic and keyword search
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of matching documents
        """
        # This is essentially the same as retrieve() but with explicit hybrid approach
        semantic_results = self.semantic_search(query, k=k)
        
        if self.bm25_retriever is None:
            return semantic_results
        
        keyword_results = self.keyword_search(query, k=k)
        
        # Combine results
        all_results = semantic_results + keyword_results
        
        # Deduplicate
        seen_contents = set()
        unique_results = []
        for doc in all_results:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(doc)
        
        # Rerank combined results
        return self.rerank_results(query, unique_results, top_k=k)


def initialize_retriever(documents_path: str = "src/data/medical_books", 
                         embeddings_path: str = "src/data/embeddings",
                         retrieval_strategy: str = "hybrid"):
    """
    Initialize the document retriever with the specified documents and model.
    
    Args:
        documents_path: Path to the documents directory
        embeddings_path: Path to save/load embeddings
        retrieval_strategy: The retrieval strategy to use
    """
    # Check if retriever already exists in registry
    if registry.has("document_retriever"):
        logger.info("Retriever already initialized in registry")
        return registry.get("document_retriever")
    
    # Make sure the directories exist
    os.makedirs(documents_path, exist_ok=True)
    os.makedirs(embeddings_path, exist_ok=True)
    
    # Initialize the retriever with the specified paths
    retriever = MedicalDocumentRetriever(
        index_path=embeddings_path,
        lazy_loading=False,
        retrieval_strategy=retrieval_strategy
    )

    # Incremental update logic
    updated_files = update_index_incrementally(documents_path, embeddings_path)
    if updated_files is False:
        # Full rebuild needed
        loader = DirectoryLoader(
            documents_path,
            glob="**/*.pdf",
            loader_cls=TextLoader
        )
        documents = loader.load()
        if documents:
            retriever.create_index(documents, use_sharding=True)
            logger.info(f"Created new index with {len(documents)} documents")
        else:
            logger.warning(f"No documents found in {documents_path}")
    elif isinstance(updated_files, list) and updated_files:
        # Only process new/modified files
        loader = DirectoryLoader(
            documents_path,
            glob="**/*.pdf",
            loader_cls=TextLoader
        )
        all_documents = loader.load()
        # Filter only updated files
        documents = [doc for doc in all_documents if os.path.basename(doc.metadata.get("source", "")) in updated_files]
        if documents:
            retriever.add_documents(documents)
            logger.info(f"Incrementally updated index with {len(documents)} documents")
        else:
            logger.warning(f"No documents found in {documents_path}")
    else:
        logger.info("No new or modified documents found, using existing index")
    
    # Store in registry
    registry.set("document_retriever", retriever)
    return retriever

# Function to retrieve relevant documents using the enhanced retriever
def retrieve_relevant_documents(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Retrieve the most relevant documents for a query using the MedicalDocumentRetriever.
    
    Args:
        query: The user's question
        top_k: Number of documents to retrieve
        
    Returns:
        Dictionary with query, documents, and scores
    """
    # Get retriever from registry or create new one
    if registry.has("document_retriever"):
        retriever = registry.get("document_retriever")
        logger.info("Using retriever from registry")
    else:
        logger.info("Creating new retriever instance")
        retriever = MedicalDocumentRetriever(lazy_loading=True)
        registry.set("document_retriever", retriever)
    
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

def update_index_incrementally(documents_path, index_path):
    """Update the index incrementally by only processing new or modified documents"""
    # Get list of all PDF files
    pdf_files = [f for f in os.listdir(documents_path) if f.lower().endswith(".pdf")]
    
    # Check if index exists
    if not os.path.exists(os.path.join(index_path, "index.faiss")):
        logger.info("No existing index found, creating new index")
        return False  # Need to build full index
    
    # Load document tracking file
    tracking_file = os.path.join(index_path, "document_tracking.json")
    if os.path.exists(tracking_file):
        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)
    else:
        tracking_data = {"documents": {}}
    
    # Check for new or modified documents
    new_or_modified = []
    for pdf_file in pdf_files:
        file_path = os.path.join(documents_path, pdf_file)
        file_mtime = os.path.getmtime(file_path)
        file_size = os.path.getsize(file_path)
        
        # Create a signature for the file
        file_signature = f"{file_mtime}_{file_size}"
        
        # Check if file is new or modified
        if pdf_file not in tracking_data["documents"] or tracking_data["documents"][pdf_file] != file_signature:
            new_or_modified.append(pdf_file)
            tracking_data["documents"][pdf_file] = file_signature
    
    # If no new or modified documents, no need to update
    if not new_or_modified:
        logger.info("No new or modified documents found, using existing index")
        return True  # Can use existing index
    
    # If more than 30% of documents are new/modified, rebuild the entire index
    if len(new_or_modified) > 0.3 * len(pdf_files):
        logger.info(f"{len(new_or_modified)} new/modified documents (>30%), rebuilding entire index")
        return False  # Need to build full index
    
    logger.info(f"Found {len(new_or_modified)} new or modified documents, updating index incrementally")
    
    # Save updated tracking data
    with open(tracking_file, 'w') as f:
        json.dump(tracking_data, f)
    
    return new_or_modified  # Return list of files to process incrementally
