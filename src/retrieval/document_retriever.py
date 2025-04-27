from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import torch
import faiss.contrib.torch_utils
import os
import time
import threading
import numpy as np
import faiss
import json
import pickle
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import concurrent.futures

from src.retrieval.embedding_manager import EmbeddingManager
import hashlib
import json
from src.utils.registry import registry
from src.utils.logger import get_data_logger

# Configure logging
logger = get_data_logger()

class MedicalDocumentRetriever:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 index_path: str = "faiss_medical_index",
                 lazy_loading: bool = True):
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device} for embeddings")
        
        # Store parameters
        self.embedding_model_name = embedding_model
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
        
        # Check if we already have an instance in registry
        if registry.has("document_retriever"):
            existing_retriever = registry.get("document_retriever")
            if existing_retriever.index_path == index_path:
                # Copy the existing instance's attributes
                logger.info("Using existing retriever from registry")
                self._embeddings = existing_retriever._embeddings
                self.index = existing_retriever.index
                self._loading_complete.set()
                return
        
        # Get embedding manager from registry or create new one
        if registry.has("embedding_manager"):
            self.embedding_manager = registry.get("embedding_manager")
        else:
            self.embedding_manager = EmbeddingManager.get_instance(embedding_model)
            registry.set("embedding_manager", self.embedding_manager)
        
        # Initialize embeddings (this is relatively fast)
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
            
            # Load the index
            self.index = FAISS.load_local(
                self.index_path, 
                self._embeddings, 
                allow_dangerous_deserialization=True
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Completed background loading of FAISS index in {elapsed:.2f} seconds")
            
            # Store in registry
            registry.set("faiss_index", self.index)
            
            self._loading_complete.set()
        except Exception as e:
            logger.error(f"Error in background loading of FAISS index: {e}")
            self._loading_complete.set()
    
    def _load_index(self):
        """Load the index, potentially waiting if it's already loading"""
        # If already loaded, return immediately
        if self.index is not None:
            return
        
        # Check if index is in registry
        if registry.has("faiss_index"):
            self.index = registry.get("faiss_index")
            logger.info("Using FAISS index from registry")
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
        """Retrieve relevant documents for a query with caching and shard support"""
        # Check cache first
        cache_key = f"{query}_{k}"
        if cache_key in self._query_cache:
            logger.info(f"Cache hit for query: {query}")
            return self._query_cache[cache_key]
        
        # Check if we're using sharded indices
        shards_dir = os.path.join(self.index_path, "shards")
        shard_metadata_path = os.path.join(self.index_path, "shard_metadata.json")
        
        if os.path.exists(shards_dir) and os.path.exists(shard_metadata_path):
            # We're using sharded indices
            return self._retrieve_from_shards(query, k)
        
        # Regular non-sharded retrieval
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
    
    def create_index(self, documents: List[Document], use_sharding: bool = None):
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
                    return self.index
            except Exception as e:
                logger.warning(f"Error checking document hash: {e}")
        
        logger.info(f"Creating new optimized index with {len(documents)} documents")
    
        try:
            # Extract text and metadata
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Create embeddings - use embedding manager from registry
            embeddings_list = self._batch_embed_documents(texts)
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings_list).astype('float32')
            
            # Create directory if it doesn't exist
            os.makedirs(self.index_path, exist_ok=True)
            
            # Determine if we should use IVF index (better for larger datasets)
            if len(documents) > 1000:
                # Use IVF index for larger datasets
                dimension = len(embeddings_list[0])
                nlist = min(int(len(documents) ** 0.5), 100)  # Number of clusters
                
                # Check if we have a GPU-enabled FAISS
                use_gpu = registry.get("use_gpu_faiss", False)
                
                if use_gpu:
                    logger.info("Using GPU for FAISS indexing")
                    
                    # Create GPU index
                    res = faiss.StandardGpuResources()
                    quantizer = faiss.IndexFlatL2(dimension)
                    gpu_quantizer = faiss.index_cpu_to_gpu(res, 0, quantizer)
                    index = faiss.IndexIVFFlat(gpu_quantizer, dimension, nlist, faiss.METRIC_L2)
                else:
                    # Use CPU index
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
            
            # Store in registry
            registry.set("faiss_index", self.index)
            
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
            
            # Update registry
            registry.set("faiss_index", self.index)
            
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
        
        # Use ThreadPoolExecutor to query shards in parallel
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_shards, 8)) as executor:
            future_to_shard = {}
            
            # Submit tasks to query each shard
            for shard_id in range(num_shards):
                future = executor.submit(
                    self._query_single_shard, 
                    shard_id, 
                    query_embedding, 
                    shard_k
                )
                future_to_shard[future] = shard_id
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_shard):
                shard_id = future_to_shard[future]
                try:
                    shard_results = future.result()
                    all_results.extend(shard_results)
                    logger.debug(f"Retrieved {len(shard_results)} results from shard {shard_id}")
                except Exception as e:
                    logger.error(f"Error querying shard {shard_id}: {e}")
        
        # Sort all results by similarity score (ascending for L2 distance)
        all_results.sort(key=lambda x: x[1])
        
        # Take top-k results
        top_k_results = all_results[:k]
        
        # Convert to Document objects
        documents = [doc for doc, score in top_k_results]
        
        # Cache the results
        cache_key = f"{query}_{k}"
        if len(self._query_cache) >= self._max_cache_size:
            # Remove oldest item
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[cache_key] = documents
        
        logger.info(f"Retrieved {len(documents)} documents from {num_shards} shards")
        return documents

    def _query_single_shard(self, shard_id: int, query_embedding: List[float], k: int) -> List[tuple]:
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
            dimension = len(embeddings_list[0])
            index = faiss.IndexFlatL2(dimension)
            
            # Add vectors to index
            index.add(embeddings_array)
            
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
            "dimension": dimension
        }
        
        with open(os.path.join(self.index_path, "shard_metadata.json"), "w") as f:
            json.dump(shard_metadata, f)
        
        logger.info(f"Created sharded index with {num_shards} shards")
        return True
def initialize_retriever(documents_path: str = "src/data/medical_books", 
                         embeddings_path: str = "src/data/embeddings"):
    """
    Initialize the document retriever with the specified documents and model.
    
    Args:
        documents_path: Path to the documents directory
        embeddings_path: Path to save/load embeddings
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
        lazy_loading=False
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
            retriever.create_index(documents)
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
            logger.info(f"Incretally udated idex with {len(documents)} documents")
        else:
            logger.warning(f"No documents found in {documents_path}")
    else:
        logger.info("No new or modified documents found, using existing index")
    # Store in registry
    registry.set("document_retriever", retriever)
    return retriever

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