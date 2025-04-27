import os
import logging
from typing import List, Dict, Any
import faiss
import numpy as np
from retrieval.document_retriever import DocumentRetriever

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self, index_path: str = None, embeddings_path: str = None):
        self.index_path = index_path or os.getenv("FAISS_INDEX_PATH", "src/data/faiss_index")
        self.embeddings_path = embeddings_path or os.getenv("EMBEDDINGS_PATH", "src/data/embeddings")
        self.retriever = DocumentRetriever(self.index_path, self.embeddings_path)
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the retrieval service"""
        if not self.is_initialized:
            try:
                self.retriever.load_index()
                self.is_initialized = True
                logger.info("Retrieval service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize retrieval service: {str(e)}")
                raise
    
    async def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve relevant context for a query
        
        Args:
            query: The user's question
            top_k: Number of documents to retrieve
            
        Returns:
            Concatenated text from relevant documents
        """
        if not self.is_initialized:
            await self.initialize()
            
        try:
            results = self.retriever.search(query, top_k=top_k)
            
            if not results:
                return ""
                
            # Concatenate the text from retrieved documents
            context = "\n\n".join([doc["text"] for doc in results])
            return context
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return ""
