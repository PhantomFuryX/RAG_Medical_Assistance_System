import os
import logging
from typing import List, Dict, Any
import faiss
import numpy as np
from retrieval.document_retriever import MedicalDocumentRetriever
from utils.logger import get_logger 

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self, index_path: str, embeddings_path: str):
        self.index_path = index_path or os.getenv("FAISS_INDEX_PATH", "src/data/faiss_index")
        self.embeddings_path = embeddings_path or os.getenv("EMBEDDINGS_PATH", "src/data/embeddings")
        self.retriever = MedicalDocumentRetriever(self.index_path, self.embeddings_path)
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the retrieval service"""
        if not self.is_initialized:
            try:
                self.retriever._load_index()
                self.is_initialized = True
                logger.info("Retrieval service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize retrieval service: {str(e)}")
                raise
    
    async def get_relevant_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
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
            results = self.retriever.retrieve(query, k=top_k)

            if not results:
                return {"context": "", "documents": []}
            context = "\n\n".join([doc.page_content for doc in results])
            return {
                "context": context,
                "documents": results  # includes metadata if available
            }
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return {"context": "", "documents": []}
