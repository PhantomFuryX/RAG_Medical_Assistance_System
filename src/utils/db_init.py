from pymongo import MongoClient, ASCENDING, DESCENDING
from src.utils.mongodb import get_database, get_chat_collection
from fastapi import BackgroundTasks
import os
from src.retrieval.document_retriever import initialize_retriever
from src.data_processing.document_processor import process_and_chunk_documents
from src.retrieval.document_retriever import MedicalDocumentRetriever
from src.data_processing.build_index import build_medical_index
from src.utils.registry import registry


import logging
from src.utils.db_manager import db_manager

logger = logging.getLogger(__name__)

async def init_database():
    """Initialize the database connection"""
    try:
        await db_manager.connect_with_retry()
        logger.info("Database initialized successfully")
        # Initialize document retriever in background
        BackgroundTasks().add_task(
            initialize_retriever,
            documents_path=os.environ.get("DOCUMENTS_PATH", "src/data/medical_books"),
            embeddings_path=os.environ.get("EMBEDDINGS_PATH", "src/data/embeddings")
        )
        return db_manager
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def initialize_retriever(documents_path="src/data/documents", 
                         embeddings_path="src/data/faiss_medical_index"):
    """Initialize the document retriever with the document corpus"""
        
    try:
        # Build the medical index
        retriever = build_medical_index(
            documents_dir=documents_path,
            index_path=embeddings_path
        )
        
        # Store the retriever in the registry
        registry.set("retriever", retriever)
        
        logger.info("Retriever initialized successfully")
        return retriever
    except Exception as e:
        logger.error(f"Error initializing retriever: {e}")
        return None

def init_app(app):
    """Initialize the application with database and retriever"""
    # Initialize the database
    db = init_database()
    
    # Initialize the retriever in the background
    @app.on_event("startup")
    def startup_event():
        BackgroundTasks().add_task(
            initialize_retriever,
            documents_path=os.environ.get("DOCUMENTS_PATH", "src/data/documents"),
            index_path=os.environ.get("INDEX_PATH", "src/data/faiss_medical_index")
        )
    
    return db

if __name__ == "__main__":
    init_database()
