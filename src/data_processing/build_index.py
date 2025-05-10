import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from src.retrieval.document_retriever import MedicalDocumentRetriever
from src.utils.logger import get_data_logger
from pathlib import Path
from src.data_processing.document_processor import process_and_chunk_documents
from src.utils.registry import registry
from src.utils.settings import settings

logger = get_data_logger()

def load_documents(data_dir="../data/medical_texts", file_types=None):
    """Load documents from the specified directory."""
    if file_types is None:
        file_types = ["pdf", "txt"]
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"Created directory {data_dir}. Please add medical files to this directory.")
        logger.info("Example: Add .pdf or .txt files containing medical information.")
        return []
    
    documents = []
    try:
        for ext in file_types:
            loader_cls = PyPDFLoader if ext == "pdf" else TextLoader
            loader = DirectoryLoader(data_dir, glob=f"**/*.{ext}", loader_cls=loader_cls)
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} {ext.upper()} documents from {data_dir}")
            documents.extend(docs)
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return documents

def split_documents(documents, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP):
    """Split documents into chunks for better embedding."""
    if not documents:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    return split_docs

def build_medical_index(documents_dir="src/data/medical_books", 
                       index_path="src/data/faiss_medical_index",
                       chunk_size=settings.CHUNK_SIZE,
                       chunk_overlap=settings.CHUNK_OVERLAP):
    """
    Build a FAISS index from medical documents, with incremental update support.
    """
    doc_paths = [os.path.join(documents_dir, f) for f in os.listdir(documents_dir) if f.endswith(".pdf")]
    logger.info(f"Found {len(doc_paths)} documents to process")

    # Check for incremental update
    from src.retrieval.document_retriever import update_index_incrementally
    updated_files = update_index_incrementally(documents_dir, index_path)
    if updated_files is False:
        # Full rebuild
        chunked_docs = process_and_chunk_documents(
            doc_paths=doc_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif isinstance(updated_files, list) and updated_files:
        # Only process new/modified files
        doc_paths = [os.path.join(documents_dir, f) for f in updated_files]
        chunked_docs = process_and_chunk_documents(
            doc_paths=doc_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        logger.info("No new or updated documents found. Skipping index update.")
        return

    # Create or update the retriever
    if registry.get("retriever") is None:
        retriever = MedicalDocumentRetriever(lazy_loading=True)
    else:
        logger.info("Using existing retriever from registry")
        retriever = registry.get("retriever")
    retriever.create_index(chunked_docs)
    
    logger.info(f"Index created successfully at {index_path}")
    return retriever

def main():
    # Load documents
    documents = load_documents()
    if not documents:
        logger.warning("No documents found. Please add PDF or text documents to the data directory.")
        return
    
    # Split documents into chunks
    split_docs = split_documents(documents)
    
    # Create the retriever and build the index
    retriever = MedicalDocumentRetriever(index_path="faiss_medical_index")
    retriever.create_index(split_docs)
    logger.info("Index created successfully!")
    
    # Test the retriever
    query = "What are the symptoms of pneumonia?"
    results = retriever.retrieve(query)
    logger.info("\nTest query results:")
    for i, doc in enumerate(results):
        logger.info(f"\nResult {i+1}:")
        content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        logger.info(content_preview)

if __name__ == "__main__":
    main()
