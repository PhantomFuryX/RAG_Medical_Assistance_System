import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import os
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import pickle
import hashlib
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.logger import get_logger
from src.utils.registry import registry

logger = get_logger("data_processing")

def process_documents_in_parallel(doc_paths: List[str], 
                                 pdf_folder: str = "src/data/medical_books",
                                 image_output_folder: str = "src/data/extracted_images",
                                 cache_dir: str = "src/data/cache",
                                 extract_images: bool = False,
                                 max_workers: Optional[int] = None) -> List[Document]:
    """
    Process multiple documents in parallel using a process pool
    
    Args:
        doc_paths: List of paths to documents to process
        pdf_folder: Base folder for PDF documents
        image_output_folder: Folder to store extracted images
        cache_dir: Directory for caching processed documents
        extract_images: Whether to extract images from PDFs
        max_workers: Maximum number of worker processes (defaults to CPU count - 1)
        
    Returns:
        List of processed Document objects
    """
    if max_workers is None:
        # Use number of CPU cores minus 1 (leave one for the main process)
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    logger.info(f"Processing {len(doc_paths)} documents using {max_workers} workers")
    
    # Define a worker function that processes a single document
    def process_single_doc(doc_path: str) -> List[str]:
        try:
            
            # Generate a cache file path
            file_name = os.path.basename(doc_path)
            mod_time = os.path.getmtime(doc_path)
            hash_key = hashlib.md5(f"{file_name}_{mod_time}".encode()).hexdigest()
            cache_path = os.path.join(cache_dir, f"{hash_key}.pkl")
            
            # Check if cached version exists
            if os.path.exists(cache_path):
                logger.info(f"Loading cached content for {os.path.basename(doc_path)}")
                with open(cache_path, 'rb') as f:
                    docs = pickle.load(f)
                return [doc.page_content for doc in docs]
            
            # If no cache, load and cache the result
            logger.info(f"Processing PDF: {os.path.basename(doc_path)}")
            loader = PyPDFLoader(doc_path)
            docs = loader.load()
            
            # Cache the result
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(docs, f)
                
            # Extract images if requested
            if extract_images:
                try:
                    
                    file_name = os.path.basename(doc_path)
                    image_cache_path = os.path.join(cache_dir, f"{file_name}_images.pkl")
                    
                    # Check if cached version exists
                    if os.path.exists(image_cache_path):
                        logger.info(f"Using cached images for {file_name}")
                    else:
                        logger.info(f"Extracting images from {file_name}")
                        images = convert_from_path(doc_path)
                        
                        os.makedirs(image_output_folder, exist_ok=True)
                        image_paths = []
                        for i, image in enumerate(images):
                            image_path = os.path.join(image_output_folder, f"{file_name}_page_{i}.jpg")
                            image.save(image_path, "JPEG")
                            image_paths.append(image_path)
                            
                        # Cache the image paths
                        with open(image_cache_path, 'wb') as f:
                            pickle.dump(image_paths, f)
                            
                        logger.info(f"Extracted {len(images)} images from {file_name}")
                except Exception as e:
                    logger.error(f"Error extracting images from {doc_path}: {e}")
            
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
            return []
    
    all_text_content = []
    
    # Process documents in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_doc, path) for path in doc_paths]
        for future in futures:
            try:
                result = future.result()
                all_text_content.extend(result)
            except Exception as e:
                logger.error(f"Error retrieving document processing result: {e}")
    
    # Convert text content to Document objects
    documents = [Document(page_content=text) for text in all_text_content]
    
    logger.info(f"Processed {len(doc_paths)} documents into {len(documents)} Document objects")
    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks for more efficient processing
    
    Args:
        documents: List of Document objects
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of chunked Document objects
    """
    
    logger.info(f"Chunking {len(documents)} documents (size={chunk_size}, overlap={chunk_overlap})")
    
    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    # Split the documents
    chunked_docs = text_splitter.split_documents(documents)
    
    logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
    return chunked_docs

def process_and_chunk_documents(doc_paths: List[str], 
                               chunk_size: int = 1000, 
                               chunk_overlap: int = 200,
                               max_workers: Optional[int] = None) -> List[Document]:
    """
    Process documents and chunk them in one operation
    
    Args:
        doc_paths: List of paths to documents to process
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        max_workers: Maximum number of worker processes
        
    Returns:
        List of chunked Document objects
    """
    # Process the documents
    documents = process_documents_in_parallel(doc_paths, max_workers=max_workers)
    
    # Chunk the documents
    chunked_docs = chunk_documents(documents, chunk_size, chunk_overlap)
    
    return chunked_docs
