from pdf2image import convert_from_path
import pytesseract
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import os
import concurrent.futures
import pickle
import hashlib
from tqdm import tqdm
import logging
from langchain_core.documents import Document
from src.retrieval.document_retriever import MedicalDocumentRetriever
from src.utils.registry import registry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDocumentProcessor:
    def __init__(self, pdf_folder: str, image_output_folder: str, cache_dir: str = "src/data/cache"):
        self.pdf_folder = pdf_folder
        self.image_output_folder = image_output_folder
        self.cache_dir = cache_dir
        
        # Create necessary directories
        os.makedirs(image_output_folder, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, pdf_path):
        """Generate a cache file path based on the PDF path and modification time"""
        file_name = os.path.basename(pdf_path)
        mod_time = os.path.getmtime(pdf_path)
        hash_key = hashlib.md5(f"{file_name}_{mod_time}".encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}.pkl")

    def _load_pdf(self, pdf_path: str):
        """Load a PDF with caching"""
        cache_path = self._get_cache_path(pdf_path)
        
        # Check if cached version exists
        if os.path.exists(cache_path):
            logger.info(f"Loading cached content for {os.path.basename(pdf_path)}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # If no cache, load and cache the result
        logger.info(f"Processing PDF: {os.path.basename(pdf_path)}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Cache the result
        with open(cache_path, 'wb') as f:
            pickle.dump(docs, f)
            
        return docs

    def extract_text_and_images(self, use_parallel=True, max_workers=4):
        """Extract text from PDFs with parallel processing and caching"""
        all_text = []
        pdf_files = [os.path.join(self.pdf_folder, file) 
                    for file in os.listdir(self.pdf_folder) 
                    if file.endswith(".pdf")]
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        if use_parallel and len(pdf_files) > 1:
            # Process PDFs in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Use tqdm for a progress bar
                futures = {executor.submit(self._load_pdf, pdf_path): pdf_path for pdf_path in pdf_files}
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(pdf_files), desc="Processing PDFs"):
                    pdf_path = futures[future]
                    try:
                        text_docs = future.result()
                        all_text.extend([doc.page_content for doc in text_docs])
                        logger.info(f"Completed processing {os.path.basename(pdf_path)}")
                    except Exception as e:
                        logger.error(f"Error processing {os.path.basename(pdf_path)}: {e}")
        else:
            # Process PDFs sequentially
            for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
                try:
                    text_docs = self._load_pdf(pdf_path)
                    all_text.extend([doc.page_content for doc in text_docs])
                except Exception as e:
                    logger.error(f"Error processing {os.path.basename(pdf_path)}: {e}")
        
        logger.info(f"Extracted {len(all_text)} text chunks from {len(pdf_files)} PDFs")
        return all_text

    def extract_images_from_pdf(self, pdf_path: str, use_cache=True):
        """Extract images from a single PDF file with caching."""
        file_name = os.path.basename(pdf_path)
        cache_path = os.path.join(self.cache_dir, f"{file_name}_images.pkl")
        
        # Check if cached version exists
        if use_cache and os.path.exists(cache_path):
            logger.info(f"Using cached images for {file_name}")
            return
            
        try:
            logger.info(f"Extracting images from {file_name}")
            images = convert_from_path(pdf_path)
            
            image_paths = []
            for i, image in enumerate(images):
                image_path = os.path.join(self.image_output_folder, f"{file_name}_page_{i}.jpg")
                image.save(image_path, "JPEG")
                image_paths.append(image_path)
                
            # Cache the image paths
            if use_cache:
                with open(cache_path, 'wb') as f:
                    pickle.dump(image_paths, f)
                    
            logger.info(f"Extracted {len(images)} images from {file_name}")
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {e}")
    
    def process_single_pdf(self, pdf_path: str, extract_images=False):
        """Process a single PDF file and return its text content"""
        if not pdf_path.endswith(".pdf"):
            raise ValueError("File must be a PDF")

        text_docs = self._load_pdf(pdf_path)
        
        if extract_images:
            self.extract_images_from_pdf(pdf_path)
            
        return [doc.page_content for doc in text_docs]
    
    def bulk_process_directory(self, output_index_path="faiss_medical_index"):
        """Process all PDFs and create the FAISS index in one operation"""
        
        
        # Extract all text
        all_text = self.extract_text_and_images(use_parallel=True)
        
        # Convert text to documents
        documents = [Document(page_content=text) for text in all_text]
        
        # Import chunk_documents here to avoid circular import
        from src.data_processing.document_processor import chunk_documents
        
        # Split text into chunks
        logger.info("Splitting text into chunks...")
        chunks = chunk_documents(documents)
        logger.info(f"Created {len(chunks)} text chunks")
        
        # Create the retriever and build the index
        logger.info("Building FAISS index...")
        if registry.get("retriever") is None:
            retriever = MedicalDocumentRetriever(lazy_loading=True)
        else:
            # Use the existing retriever from the registry
            logger.info("Using existing retriever from registry")
            retriever = registry.get("retriever")
        retriever.create_index(chunks)
        logger.info(f"Index created successfully at {output_index_path}")
        
        return retriever

# Example usage
if __name__ == "__main__":
    processor = MedicalDocumentProcessor(
        pdf_folder="src/data/medical_books",
        image_output_folder="src/data/extracted_images"
    )
    
    # Option 1: Just extract text
    # documents = processor.extract_text_and_images(use_parallel=True)
    
    # Option 2: Process everything and build the index
    retriever = processor.bulk_process_directory()
    
    # Test the retriever
    if retriever.index is not None:
        results = retriever.retrieve("What are the symptoms of pneumonia?", k=2)
        print("\nTest query results:")
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
