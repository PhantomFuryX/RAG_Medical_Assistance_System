from pdf2image import convert_from_path
import pytesseract
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import os
import concurrent.futures
import pickle
import hashlib
from tqdm import tqdm
from langchain_core.documents import Document
from src.utils.registry import registry
from src.utils.logger import get_data_logger

logger = get_data_logger()

class MedicalDocumentProcessor:
    def __init__(self, pdf_folder: str, image_output_folder: str, cache_dir: str = "src/data/cache"):
        self.pdf_folder = pdf_folder
        self.image_output_folder = image_output_folder
        self.cache_dir = cache_dir
        
        # Create necessary directories
        os.makedirs(image_output_folder, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Store in registry for reuse if not already there
        if not registry.has("document_processor"):
            registry.set("document_processor", self)
            logger.info("Document processor stored in registry")

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
        
        # Use PyPDFLoader from registry if available
        if registry.has("pdf_loader_class"):
            loader_class = registry.get("pdf_loader_class")
            loader = loader_class(pdf_path)
        else:
            loader = PyPDFLoader(pdf_path)
            registry.set("pdf_loader_class", PyPDFLoader)
            
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
        
        # Store PDF files in registry for potential reuse
        registry.set("pdf_files", pdf_files)
        
        if use_parallel and len(pdf_files) > 1:
            # Get thread pool executor from registry or create new one
            executor = None
            if registry.has("thread_pool_executor"):
                executor = registry.get("thread_pool_executor")
                if executor._max_workers != max_workers:
                    executor.shutdown(wait=True)
                    executor = None
            
            if executor is None or executor._shutdown:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
                registry.set("thread_pool_executor", executor)
            
            # Process PDFs in parallel
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
            
            # Use pdf2image converter from registry if available
            if registry.has("pdf2image_converter"):
                images = registry.get("pdf2image_converter")(pdf_path)
            else:
                images = convert_from_path(pdf_path)
                # Store the function in registry for future use
                registry.set("pdf2image_converter", convert_from_path)
            
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
    
    def process_documents_in_parallel(self, doc_paths, max_workers=None):
        """Process documents in parallel using a process pool"""
        if max_workers is None:
            max_workers = min(os.cpu_count(), 8)  # Use up to 8 cores
        
        logger.info(f"Processing {len(doc_paths)} documents using {max_workers} workers")
        
        # Create a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_path = {executor.submit(self._process_single_document, path): path for path in doc_paths}
            
            # Collect results
            results = []
            for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(doc_paths), desc="Processing documents"):
                try:
                    doc_chunks = future.result()
                    results.extend(doc_chunks)
                except Exception as e:
                    path = future_to_path[future]
                    logger.error(f"Error processing {path}: {e}")
        
        return results
    
    def _process_single_document(self, path):
        """Process a single document and return its chunks"""
        try:
            return self.process_single_pdf(path)
        except Exception as e:
            logger.error(f"Error processing document {path}: {e}")
            return []
    def process_all_documents(self):
        """Process all documents and return them as Document objects"""
        all_text = self.extract_text_and_images(use_parallel=True)
        documents = [Document(page_content=text) for text in all_text]
        return documents
    
    def bulk_process_directory(self, output_index_path="faiss_medical_index"):
        """Process all PDFs and create the FAISS index in one operation"""
        
        # Get all PDF files
        pdf_files = [os.path.join(self.pdf_folder, file) 
                    for file in os.listdir(self.pdf_folder) 
                    if file.endswith(".pdf")]
        
        # Process documents in parallel
        all_documents = self.process_documents_in_parallel(pdf_files)
        
        # Convert to Document objects
        documents = [Document(page_content=text) for text in all_documents]
        
        # Import chunk_documents here to avoid circular import
        from src.data_processing.document_processor import chunk_documents
        from src.retrieval.document_retriever import MedicalDocumentRetriever
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