import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from typing import List
from fastapi.responses import JSONResponse
from src.data_processing.document_loader import MedicalDocumentProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from src.main.pydentic_models.models import DTQueryRequest, DTQueryResponse
from src.retrieval.document_retriever import MedicalDocumentRetriever
from src.data_processing.document_processor import chunk_documents
import tempfile
from src.utils.logger import get_api_logger
from src.utils.registry import registry
from src.utils.settings import settings

logger = get_api_logger()
router = APIRouter(prefix="/documents", tags=["DOCUMENTS"])

# Define directories
MEDICAL_BOOKS_DIR = "src/data/medical_books"
EXTRACTED_IMAGES_DIR = "src/data/extracted_images"
FAISS_INDEX_DIR = "faiss_medical_index"
processing_status = {}

# Ensure directories exist
os.makedirs(MEDICAL_BOOKS_DIR, exist_ok=True)
os.makedirs(EXTRACTED_IMAGES_DIR, exist_ok=True)

# Initialize the retriever
retriever = MedicalDocumentRetriever(index_path=FAISS_INDEX_DIR)
index_loaded = retriever.vector_store is not None

def get_retriever():
    if registry.get("retriever") is None:
            retriever = MedicalDocumentRetriever(index_path=FAISS_INDEX_DIR)
    else:
        # Use the existing retriever from the registry
        logger.info("Using existing retriever from registry")
        retriever = registry.get("retriever")
    return retriever

def process_document(file_path: str) -> List[Document]:
    """Process a document file and return a list of Document objects."""
    if file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    documents = loader.load()
    
    split_docs = chunk_documents(documents, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    return split_docs

def update_vector_database(file_path: str):
    """Background task to process the new PDF and update the vector database"""
    file_name = os.path.basename(file_path)
    processing_status[file_name] = "processing"
    
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device} for processing {file_name}")
        
        # Process the new document
        processor = MedicalDocumentProcessor(
            pdf_folder=MEDICAL_BOOKS_DIR,
            image_output_folder=EXTRACTED_IMAGES_DIR
        )
        
        # Extract text and images from the specific file
        if file_path.lower().endswith('.pdf'):
            # Extract text from the specific file
            all_text = []
            loader = processor._load_pdf(file_path)
            text_docs = loader
            all_text.extend([doc.page_content for doc in text_docs])
            
            # Extract images
            processor.extract_images_from_pdf(file_path)
        elif file_path.lower().endswith('.txt'):
            # For text files, just load the content
            with open(file_path, 'r', encoding='utf-8') as f:
                all_text = [f.read()]
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Split documents into chunks
        chunks = chunk_documents(all_text, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        
        # Create embeddings with CUDA support
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device}
        )
        
        # Check if index exists and load it, otherwise create new
        if os.path.exists(FAISS_INDEX_DIR):
            # Load existing index
            vectorstore = FAISS.load_local(
                FAISS_INDEX_DIR, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            # Add new documents to the existing index
            vectorstore.add_documents(chunks)
        else:
            # Create new index
            vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Save the updated index
        vectorstore.save_local(FAISS_INDEX_DIR)
        
        # Update the global retriever with the new index
        global retriever, index_loaded
        retriever = get_retriever()
        index_loaded = retriever.vector_store is not None
        
        processing_status[file_name] = "completed"
        print(f"Successfully processed {file_path} and updated the vector database using {device}")
    except Exception as e:
        processing_status[file_name] = f"failed: {str(e)}"
        print(f"Error updating vector database: {str(e)}")

@router.post("/upload-medical-document")
async def upload_medical_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None)
):
    """Upload a document, save it to the medical_books folder, and update the vector database."""
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")
    if file.filename is None:
        raise HTTPException(status_code=400, detail="File name is missing")
    # Check file type
    if not (file.filename.lower().endswith('.pdf') or file.filename.lower().endswith('.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
    
    # Save the file to the medical_books directory
    file_path = os.path.join(str(MEDICAL_BOOKS_DIR), file.filename)
    
    try:
        # Create the file in the medical_books directory
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Schedule the background task to process the document and update the vector database
        background_tasks.add_task(update_vector_database, file_path)
        
        return JSONResponse(
            content={
                "message": f"Document '{file.filename}' uploaded successfully. Processing started in the background.",
                "status": "processing",
                "file_path": file_path
            },
            status_code=202  # Accepted
        )
    except Exception as e:
        # If an error occurs, clean up and return an error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@router.post("/query", response_model=DTQueryResponse)
def query_documents(request: DTQueryRequest, 
                    retriever: MedicalDocumentRetriever = Depends(get_retriever)):
    if not retriever or not index_loaded:
        return DTQueryResponse(
            results=[],
            success=False,
            message="FAISS index not loaded. Please upload documents first."
        )
    
    try:
        results = retriever.retrieve(request.query, k=request.top_k)
        return DTQueryResponse(
            results=[doc.page_content for doc in results],
            success=True,
            message=f"Retrieved {len(results)} documents"
        )
    except Exception as e:
        return DTQueryResponse(
            results=[],
            success=False,
            message=f"Error during retrieval: {str(e)}"
        )

@router.get("/document-processing-status/{file_name}")
async def get_processing_status(file_name: str):
    if file_name in processing_status:
        return {"file_name": file_name, "status": processing_status[file_name]}
    return {"file_name": file_name, "status": "not_found"}

@router.post("/rebuild-index")
async def rebuild_index(background_tasks: BackgroundTasks):
    """Rebuild the vector database from all documents in the medical_books folder."""
    # Check if the medical_books directory exists and has files
    if not os.path.exists(MEDICAL_BOOKS_DIR):
        raise HTTPException(status_code=404, detail=f"Directory {MEDICAL_BOOKS_DIR} not found")
    
    files = [f for f in os.listdir(MEDICAL_BOOKS_DIR) 
             if os.path.isfile(os.path.join(MEDICAL_BOOKS_DIR, f)) and 
             (f.lower().endswith('.pdf') or f.lower().endswith('.txt'))]
    
    if not files:
        raise HTTPException(status_code=404, detail=f"No PDF or TXT files found in {MEDICAL_BOOKS_DIR}")
    
    # Remove existing index if it exists
    if os.path.exists(FAISS_INDEX_DIR):
        try:
            for file in os.listdir(FAISS_INDEX_DIR):
                file_path = os.path.join(FAISS_INDEX_DIR, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error removing existing index: {str(e)}")
    
    # Process each file in the directory
    for file_name in files:
        file_path = os.path.join(MEDICAL_BOOKS_DIR, file_name)
        background_tasks.add_task(update_vector_database, file_path)
    
    return JSONResponse(
        content={
            "message": f"Rebuilding index from {len(files)} documents. This may take some time.",
            "files": files
        },
        status_code=202  # Accepted
    )

@router.get("/list-documents")
async def list_documents():
    """List all documents in the medical_books folder."""
    if not os.path.exists(MEDICAL_BOOKS_DIR):
        return {"documents": []}
    
    files = [f for f in os.listdir(MEDICAL_BOOKS_DIR) 
             if os.path.isfile(os.path.join(MEDICAL_BOOKS_DIR, f)) and 
             (f.lower().endswith('.pdf') or f.lower().endswith('.txt'))]
    
    return {
        "documents": [
            {
                "filename": f,
                "path": os.path.join(MEDICAL_BOOKS_DIR, f),
                "size_bytes": os.path.getsize(os.path.join(MEDICAL_BOOKS_DIR, f)),
                "processing_status": processing_status.get(f, "unknown")
            } for f in files
        ],
        "count": len(files)
    }

@router.get("/index-status")
async def index_status(retriever: MedicalDocumentRetriever = Depends(get_retriever)):
    """Check if the index exists."""
    return {
        "index_exists": retriever.vector_store is not None,
        "index_path": retriever.index_path
    }

