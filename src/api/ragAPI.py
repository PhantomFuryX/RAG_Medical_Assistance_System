from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from src.pipeline.diagonosis_pipeline import DiagnosisPipeline
from src.retrieval.document_retriever import MedicalDocumentRetriever
from src.nlp.diagnosis_chain import build_diagnosis_chain
from src.utils.settings import Settings
from src.main.pydentic_models.models import RAGRequest, RAGResponse
from src.utils.db_manager import db_manager
from src.utils.registry import registry
from src.utils.logger import get_api_logger
from src.utils.settings import settings

logger = get_api_logger()


router = APIRouter(prefix="/rag", tags=["RAG"])

# Initialize components
try:
    if registry.get("retriever") is None:
        retriever = MedicalDocumentRetriever(lazy_loading=True)
    else:
        # Use the existing retriever from the registry
        logger.info("Using existing retriever from registry")
        retriever = registry.get("retriever")
    index_loaded = retriever.vector_store is not None
except Exception as e:
    print(f"Error initializing retriever: {e}")
    retriever = None
    index_loaded = False
diagnosis_chain = build_diagnosis_chain(settings.MODEL_API)
pipeline = DiagnosisPipeline()


def get_retriever():
    return retriever
@router.get("/health")
def health_check():
    return {"status": "ok", "index_loaded": index_loaded}
        
@router.post("/rag-diagnosis", response_model=RAGResponse)
async def get_rag_diagnosis(request: RAGRequest):
    try:
        # Process the input using the RAG pipeline
        diagnosis = pipeline.process_input(request.symptoms, request.image_path)
        
        # Get the retrieved documents for references
        if registry.get("retriever") is None:
            retriever = MedicalDocumentRetriever(lazy_loading=True)
        else:
            # Use the existing retriever from the registry
            logger.info("Using existing retriever from registry")
            retriever = registry.get("retriever")
        index_loaded = retriever.vector_store is not None
        retrieved_docs = retriever.retrieve(request.symptoms)
        references = [doc.page_content[:200] + "..." for doc in retrieved_docs]
        
        return {
            "diagnosis": diagnosis,
            "references": references
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")