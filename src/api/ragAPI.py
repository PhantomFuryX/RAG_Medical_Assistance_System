from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from src.pipeline.diagonosis_pipeline import DiagnosisPipeline
from src.retrieval.document_retriever import MedicalDocumentRetriever
from src.nlp.diagnosis_chain import build_diagnosis_chain
from src.utils.config import DEEPSEEK_API_KEY
from src.main.pydentic_models.models import RAGRequest, RAGResponse


router = APIRouter(prefix="/rag", tags=["RAG"])

# Initialize components
try:
    retriever = MedicalDocumentRetriever(index_path="faiss_medical_index")
    index_loaded = retriever.index is not None
except Exception as e:
    print(f"Error initializing retriever: {e}")
    retriever = None
    index_loaded = False
diagnosis_chain = build_diagnosis_chain('deepseek')
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
        retrieved_docs = retriever.retrieve(request.symptoms)
        references = [doc.page_content[:200] + "..." for doc in retrieved_docs]
        
        return {
            "diagnosis": diagnosis,
            "references": references
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")