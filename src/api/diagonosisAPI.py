from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from src.pipeline.diagonosis_pipeline import DiagnosisPipeline
from src.retrieval.document_retriever import MedicalDocumentRetriever

router = APIRouter(prefix="/diagnosis", tags=["DIAGNOSIS"])

# Initialize the diagnosis pipeline
diagnosis_pipeline = DiagnosisPipeline()

class DiagnosisRequest(BaseModel):
    user_id: Optional[str] = None
    symptoms: str
    age: Optional[int] = None
    gender: Optional[str] = None
    medical_history: Optional[str] = None

class DiagnosisResponse(BaseModel):
    user_id: Optional[str] = None
    symptoms: str
    diagnosis: str
    possible_conditions: List[str]
    recommendations: List[str]
    confidence: float
    disclaimer: str

@router.post("/analyze", response_model=DiagnosisResponse)
async def analyze_symptoms(request: DiagnosisRequest):
    """Analyze symptoms and provide a possible diagnosis."""
    try:
        # Process the symptoms through the diagnosis pipeline
        result = diagnosis_pipeline.process(request.symptoms)
        
        # Extract possible conditions from the diagnosis
        possible_conditions = [entity["text"] for entity in result.get("entities", []) 
                              if entity.get("label") in ["DISEASE", "CONDITION"]]
        
        # Default recommendations
        recommendations = [
            "Consult with a healthcare professional for a proper diagnosis",
            "Do not self-medicate based on this information",
            "Seek immediate medical attention if symptoms worsen"
        ]
        
        # Calculate a confidence score (this is simplified)
        confidence = 0.7  # Default moderate confidence
        
        # Medical disclaimer
        disclaimer = "This information is provided for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition."
        
        return DiagnosisResponse(
            user_id=request.user_id,
            symptoms=request.symptoms,
            diagnosis=result["diagnosis"],
            possible_conditions=possible_conditions if possible_conditions else ["Unable to determine specific conditions"],
            recommendations=recommendations,
            confidence=confidence,
            disclaimer=disclaimer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing symptoms: {str(e)}")
