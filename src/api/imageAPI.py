from fastapi import APIRouter, UploadFile, File, HTTPException
from src.models.image_classifier import MedicalImageClassifier as ImageClassifier
import tempfile
import os

router = APIRouter(prefix="/images", tags=["IMAGES"])

# Initialize the image classifier
image_classifier = ImageClassifier()

@router.post("/analyze")
async def analyze_medical_image(file: UploadFile = File(...)):
    """Analyze a medical image and provide classification."""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only PNG and JPG images are supported")
    
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_file_path = temp_file.name
    
    try:
        # Classify the image
        classification = image_classifier.classify_image(temp_file_path)
        
        return {
            "filename": file.filename,
            "classification": classification
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
