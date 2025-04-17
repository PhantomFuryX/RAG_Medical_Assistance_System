from fastapi import APIRouter, HTTPException
from src.main.pydentic_models.models import FeedbackRequest
from src.nlp.sentiment_analyzer import analyze_sentiment
import json
import os
import time
from datetime import datetime

router = APIRouter(prefix="/feedback", tags=["FEEDBACK"])

FEEDBACK_DIR = "src/data/feedback"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

@router.post("/submit")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback for a response."""
    try:
        # Analyze sentiment of the feedback
        sentiment_analysis = analyze_sentiment(request.feedback_text)
        
        # Create feedback record
        feedback_record = {
            "user_id": request.user_id,
            "query": request.query,
            "response": request.response,
            "rating": request.rating,
            "feedback_text": request.feedback_text,
            "sentiment": sentiment_analysis,
            "timestamp": str(datetime.now())
        }
        
        # Save feedback to file
        feedback_file = os.path.join(FEEDBACK_DIR, f"feedback_{int(time.time())}.json")
        with open(feedback_file, "w") as f:
            json.dump(feedback_record, f, indent=2)
        
        return {"message": "Feedback submitted successfully", "sentiment": sentiment_analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

