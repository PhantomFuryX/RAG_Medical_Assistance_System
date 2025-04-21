from fastapi import APIRouter, HTTPException
from src.main.pydentic_models.models import FeedbackRequest, FeedbackResponse
from src.nlp.sentiment_analyzer import analyze_sentiment
import json
import os
import time
from src.utils.db_manager import db_manager

router = APIRouter(prefix="/feedback", tags=["FEEDBACK"])

FEEDBACK_DIR = "src/data/feedback"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

@router.post("/submit")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for a conversation
    """
    try:
        # Create feedback data
        feedback_data = {
            "user_id": request.user_id,
            "query": request.query,
            "response": request.response,
            "rating": request.rating,
            "comments": request.comments
        }
        
        # Save feedback
        saved_feedback = await db_manager.save_feedback(feedback_data)
        
        return FeedbackResponse(
            success=True,
            message="Feedback submitted successfully",
            feedback_id=str(saved_feedback.get("_id", ""))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

