from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from src.services.conversation_service import ConversationService
from src.utils.encryption import UserIdEncryption
import datetime

router = APIRouter()
conversation_service = ConversationService()
encryption = UserIdEncryption()

class SummaryRequest(BaseModel):
    user_id: str
    max_words: int = 500

class SummaryResponse(BaseModel):
    user_id: str
    summary: str
    timestamp: Optional[str] = None

@router.post("/generate", response_model=SummaryResponse)
async def generate_conversation_summary(request: SummaryRequest):
    """
    Generate a new summary of previous conversations for a user.
    This creates a fresh summary based on recent conversations and stores it.
    """
    try:
        # Generate and store a new summary
        summary = await conversation_service.generate_and_store_summary(
            request.user_id, 
            max_words=request.max_words
        )
        
        return SummaryResponse(
            user_id=request.user_id,
            summary=summary,
            timestamp=str(datetime.datetime.now())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@router.post("/cumulative", response_model=SummaryResponse)
async def get_cumulative_summary(request: SummaryRequest):
    """
    Get a cumulative summary from the last few conversation summaries.
    This combines existing summaries without creating a new one.
    """
    try:
        # Get cumulative summary from existing summaries
        summary = await conversation_service.get_cumulative_summary(request.user_id)
        
        return SummaryResponse(
            user_id=request.user_id,
            summary=summary,
            timestamp=str(datetime.datetime.now())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving cumulative summary: {str(e)}")

@router.get("/history/{user_id}", response_model=list[SummaryResponse])
async def get_summary_history(user_id: str, limit: int = 5):
    """
    Get the history of summaries for a user.
    """
    try:
        # Get summary history
        summaries = await conversation_service.get_summary_history(user_id, limit=limit)
        
        return [
            SummaryResponse(
                user_id=user_id,
                summary=summary.get("content", ""),
                timestamp=str(summary.get("timestamp", datetime.datetime.now()))
            ) 
            for summary in summaries
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving summary history: {str(e)}")
