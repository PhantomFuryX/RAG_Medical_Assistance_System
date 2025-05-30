from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, HTTPException, Query
from datetime import datetime
from src.main.pydentic_models.models import ChatRequest, ChatResponse
from src.services.conversation_service import ConversationService
from src.main.core.llm_engine import generate_response, generate_response_with_rag
from src.utils.db_manager import db_manager
from src.utils.encryption import UserIdEncryption
from src.utils.logger import get_api_logger
from src.utils.settings import settings
from src.main.pydentic_models.models import ChatHistoryResponse, ChatMessage
from bson import ObjectId

logger = get_api_logger()

router = APIRouter(prefix="/Chat", tags=["Chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Process a chat request and return a response.
    Only requires user_id (encrypted phone) and user_question.
    """
    try:
        # Initialize services
        conversation_service = ConversationService()
        
        # Generate response with RAG
        logger.info(f"Generating response for user_id: {request.user_id}, question: {request.user_question}")
        response_data = await generate_response_with_rag(
            user_id=request.user_id,
            user_question=request.user_question,
            chat_model=settings.MODEL_API  # You can make this configurable
        )
        
        # Extract the response text
        response = response_data["response"]
        
        # Generate and store summary in background
        logger.info(f"Generating summary for user_id: {request.user_id}")
        background_tasks.add_task(
            conversation_service.generate_and_store_summary,
            request.user_id
        )
        logger.info(f"printing chat response: {response_data}")
        return ChatResponse(
            user_id=request.user_id,
            user_question=request.user_question,
            response=response,
            timestamp=datetime.now(),
            rag_used=response_data.get("rag_used", False),
            source_documents=response_data.get("source_documents", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.get("/history/{user_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
    user_id: str,
    limit: int = Query(10, ge=1, le=50),
    db_manager = Depends(db_manager)
):
    """Get chat history for a user"""
    try:
        # Get chat history from database
        history_cursor = db_manager.db.chat_history.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(limit)
        
        # Convert MongoDB documents to Pydantic models
        history = []
        async for doc in history_cursor:
            # Convert ObjectId to string
            doc["_id"] = str(doc["_id"])
            history.append(ChatMessage(**doc))
        
        # Return the response
        return ChatHistoryResponse(history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

@router.delete("/delete_user_history/{user_id}")
async def delete_user_history(user_id: str):
    """
    Delete all chat history for a specific user
    """
    try:
        success = await db_manager.delete_chat_history(user_id)
        if success:
            return {"success": True, "message": "Chat history deleted successfully"}
        else:
            return {"success": False, "message": "Failed to delete chat history"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting chat history: {str(e)}")