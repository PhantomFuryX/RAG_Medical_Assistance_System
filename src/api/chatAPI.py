from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime
from src.main.pydentic_models.models import ChatRequest, ChatResponse
from src.services.conversation_service import ConversationService
from src.main.core.llm_engine import generate_response, generate_response_with_rag
from src.utils.db_manager import db_manager
from src.utils.encryption import UserIdEncryption
from src.utils.logger import get_api_logger

logger = get_api_logger()

router = APIRouter()

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
            chat_model='openai'  # You can make this configurable
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

@router.get("/get_user_history/{user_id}", response_model=list[ChatResponse])
async def get_user_history(user_id: str, limit: int = 10):
    """
    Get chat history for a specific user
    """
    try:
        history = await db_manager.get_chat_history(user_id, limit)
        return {"history": history}
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