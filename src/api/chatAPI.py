#Create a Fastapi app with pydentic and openapi documentation
#User can feed image or text data to the api and it will return the response 
from fastapi import APIRouter, HTTPException
from src.main.pydentic_models.models import ChatRequest, ChatResponse
# from src.main.core.llm_engine import generate_response  # you will define this
from src.nlp.chatting_chain import get_llm_response
from src.utils.db import save_chat_exchange

router = APIRouter(prefix="/chatting", tags=["Chat"])

@router.post("/chat", response_model=ChatResponse, status_code=200, response_description="Chat response")
def chat(request: ChatRequest):
    ai_response = get_llm_response(request.user_question, request.summary)
    save_chat_exchange(request.user_id, request.user_question, ai_response)

    return ChatResponse(
        user_id=request.user_id,
        user_question=request.user_question,
        ai_response=ai_response
    )
    
@router.get("/health")
def health_check():
    return {"status": "ok"}
