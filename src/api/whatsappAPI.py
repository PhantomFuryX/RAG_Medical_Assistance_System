from fastapi import APIRouter, Request, Depends, HTTPException, BackgroundTasks
from src.services.whatsapp_service import WhatsAppService
from src.utils.encryption import UserIdEncryption
from src.services.conversation_service import ConversationService
from src.main.pydentic_models.models import ChatRequest, ChatResponse
import httpx
from src.utils.settings import settings
from src.main.pydentic_models.models import WhatsAppRequest

router = APIRouter(prefix="/Whatsapp", tags=["WHATSAPP"])
whatsapp_service = WhatsAppService()
encryption = UserIdEncryption()
conversation_service = ConversationService()



@router.post("/message")
async def send_whatsapp_message(request: Request):
    """
    Endpoint to handle incoming WhatsApp messages from Twilio.
    Uses the /chat endpoint from chatAPI.py to process messages.
    """
    try:
        # Parse form data from Twilio
        form_data = await request.form()
        incoming_msg = form_data.get("Body", "").strip()
        sender = form_data.get("From", "").strip()
        
        if not incoming_msg or not sender:
            return {"status": "ignored"}
        
        # Clean the phone number (remove "whatsapp:" prefix if present)
        phone_number = sender.replace("whatsapp:", "")
        
        # Encrypt phone number to use as user_id
        user_id = encryption.encrypt_phone(phone_number)
        
        # Determine the API URL based on environment
        base_url = settings.BACKEND_URL_ONLINE if settings.ONLINE_MODE == "online" else "http://localhost:8000"
        chat_api_url = f"{base_url}/Chat/chat"
        
        # Call the /chat endpoint from chatAPI.py
        async with httpx.AsyncClient() as client:
            chat_response = await client.post(
                chat_api_url,
                json={
                    "user_id": user_id,
                    "user_question": incoming_msg
                }
            )
            
            if chat_response.status_code == 200:
                response_data = chat_response.json()
                response_text = response_data.get("response", f"{response_data}\tSorry, I couldn't process your request.")
            else:
                response_text = f"Sorry, I encountered an error processing your request. Status code: {chat_response.status_code}"
        
        # Send response back to WhatsApp
        whatsapp_service.send_message(phone_number, response_text)
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing WhatsApp message: {str(e)}")
