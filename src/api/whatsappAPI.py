from fastapi import APIRouter, Request, Depends, HTTPException, BackgroundTasks
from src.services.whatsapp_service import WhatsAppService
from src.utils.encryption import UserIdEncryption
from src.main.core.llm_engine import generate_response
from src.services.conversation_service import ConversationService

router = APIRouter()
whatsapp_service = WhatsAppService()
encryption = UserIdEncryption()
conversation_service = ConversationService()

@router.post("/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Webhook for WhatsApp messages.
    Processes incoming messages and sends responses.
    """
    try:
        # Parse the incoming webhook data
        data = await request.json()
        
        # Extract phone number and message from WhatsApp payload
        phone_number, message = whatsapp_service.extract_message_data(data)
        
        if not phone_number or not message:
            return {"status": "ignored"}
        
        # Encrypt phone number to use as user_id
        user_id = encryption.encrypt_phone(phone_number)
        
        # Generate response
        response = generate_response(message)
        
        # Store conversation in background
        background_tasks.add_task(
            conversation_service.store_conversation,
            user_id,
            message,
            response
        )
        
        # Send response back to WhatsApp
        whatsapp_service.send_message(phone_number, response)
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")
