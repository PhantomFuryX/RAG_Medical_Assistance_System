# src/integration/whatsapp_bot.py
# from fastapi import FastAPI, Request, Form
# from fastapi.responses import PlainTextResponse
# import os
# import httpx
# from twilio.twiml.messaging_response import MessagingResponse
# from src.utils.settings import settings
# from pydantic import BaseModel

# class WhatsAppRequest(BaseModel):
#     Body: str

# async def whatsapp_webhook(request: WhatsAppRequest):
#     incoming_msg = request.Body.strip()
#     response = MessagingResponse()
    
#     if incoming_msg:
#         # Use the chat API endpoint
#         try:
#             # Determine the API URL based on environment
#             base_url = settings.BACKEND_URL_ONLINE if settings.ONLINE_MODE == "online" else "http://localhost:8000"
#             chat_api_url = f"{base_url}/chat"
            
#             # Send request to chat API
#             async with httpx.AsyncClient() as client:
#                 api_response = await client.post(
#                     chat_api_url,
#                     json={
#                         "message": incoming_msg,
#                         "user_id": "whatsapp_user",  # You might want to track users by phone number
#                         "session_id": "whatsapp_session"  # You might want to track sessions
#                     }
#                 )
                
#                 if api_response.status_code == 200:
#                     result = api_response.json()
#                     answer = result.get("response", "Sorry, I couldn't process your request.")
#                     response.message(answer)
#                 else:
#                     response.message("Sorry, I encountered an error processing your request.")
#         except Exception as e:
#             response.message(f"Sorry, an error occurred: {str(e)}")
#     else:
#         response.message("Sorry, I did not receive any message.")
    
#     return PlainTextResponse(str(response))
