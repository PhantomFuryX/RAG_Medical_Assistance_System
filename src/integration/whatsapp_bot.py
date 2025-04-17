# src/integration/whatsapp_bot.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import PlainTextResponse
import os
from twilio.twiml.messaging_response import MessagingResponse
from src.nlp.openai_integration import get_openai_response
from dotenv import load_dotenv
from pydantic import BaseModel

class WhatsAppRequest(BaseModel):
    Body: str
    
    
load_dotenv()


def whatsapp_webhook(request: WhatsAppRequest):
    incoming_msg = request.Body.strip()
    response = MessagingResponse()
    # For simplicity, we directly use the OpenAI API to respond
    if incoming_msg:
        answer = get_openai_response(f"User said: {incoming_msg}. Provide a concise, medically-informed response.")
        response.message(answer)
    else:
        response.message("Sorry, I did not receive any message.")
    return PlainTextResponse(str(response))
