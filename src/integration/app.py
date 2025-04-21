# from fastapi import FastAPI,Request, Form
# import os
# import uvicorn
# from dotenv import load_dotenv
# from src.utils.db import get_db
# from src.nlp.openai_integration import get_openai_response
# from pydantic import BaseModel
# from .whatsapp_bot import whatsapp_webhook, WhatsAppRequest

# load_dotenv()

# app = FastAPI(title="Medical Assistant API")

# class OpenAIRequest(BaseModel):
#     prompt: str

# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Medical Assistant Application!"}

# @app.get("/db-test")
# def db_test():
#     db = get_db()
#     return {"message": "Connected to DB"}

# @app.post("/openai-test")
# def openai_test(request: OpenAIRequest):
#     response = get_openai_response(request.prompt)
#     return {"response": response}

# @app.post("/whatsapp-test")
# def whatsapp_test(request: WhatsAppRequest):
#     response = whatsapp_webhook(request)
#     # This is a placeholder for the WhatsApp integration test
#     return {"message": response}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
