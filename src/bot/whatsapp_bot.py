from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from src.pipeline.diagnosis_pipeline import MedicalAssistant

app = Flask(__name__)
assistant = MedicalAssistant()

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    incoming_msg = request.values.get("Body", "").strip()
    response = MessagingResponse()
    
    if incoming_msg.lower() == "hi":
        response.message("Hello! Please describe your symptoms.")
    else:
        diagnosis = assistant.process_input(incoming_msg)
        response.message(f"Diagnosis: {diagnosis}")
    
    return str(response)

if __name__ == "__main__":
    app.run(port=5000)
