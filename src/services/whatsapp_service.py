import requests
import os
import json
import logging

logger = logging.getLogger(__name__)

class WhatsAppService:
    def __init__(self):
        self.api_key = os.environ.get("WHATSAPP_API_KEY")
        self.phone_number_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
        self.base_url = "https://graph.facebook.com/v17.0"
        
    def extract_message_data(self, webhook_data):
        """Extract phone number and message from WhatsApp webhook data"""
        try:
            # Extract based on WhatsApp Business API structure
            entry = webhook_data.get('entry', [{}])[0]
            changes = entry.get('changes', [{}])[0]
            value = changes.get('value', {})
            messages = value.get('messages', [{}])[0]
            
            phone_number = messages.get('from')
            message_text = messages.get('text', {}).get('body', '')
            
            return phone_number, message_text
        except Exception as e:
            logger.error(f"Error extracting message data: {str(e)}")
            return None, None
    
    def send_message(self, to_phone: str, message: str):
        """Send a message via WhatsApp API"""
        if not self.api_key or not self.phone_number_id:
            logger.error("WhatsApp API credentials not configured")
            return False
            
        url = f"{self.base_url}/{self.phone_number_id}/messages"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_phone,
            "type": "text",
            "text": {
                "body": message
            }
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error sending WhatsApp message: {str(e)}")
            return False
