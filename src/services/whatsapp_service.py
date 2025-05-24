from twilio.rest import Client
from src.utils.settings import settings
from src.utils.logger import get_api_logger

logger = get_api_logger()

class WhatsAppService:
    def __init__(self):
        # Initialize Twilio client with your account SID and auth token
        # These should be added to your settings.py file
        self.client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        self.from_number = settings.TWILIO_WHATSAPP_NUMBER  # Your Twilio WhatsApp number
    
    def extract_message_data(self, data):
        """
        Extract phone number and message from WhatsApp payload.
        
        Args:
            data (dict): The webhook payload from WhatsApp/Twilio
            
        Returns:
            tuple: (phone_number, message)
        """
        try:
            # This is a simplified example - actual structure depends on Twilio's webhook format
            if 'From' in data and 'Body' in data:
                phone_number = data['From'].replace('whatsapp:', '')
                message = data['Body']
                return phone_number, message
                
            # For Facebook WhatsApp Business API format
            elif 'entry' in data and len(data['entry']) > 0:
                entry = data['entry'][0]
                if 'changes' in entry and len(entry['changes']) > 0:
                    change = entry['changes'][0]
                    if 'value' in change and 'messages' in change['value'] and len(change['value']['messages']) > 0:
                        message_data = change['value']['messages'][0]
                        phone_number = message_data['from']
                        message = message_data['text']['body']
                        return phone_number, message
            
            return None, None
        except Exception as e:
            print(f"Error extracting message data: {str(e)}")
            return None, None
    
    def send_message(self, to_number, message):
        """
        Send a WhatsApp message using Twilio.
        
        Args:
            to_number (str): The recipient's phone number
            message (str): The message to send
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Add 'whatsapp:' prefix if not present
            if not to_number.startswith('whatsapp:'):
                to_number = f'whatsapp:{to_number}'
                
            # Send the message
            self.client.messages.create(
                from_=f'whatsapp:{self.from_number}',
                body=message,
                to=to_number
            )
            logger.info(f"Sent WhatsApp message to {to_number}: {message}")
            return True
        except Exception as e:
            print(f"Error sending WhatsApp message: {str(e)}")
            return False
