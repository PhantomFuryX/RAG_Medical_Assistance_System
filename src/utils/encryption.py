import hashlib
import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class UserIdEncryption:
    def __init__(self, secret_key=None):
        # Use environment variable or provided key
        self.secret_key = secret_key or os.environ.get('ENCRYPTION_KEY', 'default_secret_key')
        self._setup_encryption()
        
    def _setup_encryption(self):
        # Generate a key from the secret
        salt = b'medical_assistant_salt'  # Should be stored securely in production
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.secret_key.encode()))
        self.cipher = Fernet(key)
    
    def encrypt_phone(self, phone_number):
        """Encrypt a phone number to use as user ID"""
        if not phone_number:
            return None
        return self.cipher.encrypt(phone_number.encode()).decode()
    
    def decrypt_phone(self, encrypted_id):
        """Decrypt an encrypted user ID back to phone number"""
        if not encrypted_id:
            return None
        try:
            return self.cipher.decrypt(encrypted_id.encode()).decode()
        except Exception:
            return None
    
    def hash_phone(self, phone_number):
        """Create a one-way hash of phone number (if you don't need to decrypt)"""
        if not phone_number:
            return None
        return hashlib.sha256(phone_number.encode()).hexdigest()