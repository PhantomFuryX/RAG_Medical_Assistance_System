import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")