from pydantic import EmailStr, Field
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM API Keys
    OPENAI_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    HUGGINGFACEHUB_ACCESS_TOKEN: Optional[str] = None

    # Database
    MONGODB_URI: str = Field(..., description="MongoDB URI", )
    DB_NAME: str = "ragmas"
    USER_TABLE: str = "users"
    CHAT_TABLE: str = "chats"
    CHAT_HISTORY_TABLE: str = "chat_history"
    CHAT_SUMMARY_TABLE: str = "chat_summaries"
    EMBEDDING_TABLE: str = "embeddings"

    # Vector Store
    VECTOR_STORE: str = Field("faiss", description="Options: 'faiss', 'chromadb'")
    CHROMADB_DIR: str = "src/data/chromadb"
    FAISS_INDEX_PATH: str = "src/data/faiss_index"
    EMBEDDINGS_PATH: str = "src/data/embeddings"

    # Model Selection
    MODEL_API: str = Field("openai", description="Options: 'openai', 'anthropic', 'deepseek', 'huggingface'")

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    EMBEDDING_MODEL: str = Field("text-embedding-ada-002", description="Options: 'text-embedding-ada-002', 'all-MiniLM-L6-v2'")
    # JWT/Auth (uncomment if needed)
    # JWT_PUBLIC_KEY: Optional[str] = None
    # JWT_PRIVATE_KEY: Optional[str] = None
    # REFRESH_TOKEN_EXPIRES_IN: Optional[int] = 7
    # ACCESS_TOKEN_EXPIRES_IN: Optional[int] = 1
    # JWT_ALGORITHM: Optional[str] = "RS256"

    # Email (uncomment if needed)
    # EMAIL_HOST: Optional[str] = None
    # EMAIL_PORT: Optional[int] = None
    # EMAIL_USERNAME: Optional[str] = None
    # EMAIL_PASSWORD: Optional[str] = None
    # EMAIL_FROM: Optional[EmailStr] = None

    # CORS/Frontend/Backend
    CLIENT_ORIGIN: Optional[str] = None
    CLIENT_ORIGIN_ONLINE: Optional[str] = None
    BACKEND_URL_ONLINE: Optional[str] = None
    
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_WHATSAPP_NUMBER: Optional[str] = None 

    # Environment & Mode
    ENVIRONMENT: str = Field("development", description="Options: 'development', 'production'")
    ONLINE_MODE: str | None = Field("offline", description="Options: 'online', 'offline'")

class Config:
        env_file = './.env'
        env_file_encoding = 'utf-8'

settings = Settings()
