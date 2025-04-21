from pydantic import EmailStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    DEEPSEEK_API_KEY: str
    MONGODB_URI: str
    ANTHROPIC_API_KEY: str
    # HUGGINGFACEHUB_ACCESS_TOKEN= str
    
    # ATLAS_URI: str
    # DATABASE_URL: str
    # MONGO_INITDB_DATABASE: str

    # JWT_PUBLIC_KEY: str
    # JWT_PRIVATE_KEY: str
    # REFRESH_TOKEN_EXPIRES_IN: int
    # ACCESS_TOKEN_EXPIRES_IN: int
    # JWT_ALGORITHM: str

    CLIENT_ORIGIN: str
    CLIENT_ORIGIN_ONLINE: str
    BACKEND_URL_ONLINE: str
    ONLINE_MODE: str
    ENVIRONMENT: str = "development"
    DB_NAME: str = "ragmas"

    # EMAIL_HOST: str
    # EMAIL_PORT: int
    # EMAIL_USERNAME: str
    # EMAIL_PASSWORD: str
    # EMAIL_FROM: EmailStr

    class Config:
        env_file = './.env'


settings = Settings()