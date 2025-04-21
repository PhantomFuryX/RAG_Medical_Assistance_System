import os
from pymongo import MongoClient
from typing import Optional
from dotenv import load_dotenv
from src.utils.logger import get_logger

# Load environment variables from .env file if it exists
load_dotenv()

# Get MongoDB Atlas connection string from environment variable
# Format: mongodb+srv://<username>:<password>@<cluster>.mongodb.net/<dbname>?retryWrites=true&w=majority
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "medical_assistant")

# Configure logging
logger = get_logger("mongodb")

# MongoDB client instance
_client: Optional[MongoClient] = None

def get_mongodb_client() -> Optional[MongoClient]:
    """
    Get a MongoDB Atlas client instance.
    Returns a singleton client to avoid creating multiple connections.
    """
    global _client
    if _client is None:
        try:
            logger.info(f"Connecting to MongoDB Atlas cluster using URI: {MONGODB_URI[:20]}...")
            _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            # Verify connection works by executing a simple command
            _client.admin.command('ping')
            logger.info("Successfully connected to MongoDB Atlas")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB Atlas: {e}")
            logger.warning("Using fallback storage method")
            _client = None
    return _client

def get_database():
    """Get the MongoDB database instance."""
    client = get_mongodb_client()
    if client is not None:
        return client[DB_NAME]
    return None

def get_chat_collection():
    """Get the chat history collection."""
    db = get_database()
    if db is not None:
        return db["chat_history"]
    return None

def get_feedback_collection():
    """Get the feedback collection."""
    db = get_database()
    if db is not None:
        return db["feedback"]
    return None

def get_user_collection():
    """Get the user collection."""
    db = get_database()
    if db is not None:
        return db["users"]
    return None

def close_mongodb_connection():
    """Close the MongoDB connection."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("MongoDB Atlas connection closed")

def test_connection():
    """Test the MongoDB Atlas connection and return status."""
    try:
        client = get_mongodb_client()
        if client is not None:
            # Get server info to verify connection
            server_info = client.server_info()
            version = server_info.get('version', 'unknown')
            logger.info(f"Connected to MongoDB Atlas version: {version}")
            
            # Get database and list collections
            db = client[DB_NAME]
            collections = db.list_collection_names() if db is not None else []
            
            return {
                "status": "connected",
                "version": version,
                "database": DB_NAME,
                "collections": collections
            }
        else:
            return {
                "status": "disconnected",
                "error": "Failed to establish connection to MongoDB Atlas"
            }
    except Exception as e:
        logger.error(f"Error testing MongoDB Atlas connection: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

# Run a connection test if this file is executed directly
if __name__ == "__main__":
    result = test_connection()
    print(f"MongoDB Atlas Connection Test: {result['status']}")
    if result['status'] == 'connected':
        print(f"MongoDB Atlas Version: {result['version']}")
        print(f"Database: {result['database']}")
        print(f"Collections: {', '.join(result['collections'])}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
