from pymongo import MongoClient, ASCENDING, DESCENDING
from src.utils.mongodb import get_database, get_chat_collection

def init_database():
    """Initialize the database with required collections and indexes."""
    db = get_database()
    
    # Ensure chat_history collection exists
    if "chat_history" not in db.list_collection_names():
        db.create_collection("chat_history")
    
    # Create indexes for efficient queries
    chat_collection = get_chat_collection()
    
    # Index for querying by user_id and sorting by timestamp
    chat_collection.create_index([
        ("user_id", ASCENDING),
        ("timestamp", DESCENDING)
    ])
    
    # Index for full-text search on questions and responses
    chat_collection.create_index([
        ("user_question", "text"),
        ("assistant_response", "text")
    ])
    
    print("Database initialized successfully")

if __name__ == "__main__":
    init_database()
