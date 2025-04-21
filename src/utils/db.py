import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from bson import ObjectId
from src.utils.mongodb import get_chat_collection, get_feedback_collection

# Define the directory for storing chat history as fallback
CHAT_HISTORY_DIR = "src/data/chat_history"
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# Configure logging
logger = logging.getLogger(__name__)

def save_chat_exchange(user_id: str, user_question: str, assistant_response: str, 
                       source_documents: Optional[List[str]] = None, 
                       rag_used: bool = False) -> Dict[str, Any]:
    """
    Save a chat exchange to the database or file system.
    
    Args:
        user_id: Unique identifier for the user
        user_question: The question asked by the user
        assistant_response: The response provided by the assistant
        source_documents: List of source documents used for RAG (if any)
        rag_used: Whether RAG was used for this response
        
    Returns:
        The saved chat exchange record
    """
    # Create a chat exchange record
    timestamp = datetime.now()
    chat_exchange = {
        "user_id": user_id,
        "timestamp": timestamp,
        "user_question": user_question,
        "assistant_response": assistant_response,
        "source_documents": source_documents if source_documents else [],
        "rag_used": rag_used
    }
    
    # Try to save to MongoDB first
    chat_collection = get_chat_collection()
    if chat_collection is not None:
        try:
            # Insert the chat exchange into the database
            result = chat_collection.insert_one(chat_exchange)
            
            # Add the MongoDB ID to the record
            chat_exchange["_id"] = str(result.inserted_id)
            
            logger.info(f"Chat exchange saved to MongoDB with ID: {result.inserted_id}")
            return chat_exchange
        except Exception as e:
            logger.error(f"Error saving to MongoDB: {e}")
            logger.warning("Falling back to file storage")
            # Fall back to file storage
    
    # Fallback: Save to file system
    user_dir = os.path.join(CHAT_HISTORY_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    # Generate a filename based on timestamp and a unique ID
    file_id = f"{int(time.time())}_{hash(user_question) % 10000}"
    chat_exchange["_id"] = file_id
    filename = f"{file_id}.json"
    file_path = os.path.join(user_dir, filename)
    
    # Save the chat exchange to a JSON file
    with open(file_path, "w", encoding="utf-8") as f:
        # Convert datetime to string for JSON serialization
        chat_exchange_json = chat_exchange.copy()
        chat_exchange_json["timestamp"] = timestamp.isoformat()
        json.dump(chat_exchange_json, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Chat exchange saved to file: {file_path}")
    return chat_exchange

def get_chat_history(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieve chat history for a specific user.
    
    Args:
        user_id: Unique identifier for the user
        limit: Maximum number of chat exchanges to retrieve
        
    Returns:
        List of chat exchanges, ordered by timestamp (newest first)
    """
    # Try to get from MongoDB first
    chat_collection = get_chat_collection()
    if chat_collection is not None:
        try:
            # Query the database for chat exchanges for this user
            cursor = chat_collection.find(
                {"user_id": user_id}
            ).sort("timestamp", -1).limit(limit)
            
            # Convert ObjectId to string for JSON serialization
            chat_history = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                if isinstance(doc["timestamp"], datetime):
                    doc["timestamp"] = doc["timestamp"].isoformat()
                chat_history.append(doc)
            
            if chat_history:  # If we found records in MongoDB
                logger.info(f"Retrieved {len(chat_history)} chat exchanges from MongoDB for user {user_id}")
                return chat_history
        except Exception as e:
            logger.error(f"Error retrieving from MongoDB: {e}")
            logger.warning("Falling back to file storage")
            # Fall back to file storage
    
    # Fallback: Get from file system
    user_dir = os.path.join(CHAT_HISTORY_DIR, user_id)
    
    # If user directory doesn't exist, return empty history
    if not os.path.exists(user_dir):
        logger.info(f"No chat history found for user {user_id}")
        return []
    
    # Get all JSON files in the user directory
    files = [os.path.join(user_dir, f) for f in os.listdir(user_dir) 
             if f.endswith('.json') and os.path.isfile(os.path.join(user_dir, f))]
    
    # Sort files by modification time (newest first)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Limit the number of files
    files = files[:limit]
    
    # Load chat exchanges from files
    chat_history = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chat_exchange = json.load(f)
                chat_history.append(chat_exchange)
        except Exception as e:
            logger.error(f"Error loading chat exchange from {file_path}: {e}")
    
    logger.info(f"Retrieved {len(chat_history)} chat exchanges from file system for user {user_id}")
    return chat_history

def delete_chat_history(user_id: str) -> bool:
    """
    Delete all chat history for a specific user.
    
    Args:
        user_id: Unique identifier for the user
        
    Returns:
        True if successful, False otherwise
    """
    success = False
    
    # Try to delete from MongoDB first
    chat_collection = get_chat_collection()
    if chat_collection is not None:
        try:
            result = chat_collection.delete_many({"user_id": user_id})
            logger.info(f"Deleted {result.deleted_count} chat exchanges from MongoDB for user {user_id}")
            success = True
        except Exception as e:
            logger.error(f"Error deleting from MongoDB: {e}")
            # Continue to try file system deletion
    
    # Also delete from file system (as a backup or if MongoDB failed)
    user_dir = os.path.join(CHAT_HISTORY_DIR, user_id)
    
    # If user directory doesn't exist, return current success state
    if not os.path.exists(user_dir):
        return success
    
    try:
        # Delete all files in the user directory
        for file in os.listdir(user_dir):
            file_path = os.path.join(user_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        # Remove the directory
        os.rmdir(user_dir)
        logger.info(f"Deleted chat history files for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting chat history files for user {user_id}: {e}")
        return success

def get_chat_by_id(chat_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific chat exchange by its ID.
    
    Args:
        chat_id: The ID of the chat exchange
        
    Returns:
        The chat exchange or None if not found
    """
    # Try to get from MongoDB first
    chat_collection = get_chat_collection()
    if chat_collection is not None:
        try:
            # Try to convert to ObjectId for MongoDB query
            try:
                object_id = ObjectId(chat_id)
                chat = chat_collection.find_one({"_id": object_id})
                if chat:
                    chat["_id"] = str(chat["_id"])
                    if isinstance(chat["timestamp"], datetime):
                        chat["timestamp"] = chat["timestamp"].isoformat()
                    logger.info(f"Retrieved chat {chat_id} from MongoDB")
                    return chat
            except Exception:
                # If chat_id is not a valid ObjectId, it might be a file ID
                pass
        except Exception as e:
            logger.error(f"Error retrieving chat from MongoDB: {e}")
            # Fall back to file search
    
    # Fallback: Search in file system
    # This is less efficient as we need to search all user directories
    for user_dir in os.listdir(CHAT_HISTORY_DIR):
        user_path = os.path.join(CHAT_HISTORY_DIR, user_dir)
        if os.path.isdir(user_path):
            # Look for a file that contains the chat_id
            for file in os.listdir(user_path):
                if chat_id in file and file.endswith('.json'):
                    file_path = os.path.join(user_path, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            chat = json.load(f)
                            logger.info(f"Retrieved chat {chat_id} from file system")
                            return chat
                    except Exception as e:
                        logger.error(f"Error loading chat from {file_path}: {e}")
    
    logger.warning(f"Chat {chat_id} not found in MongoDB or file system")
    return None

def update_chat_feedback(chat_id: str, feedback: Dict[str, Any]) -> bool:
    """
    Update a chat exchange with user feedback.
    
    Args:
        chat_id: The ID of the chat exchange
        feedback: The feedback data to add
        
    Returns:
        True if successful, False otherwise
    """
    # Try to update in MongoDB first
    chat_collection = get_chat_collection()
    if chat_collection is not None:
        try:
            # Try to convert to ObjectId for MongoDB query
            try:
                object_id = ObjectId(chat_id)
                result = chat_collection.update_one(
                    {"_id": object_id},
                    {"$set": {"feedback": feedback}}
                )
                if result.modified_count > 0:
                    logger.info(f"Updated feedback for chat {chat_id} in MongoDB")
                    return True
            except Exception:
                # If chat_id is not a valid ObjectId, it might be a file ID
                pass
        except Exception as e:
            logger.error(f"Error updating feedback in MongoDB: {e}")
            # Fall back to file update
    
    # Fallback: Update in file system
    # This is less efficient as we need to search all user directories
    for user_dir in os.listdir(CHAT_HISTORY_DIR):
        user_path = os.path.join(CHAT_HISTORY_DIR, user_dir)
        if os.path.isdir(user_path):
            # Look for a file that contains the chat_id
            for file in os.listdir(user_path):
                if chat_id in file and file.endswith('.json'):
                    file_path = os.path.join(user_path, file)
                    try:
                        # Read the existing chat data
                        with open(file_path, "r", encoding="utf-8") as f:
                            chat = json.load(f)
                        
                        # Add the feedback
                        chat["feedback"] = feedback
                        
                        # Write the updated chat data back to the file
                        with open(file_path, "w", encoding="utf-8") as f:
                            json.dump(chat, f, ensure_ascii=False, indent=2)
                        
                        logger.info(f"Updated feedback for chat {chat_id} in file system")
                        return True
                    except Exception as e:
                        logger.error(f"Error updating chat in {file_path}: {e}")
    
    logger.warning(f"Chat {chat_id} not found for feedback update")
    return False

def save_feedback(feedback_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save user feedback separately from chat history.
    
    Args:
        feedback_data: The feedback data to save
        
    Returns:
        The saved feedback record
    """
    # Add timestamp if not present
    if "timestamp" not in feedback_data:
        feedback_data["timestamp"] = datetime.now()
    
    # Try to save to MongoDB first
    feedback_collection = get_feedback_collection()
    if feedback_collection is not None:
        try:
            # Insert the feedback into the database
            result = feedback_collection.insert_one(feedback_data)
            
            # Add the MongoDB ID to the record
            feedback_data["_id"] = str(result.inserted_id)
            
            logger.info(f"Feedback saved to MongoDB with ID: {result.inserted_id}")
            return feedback_data
        except Exception as e:
            logger.error(f"Error saving feedback to MongoDB: {e}")
            # Fall back to file storage
    
    # Fallback: Save to file system
    feedback_dir = os.path.join("src/data", "feedback")
    os.makedirs(feedback_dir, exist_ok=True)
    
    # Generate a filename based on timestamp
    timestamp = feedback_data["timestamp"]
    if isinstance(timestamp, datetime):
        timestamp_str = timestamp.isoformat()
        feedback_data["timestamp"] = timestamp_str
    else:
        timestamp_str = str(timestamp)
    
    file_id = f"{int(time.time())}_{hash(str(feedback_data)) % 10000}"
    feedback_data["_id"] = file_id
    filename = f"feedback_{file_id}.json"
    file_path = os.path.join(feedback_dir, filename)
    
    # Save the feedback to a JSON file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(feedback_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Feedback saved to file: {file_path}")
    return feedback_data

def get_all_feedback(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Retrieve all feedback.
    
    Args:
        limit: Maximum number of feedback records to retrieve
        
    Returns:
        List of feedback records, ordered by timestamp (newest first)
    """
    # Try to get from MongoDB first
    feedback_collection = get_feedback_collection()
    if feedback_collection is not None:
        try:
            # Query the database for feedback
            cursor = feedback_collection.find().sort("timestamp", -1).limit(limit)
            
            # Convert ObjectId to string for JSON serialization
            feedback_list = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                if isinstance(doc["timestamp"], datetime):
                    doc["timestamp"] = doc["timestamp"].isoformat()
                feedback_list.append(doc)
            
            if feedback_list:  # If we found records in MongoDB
                logger.info(f"Retrieved {len(feedback_list)} feedback records from MongoDB")
                return feedback_list
        except Exception as e:
            logger.error(f"Error retrieving feedback from MongoDB: {e}")
            # Fall back to file storage
    
    # Fallback: Get from file system
    feedback_dir = os.path.join("src/data", "feedback")
    
    # If feedback directory doesn't exist, return empty list
    if not os.path.exists(feedback_dir):
        return []
    
    # Get all JSON files in the feedback directory
    files = [os.path.join(feedback_dir, f) for f in os.listdir(feedback_dir) 
             if f.startswith('feedback_') and f.endswith('.json') and os.path.isfile(os.path.join(feedback_dir, f))]
    
    # Sort files by modification time (newest first)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Limit the number of files
    files = files[:limit]
    
    # Load feedback from files
    feedback_list = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                feedback = json.load(f)
                feedback_list.append(feedback)
        except Exception as e:
            logger.error(f"Error loading feedback from {file_path}: {e}")
    
    logger.info(f"Retrieved {len(feedback_list)} feedback records from file system")
    return feedback_list

# Add a function to create a summary of chat history for context
def get_chat_history_summary(user_id: str, max_chats: int = 5) -> str:
    """
    Create a summary of recent chat history for a user.
    
    Args:
        user_id: Unique identifier for the user
        max_chats: Maximum number of recent chats to include
        
    Returns:
        A string summary of recent conversations
    """
    chat_history = get_chat_history(user_id, limit=max_chats)
    
    if not chat_history:
        return "No previous conversation history."
    
    summary = "Recent conversation history:\n\n"
    
    for i, chat in enumerate(chat_history):
        summary += f"User: {chat['user_question']}\n"
        summary += f"Assistant: {chat['assistant_response']}\n\n"
    
    return summary.strip()

def get_summary_collection():
    """Get the MongoDB collection for chat summaries"""
    from src.utils.mongodb import get_database
    db = get_database()
    return db["chat_summaries"]

def save_chat_summary(user_id: str, summary: str) -> Dict[str, Any]:
    """
    Save a chat summary to the database.
    
    Args:
        user_id: Unique identifier for the user
        summary: The generated summary of conversations
        
    Returns:
        The saved summary document
    """
    try:
        # Get the summary collection
        collection = get_summary_collection()
        
        # Create the summary document
        summary_doc = {
            "user_id": user_id,
            "content": summary,
            "timestamp": datetime.now()
        }
        
        # Insert the summary
        result = collection.insert_one(summary_doc)
        
        # Add the _id to the document
        summary_doc["_id"] = result.inserted_id
        
        return summary_doc
    except Exception as e:
        logger.error(f"Error saving chat summary: {str(e)}")
        # Save to file system as fallback
        fallback_file = os.path.join(CHAT_HISTORY_DIR, f"{user_id}_summary_{int(time.time())}.json")
        with open(fallback_file, 'w') as f:
            json.dump({"user_id": user_id, "content": summary, "timestamp": str(datetime.now())}, f)
        
        return {"user_id": user_id, "content": summary, "timestamp": datetime.now()}

def get_user_summaries(user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get the most recent summaries for a user.
    
    Args:
        user_id: Unique identifier for the user
        limit: Maximum number of summaries to retrieve
        
    Returns:
        List of summary documents
    """
    try:
        # Get the summary collection
        collection = get_summary_collection()
        
        # Query for summaries
        cursor = collection.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
        
        # Convert cursor to list
        summaries = list(cursor)
        
        return summaries
    except Exception as e:
        logger.error(f"Error retrieving user summaries: {str(e)}")
        return []
    
def get_summary_collection():
    """Get the MongoDB collection for chat summaries"""
    from src.utils.mongodb import get_database
    db = get_database()
    return db["chat_summaries"]