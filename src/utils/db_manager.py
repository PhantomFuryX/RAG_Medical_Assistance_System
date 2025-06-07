import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from bson import ObjectId
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from src.utils.logger import get_db_logger

# Configure logging
logger = get_db_logger()

# Define the directory for storing chat history as fallback
CHAT_HISTORY_DIR = "src/data/chat_history"
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

class DatabaseManager:
    """
    Class-based database manager for MongoDB Atlas integration.
    Handles connections and operations for chat history, summaries, and feedback.
    """
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one database connection"""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the database manager"""
        if self.initialized:
            return
            
        # Get MongoDB Atlas connection string from environment variables
        self.mongo_uri = os.environ.get("MONGODB_URI")
        logger.debug(f"MONGODB_URI:\t{self.mongo_uri}") 
        if not self.mongo_uri:
            logger.warning("MONGODB_URI not set in environment variables. Using default local connection.")
            self.mongo_uri = "mongodb://localhost:27017"
        self.db_name = os.environ.get("MONGODB_DB_NAME", "medical_assistant")
        self.client = None
        self.db = None
        
        # Define collection names
        self.chat_collection_name = "chat_history"
        self.summary_collection_name = "chat_summaries"
        self.feedback_collection_name = "user_feedback"
        
        self.initialized = True
    
    async def connect(self):
        """Create a connection to MongoDB Atlas"""
        if self.client is not None:
            return self.db
            
        try:
            # Connection with additional options for Atlas
            self.client = AsyncIOMotorClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=10000,
                socketTimeoutMS=45000,
                maxPoolSize=100
            )
            
            # Test the connection
            await self.client.admin.command('ping')
            
            self.db = self.client[self.db_name]
            logger.info(f"Connected to MongoDB Atlas: {self.db_name}")
            
            # Create indexes for better query performance
            await self._create_indexes()
            
            return self.db
        except Exception as e:
            logger.error(f"MongoDB Atlas connection error: {str(e)}")
            raise
    
    async def connect_with_retry(self, max_retries=5, retry_delay=2):
        """Connect to MongoDB Atlas with retry logic"""
        retries = 0
        while retries < max_retries:
            try:
                return await self.connect()
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                retries += 1
                logger.warning(f"Connection attempt {retries} failed: {str(e)}")
                if retries >= max_retries:
                    logger.error("Max retries reached. Could not connect to MongoDB Atlas.")
                    raise
                await asyncio.sleep(retry_delay)
    
    async def _create_indexes(self):
        """Create indexes for better query performance"""
        try:
            # Create index for chat history
            await self.db[self.chat_collection_name].create_index([("user_id", 1), ("timestamp", -1)])
            
            # Create index for summaries
            await self.db[self.summary_collection_name].create_index([("user_id", 1), ("timestamp", -1)])
            
            # Create index for feedback
            await self.db[self.feedback_collection_name].create_index([("user_id", 1), ("timestamp", -1)])
            
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")
    
    # Chat History Methods
    
    async def save_chat_exchange(self, user_id: str, user_question: str, assistant_response: str, 
                               source_documents: Optional[List[str]] = None, 
                               rag_used: bool = False) -> Dict[str, Any]:
        """
        Save a chat exchange to the database.
        
        Args:
            user_id: Unique identifier for the user
            user_question: The question asked by the user
            assistant_response: The response provided by the assistant
            source_documents: List of source documents used for RAG (if any)
            rag_used: Whether RAG was used for this response
            
        Returns:
            The saved chat exchange record
        """
        if self.db is None:
            await self.connect_with_retry()
        
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
        
        try:
            # Insert the chat exchange into the database
            result = await self.db[self.chat_collection_name].insert_one(chat_exchange)
            
            # Add the MongoDB ID to the record
            chat_exchange["_id"] = str(result.inserted_id)
            
            logger.info(f"Chat exchange saved to MongoDB with ID: {result.inserted_id}")
            return chat_exchange
        except Exception as e:
            logger.error(f"Error saving to MongoDB: {e}")
            logger.warning("Falling back to file storage")
            
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
    
    async def get_chat_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve chat history for a specific user.
        
        Args:
            user_id: Unique identifier for the user
            limit: Maximum number of chat exchanges to retrieve
            
        Returns:
            List of chat exchanges, ordered by timestamp (newest first)
        """
        if self.db is None:
            await self.connect_with_retry()
            
        try:
            # Query the database for chat exchanges for this user
            cursor = self.db[self.chat_collection_name].find(
                {"user_id": user_id}
            ).sort("timestamp", -1).limit(limit)
            
            # Convert cursor to list
            chat_history = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string for JSON serialization
            for doc in chat_history:
                doc["_id"] = str(doc["_id"])
            
            if chat_history:  # If we found records in MongoDB
                logger.info(f"Retrieved {len(chat_history)} chat exchanges from MongoDB for user {user_id}")
                return chat_history
        except Exception as e:
            logger.error(f"Error retrieving from MongoDB: {e}")
            logger.warning("Falling back to file storage")
        
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
    
    # Summary Methods
    
    async def save_chat_summary(self, user_id: str, summary: str) -> Dict[str, Any]:
        """
        Save a chat summary to the database.
        
        Args:
            user_id: Unique identifier for the user
            summary: The generated summary of conversations
            
        Returns:
            The saved summary document
        """
        if self.db is None:
            await self.connect_with_retry()
            
        try:
            # Create the summary document
            summary_doc = {
                "user_id": user_id,
                "content": summary,
                "timestamp": datetime.now()
            }
            
            # Insert the summary
            result = await self.db[self.summary_collection_name].insert_one(summary_doc)
            
            # Add the _id to the document
            summary_doc["_id"] = str(result.inserted_id)
            
            logger.info(f"Summary saved to MongoDB with ID: {result.inserted_id}")
            return summary_doc
        except Exception as e:
            logger.error(f"Error saving chat summary: {str(e)}")
            
            # Save to file system as fallback
            fallback_file = os.path.join(CHAT_HISTORY_DIR, f"{user_id}_summary_{int(time.time())}.json")
            with open(fallback_file, 'w') as f:
                summary_doc = {
                    "user_id": user_id, 
                    "content": summary, 
                    "timestamp": datetime.now().isoformat()
                }
                json.dump(summary_doc, f)
            
            logger.info(f"Summary saved to file: {fallback_file}")
            return summary_doc
    
    async def get_user_summaries(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent summaries for a user.
        
        Args:
            user_id: Unique identifier for the user
            limit: Maximum number of summaries to retrieve
            
        Returns:
            List of summary documents
        """
        if self.db is None:
            await self.connect_with_retry()
            
        try:
            # Query for summaries
            cursor = self.db[self.summary_collection_name].find(
                {"user_id": user_id}
            ).sort("timestamp", -1).limit(limit)
            
            # Convert cursor to list
            summaries = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string for JSON serialization
            for doc in summaries:
                doc["_id"] = str(doc["_id"])
            
            logger.info(f"Retrieved {len(summaries)} summaries from MongoDB for user {user_id}")
            return summaries
        except Exception as e:
            logger.error(f"Error retrieving user summaries: {str(e)}")
            
            # Fallback to file system
            summaries = []
            user_dir = os.path.join(CHAT_HISTORY_DIR, user_id)
            
            if os.path.exists(user_dir):
                # Get all summary files
                files = [os.path.join(user_dir, f) for f in os.listdir(user_dir) 
                        if f.startswith(f"{user_id}_summary_") and f.endswith('.json')]
                
                # Sort by modification time (newest first)
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                
                # Limit the number of files
                files = files[:limit]
                
                # Load summaries from files
                for file_path in files:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            summary = json.load(f)
                            summaries.append(summary)
                    except Exception as e:
                        logger.error(f"Error loading summary from {file_path}: {e}")
            
            logger.info(f"Retrieved {len(summaries)} summaries from file system for user {user_id}")
            return summaries
    
    # Feedback Methods
    
    async def save_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save user feedback separately from chat history.
        
        Args:
            feedback_data: The feedback data to save
            
        Returns:
            The saved feedback record
        """
        if self.db is None:
            await self.connect_with_retry()
            
        # Add timestamp if not present
        if "timestamp" not in feedback_data:
            feedback_data["timestamp"] = datetime.now()
        
        try:
            # Insert the feedback into the database
            result = await self.db[self.feedback_collection_name].insert_one(feedback_data)
            
            # Add the MongoDB ID to the record
            feedback_data["_id"] = str(result.inserted_id)
            
            logger.info(f"Feedback saved to MongoDB with ID: {result.inserted_id}")
            return feedback_data
        except Exception as e:
            logger.error(f"Error saving feedback to MongoDB: {e}")
            
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
    
    async def get_all_feedback(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve all feedback.
        
        Args:
            limit: Maximum number of feedback records to retrieve
            
        Returns:
            List of feedback records, ordered by timestamp (newest first)
        """
        if self.db is None:
            await self.connect_with_retry()
            
        try:
            # Query the database for feedback
            cursor = self.db[self.feedback_collection_name].find().sort("timestamp", -1).limit(limit)
            
            # Convert cursor to list
            feedback_list = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string for JSON serialization
            for doc in feedback_list:
                doc["_id"] = str(doc["_id"])
            
            logger.info(f"Retrieved {len(feedback_list)} feedback records from MongoDB")
            return feedback_list
        except Exception as e:
            logger.error(f"Error retrieving feedback from MongoDB: {e}")
            
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
    
    async def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about user feedback
        
        Returns:
            Dictionary with feedback statistics
        """
        if self.db is None:
            await self.connect_with_retry()
            
        try:
            # Get average rating
            pipeline = [
                {"$match": {"rating": {"$ne": None}}},
                {"$group": {"_id": None, "avg_rating": {"$avg": "$rating"}, "count": {"$sum": 1}}}
            ]
            
            result = await self.db[self.feedback_collection_name].aggregate(pipeline).to_list(length=1)
            
            if not result:
                return {"avg_rating": 0, "count": 0}
            
            return {
                "avg_rating": result[0]["avg_rating"],
                "count": result[0]["count"]
            }
        except Exception as e:
            logger.error(f"Error getting feedback stats: {e}")
            return {"avg_rating": 0, "count": 0}
    
    # Utility Methods
    
    async def get_chat_history_summary(self, user_id: str, max_chats: int = 5) -> str:
        """
        Create a summary of recent chat history for a user.
        
        Args:
            user_id: Unique identifier for the user
            max_chats: Maximum number of recent chats to include
            
        Returns:
            A string summary of recent conversations
        """
        chat_history = await self.get_chat_history(user_id, limit=max_chats)
        
        if not chat_history:
            return "No previous conversation history."
        
        summary = "Recent conversation history:\n\n"
        
        for i, chat in enumerate(chat_history):
            summary += f"User: {chat['user_question']}\n"
            summary += f"Assistant: {chat['assistant_response']}\n\n"
        
        return summary.strip()
    
    async def delete_chat_history(self, user_id: str) -> bool:
        """
        Delete all chat history for a specific user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            True if successful, False otherwise
        """
        if self.db is None:
            await self.connect_with_retry()
            
        success = False
        
        try:
            result = await self.db[self.chat_collection_name].delete_many({"user_id": user_id})
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
    
    async def get_chat_by_id(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chat exchange by its ID.
        
        Args:
            chat_id: The ID of the chat exchange
            
        Returns:
            The chat exchange or None if not found
        """
        if self.db is None:
            await self.connect_with_retry()
            
        try:
            # Try to convert to ObjectId for MongoDB query
            try:
                object_id = ObjectId(chat_id)
                chat = await self.db[self.chat_collection_name].find_one({"_id": object_id})
                if chat:
                    chat["_id"] = str(chat["_id"])
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
    
    async def update_chat_feedback(self, chat_id: str, feedback: Dict[str, Any]) -> bool:
        """
        Update a chat exchange with user feedback.
        
        Args:
            chat_id: The ID of the chat exchange
            feedback: The feedback data to add
            
        Returns:
            True if successful, False otherwise
        """
        if self.db is None:
            await self.connect_with_retry()
            
        try:
            # Try to convert to ObjectId for MongoDB query
            try:
                object_id = ObjectId(chat_id)
                result = await self.db[self.chat_collection_name].update_one(
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
    
    # Helper methods for backward compatibility
    
    def get_chat_collection(self):
        """Get the chat history collection (synchronous version for compatibility)"""
        if not self.db:
            raise ConnectionError("Database not connected. Call connect() first.")
        return self.db[self.chat_collection_name]
    
    def get_summary_collection(self):
        """Get the summary collection (synchronous version for compatibility)"""
        if not self.db:
            raise ConnectionError("Database not connected. Call connect() first.")
        return self.db[self.summary_collection_name]
    
    def get_feedback_collection(self):
        """Get the feedback collection (synchronous version for compatibility)"""
        if not self.db:
            raise ConnectionError("Database not connected. Call connect() first.")
        return self.db[self.feedback_collection_name]
    
    async def close(self):
        """Close the database connection"""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            logger.info("Database connection closed")
    
    def __call__(self):
        """Return the instance itself when called."""
        return self

# Create a singleton instance for easy import
db_manager = DatabaseManager()
