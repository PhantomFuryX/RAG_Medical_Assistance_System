from typing import List, Dict, Optional
import datetime
import json
import os
import logging
import html
from src.utils.db_manager import db_manager

logger = logging.getLogger("conversation_service")

class ConversationService:
    def __init__(self):
        # Load prompts from JSON file
        prompts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils", "prompts.json")
        with open(prompts_path, "r") as f:
            self.prompts = json.load(f)
    
    async def store_conversation(self, user_id: str, question: str, response: str):
        """Store a conversation in the database"""
        try:
            await db_manager.save_chat_exchange(
                user_id=user_id,
                user_question=question,
                assistant_response=response
            )
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")
            raise
    
    async def get_user_conversations(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversations for a user"""
        try:
            return await db_manager.get_chat_history(user_id, limit)
        except Exception as e:
            logger.error(f"Error retrieving user conversations: {str(e)}")
            return []
    
    async def generate_and_store_summary(self, user_id: str, max_words: int = 500) -> str:
        """Generate a summary of recent conversations and store it"""
        try:
            # Get recent conversations
            conversations = await self.get_user_conversations(user_id, limit=10)
            
            if not conversations:
                return "No previous conversations found."
            
            # Format conversations for summarization
            conversation_text = ""
            for conv in conversations:
                conversation_text += f"User: {conv.get('user_question', '')}\nAssistant: {conv.get('assistant_response', '')}\n\n"
            # Escape HTML characters in conversation_text
            safe_conversation_text = html.escape(conversation_text)
            
            # Use the summary_generation_prompt from prompts.json
            summary_prompt = self.prompts.get("summary_generation_prompt", {})
            if summary_prompt:
                template = summary_prompt.get("template", "")
                prompt = template.replace("{conversation_text}", safe_conversation_text).replace("{max_words}", str(max_words))
            else:
                # Fallback prompt
                prompt = f"""
                Summarize the following medical conversation history in a concise way, 
                highlighting key medical information, questions, and advice given.
                Limit the summary to approximately {max_words} words.
                
                Conversation history:
                {safe_conversation_text}
                
                Summary:
                """
            
            # Generate summary using LLM
            summary = await self._generate_response_async(prompt)
            
            # Store the summary
            await db_manager.save_chat_summary(user_id, summary)
            
            return summary
        except Exception as e:
            logger.error(f"Error generating and storing summary: {str(e)}")
            raise
    
    async def get_summary_history(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Get the history of summaries for a user"""
        try:
            return await db_manager.get_user_summaries(user_id, limit)
        except Exception as e:
            logger.error(f"Error retrieving summary history: {str(e)}")
            return []
    
    async def get_cumulative_summary(self, user_id: str, max_summaries: int = 5, max_words: int = 500) -> str:
        """
        Get a cumulative summary from the last few conversation summaries
        """
        try:
            # Get recent summaries
            
            logger.info(f"Fetching summary history for user {user_id} with limit {max_summaries}.")
            summaries = await self.get_summary_history(user_id, limit=max_summaries)
            
            if not summaries:
                logger.error("No summaries found for user.")
                return ""
            logger.info(f"Found {len(summaries)} summaries for user {user_id}.")
            # Extract content from summaries
            summary_texts = [summary.get('content', '') for summary in summaries]
            
            # If there's only one summary, return it directly
            if len(summary_texts) == 1:
                return summary_texts[0]
            
            # Format the summaries
            formatted_summaries = "\n\n".join([f"Summary {i+1}: {text}" for i, text in enumerate(summary_texts)])
            
            # Use the cumulative_summary_prompt from prompts.json
            cumulative_prompt = self.prompts.get("cumulative_summary_prompt", {})
            if cumulative_prompt:
                template = cumulative_prompt.get("template", "")
                prompt = template.replace("{formatted_summaries}", formatted_summaries)
                prompt = prompt.replace("{max_words}", str(max_words))
                prompt = prompt.replace("{len_summaries}", str(len(summary_texts)))
            else:
                # Fallback prompt
                prompt = f"""
                Below are {len(summary_texts)} summaries from previous medical conversations with a user.
                Create a comprehensive cumulative summary that captures the key medical information,
                questions, and advice across all these summaries. Focus on medical relevance and continuity.
                Limit the summary to approximately {max_words} words.
                
                {formatted_summaries}
                
                Cumulative Summary:
                """
            
            # Generate cumulative summary using LLM
            cumulative_summary = await self._generate_response_async(prompt)
            
            return cumulative_summary
        except Exception as e:
            logger.error(f"Error generating cumulative summary: {str(e)}")
            return ""
    
    async def _generate_response_async(self, prompt: str) -> str:
        # If generate_response is async, use await; otherwise, run in executor
        import asyncio
        loop = asyncio.get_event_loop()
        try:
            from src.main.core.llm_engine import generate_response
            return await loop.run_in_executor(None, generate_response, prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "Sorry, I couldn't generate a summary at this time."
