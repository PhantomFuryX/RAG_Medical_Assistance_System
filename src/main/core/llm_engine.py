from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import json
import os
from typing import Dict, Any, Optional, List
# from src.utils.config import OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY
from src.utils.settings import settings
from src.utils.db_manager import db_manager
from src.retrieval.document_retriever import retrieve_relevant_documents
from src.services.conversation_service import ConversationService

from src.utils.logger import get_llm_logger

# Configure logging
logger = get_llm_logger()

# System prompt for medical assistant
SYSTEM_PROMPT = """You're a helpful medical assistant for elderly people. Be kind, clear, and concise.
Always prioritize patient safety. If a question suggests a serious medical condition, advise consulting a healthcare professional.
Provide general information only and clarify you're not a substitute for professional medical advice."""

OPENAI_API_KEY = settings.OPENAI_API_KEY
ANTHROPIC_API_KEY = settings.ANTHROPIC_API_KEY
DEEPSEEK_API_KEY = settings.DEEPSEEK_API_KEY
def ChatboxAI(chat_model, temperature=0.7, max_tokens=150):
    """
    Initialize a chat model based on the specified provider.
    
    Args:
        chat_model: The model provider ('openai', 'anthropic', or 'deepseek')
        temperature: Controls randomness (higher = more random)
        max_tokens: Maximum number of tokens in the response
        
    Returns:
        A configured chat model instance
    """
    if chat_model == 'openai':
        logger.info("Using OpenAI API")
        api_key = OPENAI_API_KEY
        return ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-4",
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif chat_model == 'anthropic':
        api_key = ANTHROPIC_API_KEY
        return ChatAnthropic(
            anthropic_api_key=api_key,
            model="claude-v1",
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif chat_model == 'deepseek':
        logger.info("Using DeepSeek API")
        api_key = DEEPSEEK_API_KEY
        return ChatDeepSeek(
            api_key=api_key,
            model='deepseek-chat',
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        raise ValueError(f"Invalid chat model specified: {chat_model}")

def load_prompts() -> Dict[str, Any]:
    """
    Load prompts from the prompts.json file
    
    Returns:
        Dictionary containing all prompts
    """
    prompts_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "utils", "prompts.json")
    try:
        with open(prompts_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading prompts: {str(e)}")
        return {}

def generate_response(user_question: str, context: Optional[str] = None, chat_model: str = 'openai') -> str:
    """
    Generate a response to a user question using the appropriate prompt based on context availability.
    
    Args:
        user_question: The user's question
        context: Optional context from previous conversations or RAG
        chat_model: The model to use ('openai', 'anthropic', or 'deepseek')
        
    Returns:
        The generated response
    """
    prompts = load_prompts()
    
    # Initialize the chat model
    model = ChatboxAI(chat_model)
    
    # Choose the appropriate prompt based on whether context is available
    if context:
        logger.info("Context ")
        # Use the prompt with context
        prompt_template = prompts.get("medical_assistant_prompt_with_context", {})
        if not prompt_template:
            logger.warning("Medical assistant prompt with context not found, using fallback")
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Context: {context}\n\nQuestion: {user_question}")
            ]
        else:
            template = prompt_template.get("template", "")
            template = template.replace("{{context_text}}", context)
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=template.format(user_question=user_question))
            ]
    else:
        # Use the fallback prompt without context
        prompt_template = prompts.get("medical_assistant_fallback_prompt", {})
        if not prompt_template:
            logger.warning("Medical assistant fallback prompt not found, using default")
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_question)
            ]
        else:
            template = prompt_template.get("template", "")
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=template.format(user_question=user_question))
            ]
    
    try:
        # Generate response
        response = model.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I'm sorry, I encountered an error while processing your question. Please try again later."

async def generate_response_with_rag(user_id: str, user_question: str, chat_model: str = 'openai') -> Dict[str, Any]:
    """
    Generate a response using RAG (Retrieval Augmented Generation) if possible.
    
    Args:
        user_id: The user's ID
        user_question: The user's question
        chat_model: The model to use ('openai', 'anthropic', or 'deepseek')
        
    Returns:
        Dictionary containing the response and metadata
    """
    try:
        if retrieve_relevant_documents is None:
            raise ImportError("retrieve_relevant_documents is not available.")

            # Check if retriever is available
            retriever_available = False
        else:
            logger.info("relevent document available using RAG")
            retriever_available = True
        
        # Get conversation context
        conversation_service = None
        conversation_service = ConversationService()
        
        context = ""
        source_documents = []
        rag_used = False
        
        # Try to retrieve relevant documents if retriever is available
        if retriever_available:
            try:
                retrieval_results = retrieve_relevant_documents(user_question, top_k=5)
                logger.info("Found relevant documents using RAG with top k=5")
                if retrieval_results and retrieval_results.get('documents'):
                    source_documents = [doc.get('title', 'Unknown') for doc in retrieval_results['documents']]
                    context_docs = "\n\n".join([doc.get('content', '') for doc in retrieval_results['documents']])
                    context = f"Retrieved medical information:\n{context_docs}\n\n"
                    rag_used = True
            except Exception as e:
                logger.error(f"Error retrieving documents: {str(e)}")
        
        # Add conversation context if available
        if conversation_service:
            try:
                conversation_context = await conversation_service.get_cumulative_summary(user_id)
                if conversation_context:
                    if context:
                        context += f"Previous conversation context:\n{conversation_context}"
                    else:
                        context = f"Previous conversation context:\n{conversation_context}"
            except Exception as e:
                logger.error(f"Error getting conversation context: {str(e)}")
        
        # Generate response
        response_text = generate_response(user_question, context, chat_model)
        
        # Save the exchange to the database
        try:
            await db_manager.save_chat_exchange(
                user_id=user_id,
                user_question=user_question,
                assistant_response=response_text,
                source_documents=source_documents,
                rag_used=rag_used
            )
        except Exception as e:
            logger.error(f"Error saving chat exchange: {str(e)}")
        
        # Return the response with metadata
        return {
            "response": response_text,
            "rag_used": rag_used,
            "source_documents": source_documents
        }
    except Exception as e:
        logger.error(f"Error in generate_response_with_rag: {str(e)}")
        # Fallback to simple response
        response_text = generate_response(user_question, None, chat_model)
        return {
            "response": response_text,
            "rag_used": False,
            "source_documents": []
        }

def build_chain_from_prompt(prompt_name: str, chat_model: str = 'openai'):
    """
    Build a chain using a specific prompt from prompts.json
    
    Args:
        prompt_name: The name of the prompt in prompts.json
        chat_model: The model to use
        
    Returns:
        A configured chain
    """
    prompts = load_prompts()
    prompt_config = prompts.get(prompt_name)
    
    if not prompt_config:
        raise ValueError(f"Prompt '{prompt_name}' not found in prompts.json")
    
    template = prompt_config.get("template", "")
    input_variables = prompt_config.get("input_variables", [])
    
    model = ChatboxAI(chat_model)
    prompt = PromptTemplate(
        template=template,
        input_variables=input_variables
    )
    
    chain = model | prompt
    return chain
