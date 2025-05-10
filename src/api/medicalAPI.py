from fastapi import APIRouter, Depends, HTTPException
from src.main.pydentic_models.models import MChatRequest, MChatResponse
from src.retrieval.document_retriever import MedicalDocumentRetriever
from src.utils.db import get_chat_history
from src.utils.settings import settings
from src.utils.gen_utils import get_prompt_template, load_prompt_template
from src.main.pydentic_models.models import response_schemas, MChatResponse, response_schemas_rag
from langchain.output_parsers import PydanticOutputParser, StructuredOutputParser
from src.main.core.llm_engine import ChatboxAI
from src.utils.db import get_chat_history, save_chat_exchange, delete_chat_history,  get_chat_by_id, update_chat_feedback
import json

router = APIRouter(prefix="/MedicalChat", tags=["CHAT"])

# Initialize the retriever
retriever = MedicalDocumentRetriever(index_path="faiss_medical_index")

def get_retriever():
    return retriever

@router.post("/medical-chat", response_model=MChatResponse)
async def medical_chat(
    request: MChatRequest,
    retriever: MedicalDocumentRetriever = Depends(get_retriever)
):
    """
    Process a chat request using RAG with fallback to general LLM.
    
    1. First tries to retrieve relevant medical information from the vector database
    2. If good results are found, uses them to augment the LLM response
    3. If no good results are found, falls back to the general LLM
    """
    user_id = request.user_id
    user_question = request.user_question
    
    # Get chat history for context
    # chat_history = get_chat_history(user_id)
    
    # Step 1: Try to retrieve relevant information from the vector database
    rag_used = False
    context_docs = []
    
    try:
        if retriever.index is not None:
            # Retrieve relevant documents
            retrieved_docs = retriever.retrieve(user_question, k=3)
            
            # Check if the retrieved documents are relevant enough
            if retrieved_docs and len(retrieved_docs) > 0:
                # Extract the content from the retrieved documents
                context_docs = [doc.page_content for doc in retrieved_docs]
                rag_used = True
    except Exception as e:
        print(f"Error during retrieval: {str(e)}")
        # Continue with fallback if retrieval fails
    
    # Step 2: Generate response based on whether we have relevant documents
    if rag_used and context_docs:
        # Format the context for the LLM
        print("Context:")
        print("Length of context_docs:", len(context_docs)) 
        context_text = "\n\n".join(context_docs)
        # print("Context text:", context_text)
        # print("datatype of context_text:", type(context_text))
        prompt_data = get_prompt_template()
        # Create a prompt that includes the retrieved information
        augmented_output_parser = StructuredOutputParser.from_response_schemas(response_schemas_rag)

        # Get format instructions for the prompt
        format_instructions = augmented_output_parser.get_format_instructions()
        augmented_prompt = load_prompt_template(prompt_data["medical_assistant_prompt_with_context"], format_instructions)
        model = ChatboxAI(settings.MODEL_API)
        print ("Prompt: ", augmented_prompt)
        chain = augmented_prompt | model
        # Get response from OpenAI with the augmented prompt
        raw_response = chain.invoke({"user_question": user_question, "context": context_text})
        
        print("Raw LLM Response:", raw_response)
        if hasattr(raw_response, 'content'):
            # If raw_response is a Message object with content attribute
            response_text = raw_response.content
        elif isinstance(raw_response, dict) and 'content' in raw_response:
            # If raw_response is a dict with content key
            response_text = raw_response['content']
        elif isinstance(raw_response, str):
            # If raw_response is already a string
            response_text = raw_response
        else:
            # Fallback
            response_text = str(raw_response)
        # Create the response object
        response = MChatResponse(
            user_id=user_id,
            user_question=user_question,
            assistant_response=response_text,
            source_documents=context_docs,
            rag_used=True
        )
    else:
        # Fallback to general LLM without specific medical context
        prompt_data = get_prompt_template()
        fallback_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        # Get format instructions for the prompt
        format_instructions = fallback_output_parser.get_format_instructions()
        fallback_prompt = load_prompt_template(prompt_data["medical_assistant_fallback_prompt"], format_instructions)
        model = ChatboxAI(settings.MODEL_API)
        chain = fallback_prompt | model 
        # Get response from OpenAI with the fallback prompt
        raw_response = chain.invoke({"user_question": user_question})
        
        print("Raw LLM Response:", raw_response)
        if hasattr(raw_response, 'content'):
            # If raw_response is a Message object with content attribute
            response_text = raw_response.content
        elif isinstance(raw_response, dict) and 'content' in raw_response:
            # If raw_response is a dict with content key
            response_text = raw_response['content']
        elif isinstance(raw_response, str):
            # If raw_response is already a string
            response_text = raw_response
        # Create the response object
        response = MChatResponse(
            user_id=user_id,
            user_question=user_question,
            assistant_response=response_text,
            source_documents=[],
            rag_used=False
        )
    saved_chat = save_chat_exchange(
        user_id=user_id, 
        user_question=user_question, 
        assistant_response=response_text,
        source_documents=context_docs,
        rag_used=rag_used
    )
        # Add the chat ID to the response
    response.chat_id = saved_chat.get("_id")
    return response

@router.get("/history/{user_id}")
async def get_user_chat_history(user_id: str, limit: int = 10):
    """Get chat history for a specific user."""
    try:
        history = get_chat_history(user_id, limit)
        return {"user_id": user_id, "chat_history": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

@router.delete("/history/{user_id}")
async def delete_user_chat_history(user_id: str):
    """Delete all chat history for a specific user."""
    try:
        success = delete_chat_history(user_id)
        if success:
            return {"message": f"Chat history for user {user_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to delete chat history for user {user_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting chat history: {str(e)}")

@router.get("/chat/{chat_id}")
async def get_chat(chat_id: str):
    """Get a specific chat exchange by ID."""
    chat = get_chat_by_id(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail=f"Chat with ID {chat_id} not found")
    return chat

@router.post("/chat/{chat_id}/feedback")
async def add_feedback_to_chat(chat_id: str, feedback: dict):
    """Add feedback to a specific chat exchange."""
    chat = get_chat_by_id(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail=f"Chat with ID {chat_id} not found")
    
    success = update_chat_feedback(chat_id, feedback)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to update feedback for chat {chat_id}")
    
    return {"message": f"Feedback added to chat {chat_id} successfully"}
#use chatboxAI to get the response from the user question and summary of the chat history
#save the chat history to the database