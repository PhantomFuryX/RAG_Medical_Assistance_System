# from langchain.chat_models import ChatOpenAI, ChatAnthropic, ChatDeepseek
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from textblob import TextBlob
# from src.utils.config import OPENAI_API_KEY
from src.utils.config import DEEPSEEK_API_KEY
from src.utils.settings import settings
import os
from src.main.core.llm_engine import ChatboxAI
from src.main.pydentic_models.models import response_schemas
from src.utils.gen_utils import get_prompt_template, build_chatting_chain
import json

chat_history = []

# Create the parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Get format instructions for the prompt
format_instructions = output_parser.get_format_instructions()

def build_chat_history(messages):  
    for message in messages:
        if message['role'] == 'user':
            chat_history.append(f"User: {message['content']}")
        elif message['role'] == 'assistant':
            chat_history.append(f"Assistant: {message['content']}")
    return "\n".join(chat_history)

def format_chat_history(chat_history):
    return "\n".join([f"User: {entry['user']}\nAI: {entry['ai']}" for entry in chat_history])

def load_prompt_template(messages):
    # chat_history = build_chat_history(messages)
    prompt_data = get_prompt_template()
    prompt = prompt_data['medical_assistant_prompt']
    prompt = PromptTemplate(
        input_variables=prompt['input_variables'],
        template=prompt['template'],
        partial_variables={"format_instructions": format_instructions}
    )
    return prompt

def get_llm_response(user_question: str, summary: str):

    chain = build_chatting_chain(settings.MODEL_API)
    
    input_data = {
        "summary": summary,
        "user_question": user_question
    }
    raw_response = chain.invoke(input_data)
    # print(raw_response)
    # Ensure the response is valid JSON
    if isinstance(raw_response, dict):
        parsed = raw_response
    else:
        # Ensure the response is valid JSON
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed = {"error": "Failed to parse LLM response as JSON", "raw_response": raw_response}
    
    return parsed

def build_chat_chain(chat_model, summary, user_question):
    """
    Build a chat chain using prompts from prompts.json
    """
    # Load prompts
    prompts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils", "prompts.json")
    with open(prompts_path, "r") as f:
        prompts = json.load(f)
    
    # Get the medical assistant prompt
    medical_prompt = prompts.get("medical_assistant_prompt", {})
    if not medical_prompt:
        raise ValueError("Medical assistant prompt not found in prompts.json")
    
    # Create the prompt template
    template = medical_prompt.get("template", "")
    input_variables = medical_prompt.get("input_variables", [])
    
    # Create the chain
    model = ChatboxAI(chat_model)
    prompt_template = PromptTemplate(
        template=template,
        input_variables=input_variables
    )
    
    chain = model | prompt_template
    
    # Prepare input data
    input_data = {
        "summary": summary,
        "user_question": user_question
    }
    
    # Generate response
    raw_response = chain.invoke(input_data)
    
    return raw_response

# Example Usage
if __name__ == "__main__":
    chain = build_chatting_chain(settings.MODEL_API)
    raw_response = chain.run(summary="", user_question="")
    print(raw_response)