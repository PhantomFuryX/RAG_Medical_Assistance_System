import os
import json
import time
from langchain.prompts import PromptTemplate
from typing import Dict, Any, Optional
from langchain.output_parsers import PydanticOutputParser, StructuredOutputParser
from src.main.core.llm_engine import ChatboxAI
from src.main.pydentic_models.models import response_schemas, MChatResponse
import functools

pydanticop = PydanticOutputParser(pydantic_object=MChatResponse)
format_instructions1 = pydanticop.get_format_instructions()

from src.utils.config import DEEPSEEK_API_KEY
prompt_name = "prompts.json"
def get_current_timestamp():
    return int(time.time())

@functools.lru_cache(maxsize=16)
def get_prompt_template():
    with open(os.path.join(os.path.dirname(__file__), prompt_name), "r") as f:
        prompt_data = json.load(f)
    return prompt_data

def load_prompt_template(prompt_data, format_instructions):
    # chat_history = build_chat_history(messages)
    prompt = PromptTemplate(
        input_variables=prompt_data['input_variables'],
        template=prompt_data['template'],
        partial_variables={"format_instructions": format_instructions}
    )
    return prompt

def load_prompts() -> Dict[str, Any]:
    """Load prompts from the prompts.json file"""
    prompts_path = os.path.join(os.path.dirname(__file__), "prompts.json")
    with open(prompts_path, "r") as f:
        return json.load(f)

def get_prompt_template_by_key(prompt_key: str) -> Optional[PromptTemplate]:
    """
    Get a specific prompt template by key
    
    Args:
        prompt_key: The key of the prompt in prompts.json
        
    Returns:
        A PromptTemplate object or None if not found
    """
    prompts = load_prompts()
    
    if prompt_key not in prompts:
        return None
    
    prompt_data = prompts[prompt_key]
    
    # Create a PromptTemplate from the data
    template = prompt_data["template"]
    input_variables = prompt_data["input_variables"]
    
    # Handle partial variables if present
    partial_variables = prompt_data.get("partial_variables", {})
    
    return PromptTemplate(
        template=template,
        input_variables=input_variables,
        partial_variables=partial_variables
    )
    
def build_chatting_chain(chat_model, messages=None):
    model = ChatboxAI(chat_model)
    parser = pydanticop
    prompt = load_prompt_template(messages, format_instructions1)
    chain = prompt | model | parser
    return chain