import os
import json
import time
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, StructuredOutputParser
from src.main.core.llm_engine import ChatboxAI
from src.main.pydentic_models.models import response_schemas, MChatResponse

pydanticop = PydanticOutputParser(pydantic_object=MChatResponse)
format_instructions1 = pydanticop.get_format_instructions()

from src.utils.config import DEEPSEEK_API_KEY
prompt_name = "prompts.json"
def get_current_timestamp():
    return int(time.time())

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

def build_chatting_chain(chat_model, messages=None):
    model = ChatboxAI(chat_model)
    parser = pydanticop
    prompt = load_prompt_template(messages)
    chain = prompt | model | parser
    return chain