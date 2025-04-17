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

    chain = build_chatting_chain('deepseek')
    
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
    
# Example Usage
if __name__ == "__main__":
    chain = build_chatting_chain('deepseek')
    raw_response = chain.run(summary="", user_question="")
    print(raw_response)