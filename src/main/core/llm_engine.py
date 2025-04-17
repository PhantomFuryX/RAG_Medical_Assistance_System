from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from src.utils.config import OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY

# You can also use HuggingFacePipeline if you're hosting LLaMA locally  # uses your OpenAI API key

# Prompt Template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful medical assistant for elderly people. Be kind and clear."),
    ("human", "{input}")
])

# def generate_response(query: str) -> str:
#     messages = chat_template.format_messages(input=query)
#     response = llm(messages)
#     return response.content


def ChatboxAI(chat_model):
    if chat_model == 'openai':
        api_key = OPENAI_API_KEY
        return ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=150
        )
    elif chat_model == 'anthropic':
        api_key = ANTHROPIC_API_KEY
        return ChatAnthropic(
            anthropic_api_key=api_key,
            model="claude-v1",
            temperature=0.7,
            max_tokens_to_sample=150
        )
    elif chat_model == 'deepseek':
        api_key = DEEPSEEK_API_KEY
        return ChatDeepSeek(
            api_key=api_key,
            model='deepseek-chat',
            temperature=0.7,
            max_tokens=500
        )
    else:
        raise ValueError("Invalid chat model specified.")