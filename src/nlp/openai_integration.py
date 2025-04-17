from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
# Set the OpenAI API key
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def get_openai_response(prompt: str) -> str:
    response = client.responses.create(
        model="gpt-4o",
        input=prompt
    )
    return response.output_text
