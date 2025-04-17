# from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from textblob import TextBlob
from src.main.core.llm_engine import ChatboxAI
# from src.utils.config import OPENAI_API_KEY
from src.utils.config import DEEPSEEK_API_KEY

def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment < -0.3:
        return "negative"
    elif sentiment > 0.3:
        return "positive"
    else:
        return "neutral"

# def load_prompt_template():
    
def build_diagnosis_chain(chat_model):
    model = ChatboxAI(chat_model)
    template = PromptTemplate(
        input_variables=["symptoms"],
        template="A patient reports the following symptoms: {symptoms}. What possible conditions could this indicate?"
    )
    chain =  model | template
    return chain

# Example Usage
if __name__ == "__main__":
    chain = build_diagnosis_chain(DEEPSEEK_API_KEY, 'deepseek')
    
    user_symptoms = "I have a fever and body aches."
    response = chain.run({"symptoms": user_symptoms})
    
    sentiment = analyze_sentiment(response)
    print("Diagnosis:", response)
    print("Sentiment:", sentiment)