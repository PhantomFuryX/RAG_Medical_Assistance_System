from src.main.core.llm_engine import ChatboxAI
from langchain.prompts import PromptTemplate
from src.utils.settings import settings

def build_diagnosis_chain(chat_model):
    """
    Build a diagnosis chain using the LLM engine
    """
    model = ChatboxAI(chat_model)
    template = PromptTemplate(
        input_variables=["symptoms"],
        template="A patient reports the following symptoms: {symptoms}. What possible conditions could this indicate?"
    )
    chain = model | template
    return chain

def analyze_sentiment(text):
    """
    Analyze the sentiment of a text
    """
    # This is a placeholder - you might want to implement a proper sentiment analysis
    if "serious" in text.lower() or "emergency" in text.lower():
        return "urgent"
    elif "mild" in text.lower() or "common" in text.lower():
        return "non-urgent"
    else:
        return "neutral"

# Example Usage
if __name__ == "__main__":
    chain = build_diagnosis_chain(settings.MODEL_API)
    
    user_symptoms = "I have a fever and body aches."
    response = chain.invoke({"symptoms": user_symptoms})
    
    sentiment = analyze_sentiment(response)
    print("Diagnosis:", response)
    print("Sentiment:", sentiment)
