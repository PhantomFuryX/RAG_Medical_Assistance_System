from textblob import TextBlob

def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of text using TextBlob."""
    analysis = TextBlob(text)
    
    # Get polarity (-1 to 1) and subjectivity (0 to 1)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    
    # Determine sentiment category
    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "sentiment": sentiment,
        "polarity": polarity,
        "subjectivity": subjectivity
    }
