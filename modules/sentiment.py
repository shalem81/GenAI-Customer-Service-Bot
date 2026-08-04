"""
Sentiment Analysis Module
GenAI Customer Service Bot

Detects positive, negative, and neutral sentiment in customer messages.
Uses VADER sentiment analysis.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Create analyzer once and reuse it
analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of a user message.

    Returns:
        {
            "sentiment": "Positive" | "Negative" | "Neutral",
            "confidence": float,
            "compound": float,
            "scores": {...}
        }
    """

    if not text or not text.strip():
        return {
            "sentiment": "Neutral",
            "confidence": 1.0,
            "compound": 0.0,
            "scores": {
                "neg": 0.0,
                "neu": 1.0,
                "pos": 0.0,
                "compound": 0.0,
            },
        }

    scores = analyzer.polarity_scores(text)

    compound = scores["compound"]

    # Standard VADER thresholds
    if compound >= 0.05:
        sentiment = "Positive"
        confidence = scores["pos"]

    elif compound <= -0.05:
        sentiment = "Negative"
        confidence = scores["neg"]

    else:
        sentiment = "Neutral"
        confidence = scores["neu"]

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 3),
        "compound": round(compound, 3),
        "scores": scores,
    }


def get_sentiment_response(text: str) -> dict:
    """
    Analyze a customer message and generate an appropriate
    customer-service response.
    """

    result = analyze_sentiment(text)

    sentiment = result["sentiment"]

    if sentiment == "Positive":
        response = (
            "I'm glad to hear that! Thank you for sharing your positive "
            "experience. How else can I help you today?"
        )

    elif sentiment == "Negative":
        response = (
            "I'm sorry to hear about your experience. I understand your "
            "frustration. Please tell me more about the issue so I can "
            "help you resolve it."
        )

    else:
        response = (
            "Thank you for your message. Please provide any additional "
            "details, and I'll do my best to assist you."
        )

    return {
        "message": text,
        "sentiment": sentiment,
        "confidence": result["confidence"],
        "compound": result["compound"],
        "response": response,
    }


if __name__ == "__main__":

    test_messages = [
        "I absolutely love your service!",
        "This is terrible and I am very disappointed.",
        "Where is my order?",
    ]

    for message in test_messages:

        result = get_sentiment_response(message)

        print("-" * 60)
        print("Message:", result["message"])
        print("Sentiment:", result["sentiment"])
        print("Confidence:", result["confidence"])
        print("Compound:", result["compound"])
        print("Response:", result["response"])