"""
Tests for the Sentiment Analysis Module.
"""

from modules.sentiment import analyze_sentiment, get_sentiment_response


def test_positive_sentiment():
    result = analyze_sentiment("I absolutely love this service!")
    assert result["sentiment"] == "Positive"


def test_negative_sentiment():
    result = analyze_sentiment("This service is terrible and disappointing.")
    assert result["sentiment"] == "Negative"


def test_neutral_sentiment():
    result = analyze_sentiment("Where is my order?")
    assert result["sentiment"] == "Neutral"


def test_empty_message():
    result = analyze_sentiment("")
    assert result["sentiment"] == "Neutral"


def test_confidence_range():
    result = analyze_sentiment("The customer support was amazing!")
    assert 0.0 <= result["confidence"] <= 1.0


def test_positive_response():
    result = get_sentiment_response("I really like your service!")

    assert result["sentiment"] == "Positive"
    assert result["response"]
    assert isinstance(result["response"], str)


def test_negative_response():
    result = get_sentiment_response(
        "I am extremely disappointed with your service."
    )

    assert result["sentiment"] == "Negative"
    assert result["response"]
    assert isinstance(result["response"], str)


def test_neutral_response():
    result = get_sentiment_response("Where is my order?")

    assert result["sentiment"] == "Neutral"
    assert result["response"]
    assert isinstance(result["response"], str)


def test_result_structure():
    result = get_sentiment_response("The service was excellent.")

    expected_keys = {
        "message",
        "sentiment",
        "confidence",
        "compound",
        "response",
    }

    assert expected_keys.issubset(result.keys())