from src.sentiment_analysis import classify_sentiment

def test_sentiment_positive():
    r = classify_sentiment("I love this product")
    assert r["label"] in ["Positive", "Neutral"]

def test_sentiment_negative():
    r = classify_sentiment("bad terrible worst")
    assert r["label"] in ["Negative", "Neutral"]

def test_sentiment_empty():
    r = classify_sentiment("")
    assert r["label"] == "Neutral"
