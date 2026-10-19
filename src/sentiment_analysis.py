from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

ASPECT_KEYWORDS = {
    "food_quality": ["food", "taste", "flavor", "quality", "delicious", "fresh"],
    "delivery_speed": ["delivery", "fast", "late", "delay", "arrive", "time"],
    "packaging": ["pack", "package", "box"],
    "price_value": ["price", "value", "cheap", "expensive", "cost"],
    "service": ["service", "seller", "support", "help"],
}

def classify_sentiment(text):
    if not text or str(text).strip() == "":
        return {"compound": 0.0, "label": "Neutral"}
    scores = analyzer.polarity_scores(str(text))
    comp = scores["compound"]
    if comp >= 0.05:
        label = "Positive"
    elif comp <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return {"compound": comp, "label": label}

def extract_aspects(text):
    if not text:
        return {}
    t = str(text).lower()
    res = {}
    for k, kws in ASPECT_KEYWORDS.items():
        if any(kw in t for kw in kws):
            res[k] = True
    return res
