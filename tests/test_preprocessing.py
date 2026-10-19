from src.preprocessing import detect_lang, is_english_like, extract_reviews


def test_detect_lang_basic():
    # detect_lang retained as optional helper; must not crash on empty inputs
    assert detect_lang("") == "empty"
    assert detect_lang(None) == "empty"


def test_is_english_like():
    assert is_english_like("The food was great and delivery was fast") is True
    assert is_english_like("short") is False
    assert is_english_like("") is False


def test_extract_reviews():
    blob = "('Rated 5.0', '\\n  Great food and quick delivery.\\n  '), ('Rated 4.0', '\\n  Nice place.\\n  ')"
    revs = extract_reviews(blob, limit=3)
    assert len(revs) == 2
    assert "Great food" in revs[0]
