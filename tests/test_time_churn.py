import pandas as pd
import numpy as np
from src.preprocessing import (build_reviewer_features, derive_churn_threshold,
                               pick_churn_features, normalize_playstore)
from src.churn_prediction import train_models


def _fake_reviews():
    base = pd.Timestamp("2026-10-17")
    rows = []
    # Some recent reviewers and some inactive ones (real timestamps)
    for i in range(20):
        recency = i % 10  # 0..9 days ago
        rows.append({
            "app": "Swiggy", "reviewer": f"u{i}", "review_text": "good app" * 5,
            "score": 3 + (i % 3), "review_date": base - pd.Timedelta(days=recency),
            "thumbsUpCount": i,
        })
    return pd.DataFrame(rows)


def test_normalize_playstore():
    raw = pd.DataFrame({
        "content": ["nice app", ""],
        "userName": ["a", "b"],
        "score": [5, 1],
        "at": ["2026-10-01", "2026-10-02"],
        "app": ["Swiggy", "Swiggy"],
    })
    out = normalize_playstore(raw)
    assert len(out) == 1  # empty text dropped
    assert "review_text" in out.columns and "review_date" in out.columns


def test_build_reviewer_features_time_based():
    rv = _fake_reviews()
    feats, thresh, max_date, window = build_reviewer_features(rv)
    assert not feats.empty
    assert "days_since_last_review" in feats.columns
    assert "is_churned_proxy" in feats.columns
    assert thresh >= 1
    assert window >= 1
    # inactivity label should mark the least-recent reviewers
    assert feats["is_churned_proxy"].mean() > 0


def test_derive_churn_threshold_positive():
    s = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert derive_churn_threshold(s) >= 1


def test_pick_churn_features_excludes_label_definer():
    feats_time = pick_churn_features(pd.DataFrame({
        "total_reviews": [1], "avg_score": [3], "score_std": [0],
        "negative_ratio": [0], "tenure_days": [1], "avg_gap_days": [1],
        "avg_thumbs": [0], "days_since_last_review": [5],  # label-definer, must be excluded
        "is_churned_proxy": [1],
    }))
    assert "days_since_last_review" not in feats_time
    assert "is_churned_proxy" not in feats_time


def test_train_models_time_based():
    rv = _fake_reviews()
    feats, _, _, _ = build_reviewer_features(rv)
    feats = feats.fillna(0)
    res, used = train_models(feats)
    assert len(res) == 3
    assert "days_since_last_review" not in used
