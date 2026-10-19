import pandas as pd
from src.churn_prediction import train_models


def test_train_models_small():
    df = pd.DataFrame({
        "review_count": [1, 1, 2, 1, 3, 2],
        "avg_votes": [10, 25, 5, 40, 8, 60],
        "avg_cost": [300, 500, 250, 800, 400, 600],
        "online_order": [1, 1, 0, 1, 1, 0],
        "book_table": [0, 1, 0, 1, 0, 1],
        "is_churned_proxy": [0, 0, 1, 0, 1, 1],
    })
    res, feats = train_models(df)
    assert len(res) == 3
    assert "F1" in res.columns
    assert "avg_rating" not in feats  # leakage guard: target-defining feature excluded
