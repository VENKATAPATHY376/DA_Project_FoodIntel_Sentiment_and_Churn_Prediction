import pandas as pd
import numpy as np
from src.customer_segmentation import hierarchical_linkage, dendrogram_fig


def _fake_reviewers(n=60):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "app": rng.choice(["Swiggy", "Zomato", "Zepto"], n),
        "reviewer": [f"u{i}" for i in range(n)],
        "total_reviews": rng.integers(1, 6, n),
        "avg_score": rng.uniform(1, 5, n).round(2),
        "tenure_days": rng.uniform(0, 5, n).round(2),
        "avg_gap_days": rng.uniform(0, 3, n).round(2),
        "days_since_last_review": rng.uniform(0, 5, n).round(2),
        "is_churned_proxy": rng.integers(0, 2, n),
    })


def test_hierarchical_linkage_shapes():
    df = _fake_reviewers()
    Z, sample, cols = hierarchical_linkage(df)
    assert Z.shape[0] == len(sample) - 1  # n-1 merges for n points
    assert len(cols) >= 2
    assert len(sample) == len(df)  # n <= sample_n


def test_dendrogram_fig_renders():
    df = _fake_reviewers()
    Z, _, _ = hierarchical_linkage(df)
    fig = dendrogram_fig(Z)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_dendrogram_sample_cap():
    df = _fake_reviewers(500)
    Z, sample, _ = hierarchical_linkage(df, sample_n=100)
    assert len(sample) == 100
    assert Z.shape[0] == 99
