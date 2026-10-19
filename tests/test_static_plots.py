import pandas as pd
import numpy as np
from src.plotting_static import generate_all_static, seaborn_corr_heatmap, seaborn_violin


def _fake():
    reviews = pd.DataFrame({
        "app": ["Swiggy", "Zomato", "Zepto"] * 10,
        "score": np.random.default_rng(0).integers(1, 6, 30),
        "sent_label": ["Positive", "Negative", "Neutral"] * 10,
    })
    reviewers = pd.DataFrame({
        "total_reviews": np.random.default_rng(1).integers(1, 6, 30),
        "avg_score": np.random.default_rng(2).uniform(1, 5, 30),
        "days_since_last_review": np.random.default_rng(3).uniform(0, 5, 30),
        "avg_gap_days": np.random.default_rng(4).uniform(0, 3, 30),
        "is_churned_proxy": np.random.default_rng(5).integers(0, 2, 30),
    })
    return reviews, reviewers


def test_generate_all_static_returns_pngs():
    r, c = _fake()
    figs = generate_all_static(r, c)
    assert len(figs) >= 3
    for p in figs.values():
        assert str(p).endswith(".png")
        assert p.exists()


def test_seaborn_corr_heatmap_path():
    _, c = _fake()
    p = seaborn_corr_heatmap(c, name="test_corr.png")
    assert p is not None and p.exists()
    p.unlink(missing_ok=True)


def test_seaborn_violin_path():
    r, _ = _fake()
    p = seaborn_violin(r, name="test_violin.png")
    assert p is not None and p.exists()
    p.unlink(missing_ok=True)


def test_generate_all_static_handles_empty():
    figs = generate_all_static(pd.DataFrame(), pd.DataFrame())
    assert figs == {}
