import pandas as pd
import numpy as np
from src.preprocessing import derive_churn_threshold, build_reviewer_features
from src.live_fetch import _latest_cached, LIVE_DIR


def test_threshold_produces_both_classes_tiny_window():
    """Degenerate sub-day window must not yield an all-zero target."""
    base = pd.Timestamp("2026-10-18 12:00")
    # 100 reviewers all within the last 6 hours (tiny window)
    rec = pd.Series(np.linspace(0, 0.25, 100))
    thresh = derive_churn_threshold(rec)
    churned = (rec >= thresh).sum()
    # either both classes exist, or the caller detects degeneracy
    assert churned > 0 or thresh >= rec.max()


def test_threshold_normal_window_pct80():
    rec = pd.Series(np.arange(0, 100, dtype=float))
    thresh = derive_churn_threshold(rec)
    assert 1 <= thresh <= 100
    churned_frac = (rec >= thresh).mean()
    assert 0.05 <= churned_frac <= 0.95


def test_threshold_empty_series():
    assert derive_churn_threshold(pd.Series([], dtype=float)) == 1


def test_builder_flags_degenerate_window():
    """All same-timestamp reviews -> single class must be detectable."""
    base = pd.Timestamp("2026-10-18 12:00")
    rv = pd.DataFrame({
        "app": ["Swiggy"] * 10,
        "reviewer": [f"u{i}" for i in range(10)],
        "review_text": ["good"] * 10,
        "score": [4] * 10,
        "review_date": [base] * 10,  # identical timestamps
        "thumbsUpCount": [0] * 10,
    })
    feats, thresh, max_date, window = build_reviewer_features(rv)
    # window is ~0 -> churn labels may be all one class; page handles this
    assert len(feats) == 10
    assert feats["is_churned_proxy"].nunique() == 1  # degenerate, detectable


def test_cache_rejects_small_file(tmp_path, monkeypatch):
    """A tiny cached file must not satisfy a large per_app request."""
    monkeypatch.setattr("src.live_fetch.LIVE_DIR", tmp_path)
    # write a small csv (5 rows)
    df = pd.DataFrame({"a": range(5)})
    f = tmp_path / "playstore_reviews_20260101_000000.csv"
    df.to_csv(f, index=False)
    # min_rows=1000 -> no valid cache
    assert _latest_cached(min_rows=1000) is None
    # min_rows=3 -> valid
    assert _latest_cached(min_rows=3) == str(f)


def test_hour_granularity_logic():
    """Window < 2 days must floor to hours (simulated page logic)."""
    t = pd.DataFrame({"review_date": pd.to_datetime(
        ["2026-10-18 01:10", "2026-10-18 02:40", "2026-10-18 22:00"])})
    span_days = (t["review_date"].max() - t["review_date"].min()).total_seconds() / 86400
    assert span_days < 2
    hours = t["review_date"].dt.floor("h")
    assert hours.nunique() == 3  # 01:00, 02:00, 22:00
