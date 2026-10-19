import pandas as pd
import numpy as np
import re


def detect_lang(text):
    """Optional language detection (lazy import; not used in hot path)."""
    if text is None or pd.isna(text) or str(text).strip() in ("", "nan"):
        return "empty"
    try:
        from langdetect import detect
        return detect(str(text)[:500])
    except Exception:
        return "unknown"


def extract_reviews(blob: str, limit: int = 5):
    """Split Zomato reviews_list blob into individual review texts (limit per cell)."""
    if not blob or blob == "nan":
        return []
    # Pattern: "('Rated 4.0', '   text...')" repeated
    parts = re.split(r"Rated\s+[0-9.]+", str(blob))
    out = []
    for p in parts[1:]:
        # strip tuple punctuation/quotes
        t = re.sub(r"^[,()'\"]+|[,()'\"]+$", "", p).strip()
        t = re.sub(r"\\n", " ", t).strip()
        if len(t) >= 15:
            out.append(t)
        if len(out) >= limit:
            break
    return out


def is_english_like(text: str) -> bool:
    """Fast ASCII heuristic to filter likely-English review text (no langdetect in hot path)."""
    if not text or len(text) < 15:
        return False
    sample = text[:500]
    ascii_ratio = sum(1 for ch in sample if ord(ch) < 128) / max(len(sample), 1)
    return ascii_ratio > 0.90


def preprocess(df, max_reviews=500):
    df = df.copy()

    # Extract review text column from Zomato structure
    if "reviews_list" in df.columns and "review_text" not in df.columns:
        df["review_text"] = df["reviews_list"].astype(str)

    # Numeric rating from Zomato "rate" column (e.g. "4.1/5", "3.5/5")
    if "rate" in df.columns and "rating" not in df.columns:
        m = df["rate"].astype(str).str.extract(r"([0-9.]+)")
        df["rating"] = pd.to_numeric(m[0], errors="coerce")
    elif "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Sample EARLY to keep all downstream work cheap
    if max_reviews is not None and len(df) > max_reviews:
        df = df.sample(max_reviews, random_state=42)

    # Clean review text
    if "review_text" in df.columns:
        df["review_text"] = (
            df["review_text"].astype(str).str.strip().str.replace(r"[\[\]'\"]", " ", regex=True)
        )
        df.loc[df["review_text"].str.lower() == "nan", "review_text"] = ""

    df = df.drop_duplicates()
    return df


def build_customer_features(df):
    """Customer-level features built ONLY from real available columns.

    Zomato Bangalore dataset has no customer IDs, order dates, or order values.
    Each row is a restaurant-review record, so a "customer" proxy unit is the
    reviewing customer record. Churn proxy is DISFACTOR-based (dissatisfaction),
    NOT inactivity-based, because the dataset has no time dimension per customer.
    This limitation is disclosed in the dashboard.
    """
    out = df.copy()
    out["customer_unique_id"] = out.index
    key = "customer_unique_id"
    agg = out.groupby(key).agg(
        review_count=("review_text", lambda x: int((x.astype(str).str.len() > 0).sum())),
        avg_rating=("rating", "mean"),
        avg_votes=("votes", "mean") if "votes" in out.columns else ("rating", "size"),
    ).reset_index()
    # Engagement / behavioral features available at prediction time (NOT target-derived)
    if "approx_cost(for two people)" in out.columns:
        cost = pd.to_numeric(out["approx_cost(for two people)"].astype(str).str.replace(",", "", regex=False), errors="coerce")
        out2 = out.assign(_cost=cost)
        cost_agg = out2.groupby(key)["_cost"].mean().reset_index().rename(columns={"_cost": "avg_cost"})
        agg = agg.merge(cost_agg, on=key, how="left")
    for bin_col in ["online_order", "book_table"]:
        if bin_col in out.columns:
            b = (out[bin_col].astype(str).str.strip().str.lower() == "yes").astype(int)
            b_agg = out.assign(_b=b).groupby(key)["_b"].first().reset_index().rename(columns={"_b": bin_col})
            agg = agg.merge(b_agg, on=key, how="left")
    agg["avg_rating"] = agg["avg_rating"].fillna(agg["avg_rating"].median())
    # Proxy label: dissatisfaction-based churn risk (NOT inactivity - dataset has no time dimension)
    thresh = 3.0
    agg["is_churned_proxy"] = (agg["avg_rating"] < thresh).astype(int)
    return agg, thresh, None


# ---------------------------------------------------------------------------
# Google Play Store live reviews: normalization + TIME-BASED churn features
# ---------------------------------------------------------------------------

def normalize_playstore(df):
    """Normalize google-play-scraper output into the common review schema."""
    out = df.copy()
    rename = {
        "content": "review_text",
        "userName": "reviewer",
        "at": "review_date",
        "score": "score",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    keep = ["reviewId", "app", "reviewer", "review_text", "score", "review_date",
            "thumbsUpCount", "replyContent"]
    out = out[[c for c in keep if c in out.columns]]
    if "review_date" in out.columns:
        out["review_date"] = pd.to_datetime(out["review_date"], errors="coerce")
    if "review_text" in out.columns:
        out["review_text"] = out["review_text"].astype(str).str.strip()
        out.loc[out["review_text"].str.lower().isin(["", "nan"]), "review_text"] = ""
    out = out.dropna(subset=["review_text"])
    out = out[out["review_text"].str.len() > 0]
    return out.drop_duplicates()


def derive_churn_threshold(days_since_series):
    """Inactivity threshold DERIVED FROM DATA (documented, not arbitrary).

    Live Play Store data is newest-first and spans a short window (popular
    apps accumulate thousands of reviews/day), so an absolute "30-day"
    threshold would label nobody. Instead we use a relative RFM-style rule:
    churn proxy = reviewers at or beyond a high percentile of recency.

    Robustness: the threshold must produce BOTH classes. We try P80, then
    step down (P60, P50, P40) until at least 5% of reviewers are churned.
    If even the widest split cannot create two classes (degenerate tiny
    window), we return the max recency so the caller can detect the
    degenerate case (all-but-newest marked) rather than silently failing.
    The chosen threshold is always displayed in the dashboard.
    """
    s = pd.Series(days_since_series).dropna()
    if s.empty:
        return 1
    n = len(s)
    for pct in (0.8, 0.6, 0.5, 0.4):
        cand = max(1, int(round(s.quantile(pct))))
        churned = (s >= cand).sum()
        if 0.05 * n <= churned <= 0.95 * n:
            return cand
    # Degenerate window: fall back to a split at the max recency value
    # (marks only the very newest as active); caller should treat a
    # resulting single-class target as an insufficient-window condition.
    return max(1, int(round(s.max())))


def build_reviewer_features(reviews):
    """TIME-BASED reviewer features + inactivity churn proxy.

    Unit: (app, reviewer). Labels use REAL review timestamps:
    is_churned_proxy = 1 if the reviewer's last review is >= threshold days
    before the dataset's max review date. days_since_last_review defines the
    label, so it must NOT be used as a model feature (leakage guard).
    Returns (features_df, threshold_days, max_date, window_days).
    """
    rv = reviews.dropna(subset=["review_date"]).copy()
    if rv.empty:
        return pd.DataFrame(), 0, None, 0

    key = ["app", "reviewer"]
    feats = rv.groupby(key).agg(
        total_reviews=("review_text", "count"),
        avg_score=("score", "mean"),
        score_std=("score", "std"),
        negative_ratio=("score", lambda s: (s <= 2).mean()),
        first_review=("review_date", "min"),
        last_review=("review_date", "max"),
        avg_thumbs=("thumbsUpCount", "mean") if "thumbsUpCount" in rv.columns else ("review_text", "size"),
    ).reset_index()
    feats["score_std"] = feats["score_std"].fillna(0)
    feats["avg_score"] = feats["avg_score"].fillna(feats["avg_score"].median())

    max_date = rv["review_date"].max()
    min_date = rv["review_date"].min()
    window_days = max((max_date - min_date).days, 1)
    # Fractional days: live windows are short, integer days would collapse the distribution
    feats["days_since_last_review"] = ((max_date - feats["last_review"]).dt.total_seconds() / 86400).round(2)
    feats["tenure_days"] = ((feats["last_review"] - feats["first_review"]).dt.total_seconds() / 86400).round(2)
    # review frequency: average gap between consecutive reviews (behavioral, not label-defining)
    feats["avg_gap_days"] = np.where(
        feats["total_reviews"] > 1,
        feats["tenure_days"] / (feats["total_reviews"] - 1),
        np.nan,
    )
    med_gap = feats["avg_gap_days"].median()
    feats["avg_gap_days"] = feats["avg_gap_days"].fillna(med_gap if pd.notna(med_gap) else 0)

    thresh = derive_churn_threshold(feats["days_since_last_review"])
    feats["is_churned_proxy"] = (feats["days_since_last_review"] >= thresh).astype(int)
    return feats, thresh, max_date, window_days


# Leakage-safe feature sets for churn models
TIME_BASED_FEATURES = ["total_reviews", "avg_score", "score_std", "negative_ratio",
                       "tenure_days", "avg_gap_days", "avg_thumbs"]
LEGACY_FEATURES = ["avg_votes", "avg_cost", "online_order", "book_table",
                   "neg_sent_ratio", "pos_sent_ratio", "avg_sent_comp", "review_count"]


def pick_churn_features(df):
    """Select available features; NEVER includes label-defining columns."""
    time_avail = [f for f in TIME_BASED_FEATURES if f in df.columns]
    if len(time_avail) >= 3:
        return time_avail
    legacy = [f for f in LEGACY_FEATURES if f in df.columns]
    return legacy
