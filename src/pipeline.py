"""Pipeline orchestrator.

Primary path: LIVE Google Play Store reviews (Swiggy/Zomato/Zepto) via
src.live_fetch -> normalized schema -> VADER sentiment + aspects ->
TIME-BASED reviewer features with inactivity churn proxy.

Fallback path: static Zomato Bangalore CSV (existing dataset) normalized
into the same schema; time-based churn unavailable there (no timestamps),
which is disclosed via status["mode"].
"""
import pandas as pd

from src.data_ingestion import load_raw, merge_data, find_dataset
from src.preprocessing import (preprocess, is_english_like, extract_reviews,
                               normalize_playstore, build_reviewer_features)
from src.sentiment_analysis import classify_sentiment, extract_aspects
from src.utils import get_proc_file, get_report_file

SENTIMENT_CAP = 4000


def _score_sentiment(reviews_df, cap=SENTIMENT_CAP):
    """Attach VADER sentiment + aspects to English-like reviews (capped)."""
    if reviews_df.empty:
        return reviews_df
    df = reviews_df.copy()
    mask = df["review_text"].apply(is_english_like)
    eng = df[mask]
    if len(eng) < 30:
        eng = df[df["review_text"].str.len() > 25]
    eng = eng.head(cap).copy()
    sents = eng["review_text"].apply(classify_sentiment)
    eng["sent_comp"] = sents.apply(lambda x: x["compound"])
    eng["sent_label"] = sents.apply(lambda x: x["label"])
    aspects = eng["review_text"].apply(extract_aspects)
    eng["aspects"] = aspects.apply(lambda d: ",".join(d.keys()))
    return eng


def _zomato_fallback():
    """Normalize the static Zomato Bangalore CSV into the common schema."""
    data, found = load_raw()
    merged = merge_data(data)
    merged_p = preprocess(merged, max_reviews=1200)
    rows = []
    for idx, row in merged_p.iterrows():
        for rt in extract_reviews(row.get("review_text", ""), limit=3):
            rows.append({
                "app": "Zomato (static)",
                "reviewer": f"zomato_row_{idx}",
                "review_text": rt,
                "score": row.get("rating", None),
                "review_date": pd.NaT,
                "thumbsUpCount": row.get("votes", None),
            })
    rev = pd.DataFrame(rows)
    status = {
        "is_live": False,
        "source": "zomato_static",
        "mode": "static",
        "fetched_at": None,
        "saved_file": str(found) if found else None,
        "errors": ["live fetch unavailable; using static Zomato Bangalore dataset "
                   "(no timestamps -> no time-based churn)"],
    }
    return rev, status


def run_pipeline(force_refresh=False, per_app=1500):
    """Run the full analysis pipeline. Returns dict with reviews/reviewers/status."""
    from src.live_fetch import get_live_reviews
    try:
        raw, status = get_live_reviews(force_refresh=force_refresh, per_app=per_app)
        rev = normalize_playstore(raw)
        status["mode"] = "live"
        if rev.empty:
            raise ValueError("normalized live reviews are empty")
    except Exception as e:
        status = {"is_live": False, "source": "zomato_static", "mode": "static",
                  "fetched_at": None, "saved_file": None,
                  "errors": [f"live path failed: {type(e).__name__}: {e}"]}
        if find_dataset() is not None:
            rev, status = _zomato_fallback()
        else:
            raise RuntimeError(f"Live fetch failed ({e}) and no fallback dataset in data/raw/.")

    reviews_sent = _score_sentiment(rev)

    # Time-based reviewer features (only when real timestamps exist)
    reviewers = pd.DataFrame()
    churn_days, max_date, window_days = 0, None, 0
    if "review_date" in rev.columns and rev["review_date"].notna().any():
        reviewers, churn_days, max_date, window_days = build_reviewer_features(rev)
        if not reviews_sent.empty:
            agg = (reviews_sent.groupby(["app", "reviewer"])
                   .agg(neg_sent_ratio=("sent_label", lambda s: (s == "Negative").mean()),
                        pos_sent_ratio=("sent_label", lambda s: (s == "Positive").mean()),
                        avg_sent_comp=("sent_comp", "mean"))
                   .reset_index())
            reviewers = reviewers.merge(agg, on=["app", "reviewer"], how="left")
            for c in ["neg_sent_ratio", "pos_sent_ratio", "avg_sent_comp"]:
                reviewers[c] = reviewers[c].fillna(0)

    reviews_sent.to_csv(get_proc_file("cleaned_reviews.csv"), index=False)
    if not reviewers.empty:
        reviewers.to_csv(get_proc_file("customer_features.csv"), index=False)

    with open(get_report_file("data_quality_report.md"), "w", encoding="utf-8") as f:
        f.write("# Data Quality Report - FoodIntel AI (Live Play Store Reviews)\n\n")
        f.write(f"- **Mode:** {status.get('mode')}\n")
        f.write(f"- **Source:** {status.get('source')} ({status.get('saved_file')})\n")
        f.write(f"- **Fetched at:** {status.get('fetched_at')}\n")
        f.write(f"- **Reviews normalized:** {len(rev)}\n")
        f.write(f"- **Reviews with sentiment (English-like):** {len(reviews_sent)}\n")
        if not reviewers.empty:
            f.write(f"- **Reviewer units:** {len(reviewers)}\n")
            f.write(f"- **Observation window:** {window_days} days "
                    f"(max date {max_date.date() if max_date is not None else 'n/a'})\n")
            f.write(f"- **Time-based churn rule:** no review for >= {churn_days} days "
                    f"before max date (proxy)\n")
        f.write("- **Platform(s):** " + ", ".join(sorted(rev['app'].unique())) + "\n")

    return {
        "reviews": reviews_sent,
        "reviewers": reviewers,
        "status": status,
        "churn_days": churn_days,
        "max_date": max_date,
        "window_days": window_days,
    }


def run_full_pipeline(max_reviews=1200, max_sentiment=800):
    """Legacy entrypoint (kept for compatibility with older tests/pages)."""
    data = run_pipeline()
    return data["reviews"], data["reviewers"], data["churn_days"]
