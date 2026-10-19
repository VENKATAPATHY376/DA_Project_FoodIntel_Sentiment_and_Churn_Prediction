"""Fill the standalone FoodIntel AI.html dashboard with REAL project contents.

The template (FoodIntel AI.html) is a self-contained HTML/Plotly dashboard that
originally fell back to built-in SAMPLE data and hardcoded (fabricated) model
metrics. This module replaces both with genuine pipeline outputs:

- per-review rows in the template's 23-field schema, sourced from the live
  Google Play Store pipeline (VADER sentiment, real timestamps, real Play Store
  reply presence, reviewer-level churn probabilities from trained models)
- model_results: REAL metrics for Logistic Regression / SVM / Random Forest
  (accuracy, precision, recall, F1, ROC-AUC, 5-fold CV F1, confusion matrices,
  real ROC points) plus Logistic Regression odds ratios / standardised coefs

Disclosures (also shown on the page):
- asp_* theme flags and churn_intent are KEYWORD matches on review text
  (approximate themes, not manual labels)
- daypart / hour / dayofweek / is_weekend are derived from the review timestamp
- period = recent/earlier split at the dataset's MEDIAN timestamp (the live
  window is a few days, not 30 days - template wording was adjusted)
- churn_score = trained model probability per reviewer; when training is not
  possible (single class / no timestamps) a disclosed score heuristic is used
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_FILE = ROOT / "FoodIntel AI.html"
OUTPUT_DIR = ROOT / "output"
STATIC_DATA_DIR = ROOT / "static" / "data"

# Template markers injected by fill_html (must match the template exactly)
MARK_ALL = "let ALL=[],MODEL={models:{},odds:{},coefs:{}};"
MARK_SRC = "let SRC='Sample data'"

# Exact column order the template's toRows()/explorer expects
ROW_SCHEMA = [
    "reviewId", "app", "at", "date", "score", "content", "thumbsUpCount",
    "has_reply", "hour", "dayofweek", "is_weekend", "daypart", "sentiment",
    "sentiment_label", "asp_delivery", "asp_food", "asp_refund", "asp_price",
    "asp_app", "churn_intent", "churn_risk", "churn_score", "period",
]

# Keyword themes (approximate, disclosed) matching the template's theme filter
THEME_KEYWORDS = {
    "asp_delivery": ["delivery", "rider", "late", "delay", "arrived", "eta",
                     "cancel", "courier", "deliver"],
    "asp_food": ["food", "taste", "quality", "cold", "stale", "fresh", "tasty",
                 "delicious", "hygien", "rotten", "portion"],
    "asp_refund": ["refund", "money back", "support", "customer care",
                   "complaint", "no response", "replace", "redress"],
    "asp_price": ["price", "expensive", "charge", "fee", "costly", "discount",
                  "offer", "coupon"],
    "asp_app": ["app", "crash", "bug", "slow", "payment", "glitch", "hang",
                "loading", "buggy"],
}
CHURN_INTENT_KEYWORDS = [
    "uninstall", "delete", "switch", "stop using", "stop ordering", "no longer",
    "alternative", "other app", "quit", "don't use", "dont use", "never use",
    "stop buying", "not order",
]


def _daypart(hour):
    if hour < 5:
        return "Late night"
    if hour < 12:
        return "Morning"
    if hour < 17:
        return "Afternoon"
    if hour < 21:
        return "Evening"
    return "Night"


def _heuristic_churn_score(score, churn_intent):
    """Fallback risk score (disclosed) when model training is unavailable."""
    return float(np.clip(0.95 - 0.17 * float(score) + 0.20 * float(churn_intent), 0.02, 0.99))


def build_rows(reviews, reviewers=None, churn_scores=None):
    """Map pipeline reviews to the template's 23-field row schema."""
    if reviews is None or reviews.empty:
        return pd.DataFrame(columns=ROW_SCHEMA)
    rv = reviews.copy()

    rd = pd.to_datetime(rv.get("review_date"), errors="coerce") \
        if "review_date" in rv.columns else pd.Series(pd.NaT, index=rv.index)
    hour = rd.dt.hour.fillna(12).astype(int)
    dow = rd.dt.dayofweek.fillna(0).astype(int)  # Monday=0 (matches template)

    text = rv.get("review_text", pd.Series("", index=rv.index)).fillna("").astype(str)
    t = text.str.lower()
    flags = {}
    for col, kws in THEME_KEYWORDS.items():
        pat = "|".join(k.replace("'", "'") for k in kws)  # plain substring match
        flags[col] = t.str.contains(pat, na=False, regex=True).astype(int)
    churn_intent = t.str.contains("|".join(CHURN_INTENT_KEYWORDS), na=False,
                                  regex=True).astype(int)

    # sentiment (VADER from the pipeline)
    sent = rv.get("sent_comp", pd.Series(0.0, index=rv.index)).fillna(0.0)
    sent_label = rv.get("sent_label", pd.Series("Neutral", index=rv.index)) \
        .fillna("Neutral").astype(str)

    # real Play Store reply presence (normalized 'replyContent', if kept)
    if "replyContent" in rv.columns:
        rc = rv["replyContent"]
        has_reply = (rc.notna() & ~rc.astype(str).str.strip().str.lower()
                     .isin(["", "nan", "none"])).astype(int)
    else:
        has_reply = pd.Series(0, index=rv.index)

    # churn_score: trained-model probability per (app, reviewer), else heuristic
    scores = pd.Series(np.nan, index=rv.index, dtype=float)
    if churn_scores is not None and reviewers is not None and not reviewers.empty:
        risk = reviewers[["app", "reviewer"]].copy()
        risk["cs"] = pd.Series(churn_scores, index=reviewers.index).values
        risk_map = {(a, r): c for a, r, c in
                    zip(risk["app"], risk["reviewer"], risk["cs"]) if pd.notna(c)}
        if risk_map:
            scores = pd.Series(
                [risk_map.get((a, r), np.nan)
                 for a, r in zip(rv.get("app", ""), rv.get("reviewer", ""))],
                index=rv.index)
    heur = [_heuristic_churn_score(s, ci) for s, ci in
            zip(rv.get("score", 3).fillna(3), churn_intent)]
    scores = scores.fillna(pd.Series(heur, index=rv.index)).round(2)

    # period = recent/earlier at the MEDIAN timestamp of the window
    med = rd.median()
    period = np.where(rd.isna() | (rd >= med), "recent", "earlier")

    # ids / timestamps
    if "reviewId" in rv.columns:
        rid = rv["reviewId"].fillna("")
        rid = [v if str(v).strip() and str(v) != "nan" else f"{a}_{i}"
               for i, (v, a) in enumerate(zip(rid, rv.get("app", "app")))]
    else:
        rid = [f"{rv.get('app', pd.Series(['app'] * len(rv))).iloc[i]}_{i}"
               for i in range(len(rv))]

    out = pd.DataFrame({
        "reviewId": rid,
        "app": rv.get("app", "Unknown").fillna("Unknown").astype(str),
        "at": rd.dt.strftime("%Y-%m-%d %H:%M").fillna(""),
        "date": rd.dt.strftime("%Y-%m-%d").fillna(""),
        "score": rv.get("score", 3).fillna(3).astype(int),
        "content": text.values,
        "thumbsUpCount": rv.get("thumbsUpCount", 0).fillna(0).astype(int),
        "has_reply": has_reply.values,
        "hour": hour.values,
        "dayofweek": dow.values,
        "is_weekend": (dow > 4).astype(int).values,
        "daypart": [_daypart(h) for h in hour],
        "sentiment": sent.round(2).values,
        "sentiment_label": sent_label.values,
        "asp_delivery": flags["asp_delivery"].values,
        "asp_food": flags["asp_food"].values,
        "asp_refund": flags["asp_refund"].values,
        "asp_price": flags["asp_price"].values,
        "asp_app": flags["asp_app"].values,
        "churn_intent": churn_intent.values,
        "churn_risk": (scores >= 0.5).astype(int).values,
        "churn_score": scores.values,
        "period": period,
    })
    return out[ROW_SCHEMA]


def fill_html(template_text, rows_df, model_json, src_label):
    """Inject real rows + model results into the HTML template.

    Raises ValueError if template markers are missing (template changed).
    """
    if MARK_ALL not in template_text:
        raise ValueError(f"template marker not found: {MARK_ALL!r}")
    if MARK_SRC not in template_text:
        raise ValueError(f"template marker not found: {MARK_SRC!r}")
    # to_json handles numpy dtypes; "</" escaped so review text cannot close the <script>
    rows_payload = rows_df.to_json(orient="records", force_ascii=False) \
        .replace("</", "<\\/")
    model_payload = json.dumps(model_json, ensure_ascii=False).replace("</", "<\\/")
    out = template_text.replace(MARK_ALL, f"let ALL={rows_payload},MODEL={model_payload};")
    out = out.replace(MARK_SRC, f"let SRC={json.dumps(src_label, ensure_ascii=False)}")
    return out


def build_all(data):
    """One-call bundle for the dashboard page.

    data = run_pipeline() dict. Returns dict with:
      rows (DataFrame), model_json (dict), html (filled str), src (str),
      model_note (str), n_models (int), used_model_scores (bool)
    """
    from src.churn_prediction import export_model_report

    reviews, reviewers = data.get("reviews"), data.get("reviewers")
    status = data.get("status", {})

    report, scores = ({"models": {}, "odds": {}, "coefs": {}, "features": []}, None)
    if reviewers is not None and not reviewers.empty and "is_churned_proxy" in reviewers.columns:
        try:
            report, scores = export_model_report(reviewers)
        except Exception:
            report, scores = ({"models": {}, "odds": {}, "coefs": {}, "features": []}, None)

    rows = build_rows(reviews, reviewers, scores)
    n_models = len(report.get("models", {}))
    used_model_scores = scores is not None

    if status.get("mode") == "live":
        src = (f"Live Play Store reviews · {len(rows)} reviews · "
               f"window {data.get('window_days', '?')}d · fetched {status.get('fetched_at', '?')}")
        mode_note = "Source: public Google Play Store reviews (Swiggy/Zomato/Zepto) via google-play-scraper."
    else:
        src = f"Static Zomato dataset (fallback) · {len(rows)} reviews"
        mode_note = "Source: static Zomato Bangalore CSV fallback (no timestamps)."

    if n_models:
        model_note = (f"Model metrics: REAL training run ({n_models} classifiers, "
                      "stratified 80/20 + 5-fold CV). Theme flags are keyword matches. "
                      "churn_score = best model's predicted churn probability per reviewer.")
    else:
        model_note = ("Model training unavailable for this window - churn_score uses a "
                      "disclosed heuristic (low rating + churn-intent keywords). "
                      "Theme flags are keyword matches.")

    return {
        "rows": rows,
        "model_json": report,
        "html": fill_html(TEMPLATE_FILE.read_text(encoding="utf-8"), rows, report, src),
        "src": src,
        "mode_note": mode_note,
        "model_note": model_note,
        "n_models": n_models,
        "used_model_scores": used_model_scores,
    }


def write_filled_html(html, path=None):
    """Write the self-contained filled HTML; returns the path.

    Also mirrors static/vendor/ into <output>/vendor/ so opening the file
    from disk works fully offline (the template's asset cascade tries
    ./vendor/... first).
    """
    import shutil
    path = Path(path) if path else OUTPUT_DIR / "FoodIntel_AI_filled.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    vendor_src = ROOT / "static" / "vendor"
    if vendor_src.is_dir():
        vendor_dst = path.parent / "vendor"
        vendor_dst.mkdir(parents=True, exist_ok=True)
        for f in vendor_src.iterdir():
            if f.is_file():
                shutil.copy2(f, vendor_dst / f.name)
    return path


def ensure_static_assets(rows_df, model_json):
    """Write static/data/ so the ORIGINAL template's fetch('/data/...') finds
    real data when served by Streamlit (server.enableStaticServing, URL prefix
    /app/static/). Returns paths."""
    STATIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = STATIC_DATA_DIR / "foodintel_clean.csv"
    json_path = STATIC_DATA_DIR / "model_results.json"
    rows_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(model_json, ensure_ascii=False),
                         encoding="utf-8")
    return csv_path, json_path
