import json

import numpy as np
import pandas as pd
import pytest

from src.html_export import (
    ROW_SCHEMA, TEMPLATE_FILE, MARK_ALL, MARK_SRC,
    build_rows, fill_html, build_all, write_filled_html, ensure_static_assets,
)
from src.churn_prediction import export_model_report


def _fake_reviews(n=40):
    rng = np.random.default_rng(3)
    dates = pd.to_datetime("2026-10-15") + pd.to_timedelta(
        rng.uniform(0, 5, n), unit="D")
    return pd.DataFrame({
        "reviewId": [f"rid{i}" for i in range(n)],
        "app": rng.choice(["Swiggy", "Zomato", "Zepto"], n),
        "reviewer": [f"u{i % 20}" for i in range(n)],
        "review_text": [
            "delivery was late again, app payment failed" if i % 3 == 0
            else ("great food and tasty quality" if i % 3 == 1
                  else "refund not received from support")
            for i in range(n)
        ],
        "score": rng.integers(1, 6, n),
        "review_date": dates,
        "thumbsUpCount": rng.integers(0, 10, n),
        "replyContent": [None if i % 4 else "Thanks for the feedback!"
                         for i in range(n)],
    })


def _fake_reviewers(n=200, seed=5):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "app": rng.choice(["Swiggy", "Zomato", "Zepto"], n),
        "reviewer": [f"u{i}" for i in range(n)],
        "total_reviews": rng.integers(1, 6, n),
        "avg_score": rng.uniform(1, 5, n).round(2),
        "score_std": rng.uniform(0, 2, n).round(2),
        "negative_ratio": rng.uniform(0, 1, n).round(2),
        "tenure_days": rng.uniform(0, 6, n).round(2),
        "avg_gap_days": rng.uniform(0, 3, n).round(2),
        "avg_thumbs": rng.uniform(0, 5, n).round(2),
        "is_churned_proxy": rng.integers(0, 2, n),
    })


# ---------- build_rows ----------

def test_build_rows_schema_and_types():
    rows = build_rows(_fake_reviews())
    assert list(rows.columns) == ROW_SCHEMA
    for c in ["has_reply", "is_weekend", "asp_delivery", "asp_food",
              "asp_refund", "asp_price", "asp_app", "churn_intent", "churn_risk"]:
        assert set(rows[c].unique()) <= {0, 1}, c
    assert rows["churn_score"].between(0.02, 0.99).all()
    assert set(rows["period"]) <= {"recent", "earlier"}
    assert set(rows["daypart"]) <= {"Night", "Morning", "Afternoon", "Evening", "Late night"}
    assert set(rows["sentiment_label"]) <= {"Positive", "Neutral", "Negative"}
    assert (rows["score"] >= 1).all() and (rows["score"] <= 5).all()


def test_build_rows_timestamps_and_reply():
    rows = build_rows(_fake_reviews())
    assert rows["at"].str.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}$").all()
    assert rows["date"].str.match(r"\d{4}-\d{2}-\d{2}$").all()
    assert rows["has_reply"].sum() == 10  # every 4th row has a reply (n=40)


def test_build_rows_period_split_at_median():
    rows = build_rows(_fake_reviews(40))
    ts = pd.to_datetime(rows["at"])
    med = ts.median()
    assert (ts[rows["period"] == "recent"] >= med).all()
    assert (ts[rows["period"] == "earlier"] < med).all()


def test_build_rows_model_scores_used_when_given():
    rv = _fake_reviews(30)
    reviewers = rv[["app", "reviewer"]].drop_duplicates().reset_index(drop=True)
    scores = pd.Series(np.linspace(0.1, 0.9, len(reviewers)), index=reviewers.index)
    rows = build_rows(rv, reviewers, scores)
    # at least one row must carry a model probability (not the heuristic floor)
    assert rows["churn_score"].max() > 0.1


def test_build_rows_empty():
    rows = build_rows(pd.DataFrame())
    assert list(rows.columns) == ROW_SCHEMA
    assert rows.empty


# ---------- fill_html ----------

def _template():
    return TEMPLATE_FILE.read_text(encoding="utf-8")


def test_fill_html_injects_all_and_src():
    tmpl = _template()
    rows = build_rows(_fake_reviews())
    model = {"models": {"Logistic Regression": {"acc": 0.5, "prec": 0.5, "rec": 0.5,
                                                "f1": 0.5, "auc": 0.5, "cvf1": 0.5,
                                                "cm": [[1, 0], [0, 1]]}},
             "odds": {}, "coefs": {}, "features": []}
    out = fill_html(tmpl, rows, model, "Live Play Store reviews · 40 reviews")
    assert MARK_ALL not in out
    assert MARK_SRC not in out
    assert "let ALL=[{" in out
    assert 'let SRC="Live Play Store reviews' in out
    # injection must not leave the sample-data bootstrap path active for filled data
    assert "if(!ALL.length){const a=await pick(" in out


def test_fill_html_escapes_script_in_content():
    tmpl = _template()
    rows = build_rows(_fake_reviews(5))
    rows.loc[0, "content"] = "evil </script><script>alert(1)</script>"
    out = fill_html(tmpl, rows, {"models": {}, "odds": {}, "coefs": {},
                                 "features": []}, "t")
    # raw close-script sequences inside the payload are escaped
    payload = out.split("let ALL=", 1)[1].split(";", 1)[0]
    assert "</script>" not in payload
    assert "<\\/script>" in payload


def test_fill_html_missing_marker_raises():
    rows = build_rows(_fake_reviews(3))
    with pytest.raises(ValueError):
        fill_html("<html>no markers</html>", rows, {}, "t")


def test_template_asset_cascade_is_safe():
    """The template must not depend on a single (possibly broken) CDN URL.

    Guards the two failure modes that produced EMPTY charts:
    - the old pinned cdnjs Plotly URL (404)
    - trusting r.ok/onload when an app server returns index.html with HTTP 200
    """
    tmpl = _template()
    assert "cdnjs.cloudflare.com/ajax/libs/plotly.js" not in tmpl  # the 404 URL
    assert "ASSETS_READY" in tmpl            # bootstrap waits for assets
    assert "looksLike" in tmpl               # content validation before inject
    assert "typeof Plotly!=='undefined'" in tmpl  # global check for CDN loads
    # data fetches are content-validated too (200-HTML fallback guard)
    assert "csvOK" in tmpl and "jsonOK" in tmpl


# ---------- export_model_report ----------

def test_export_model_report_real_metrics():
    df = _fake_reviewers(200)
    report, scores = export_model_report(df)
    assert set(report["models"]) == {"Logistic Regression", "SVM", "Random Forest"}
    for name, m in report["models"].items():
        for k in ["acc", "prec", "rec", "f1", "auc", "cvf1"]:
            assert 0.0 <= m[k] <= 1.0, (name, k)
        assert len(m["cm"]) == 2 and all(len(r) == 2 for r in m["cm"])
        assert len(m["fpr"]) == len(m["tpr"]) and len(m["fpr"]) >= 2
    # HTML renders CM cards for the first two entries -> LR and SVM
    assert list(report["models"])[0] == "Logistic Regression"
    assert list(report["models"])[1] == "SVM"
    assert report["odds"] and report["coefs"]
    assert set(report["odds"]) == set(report["coefs"])
    assert scores is not None
    assert scores.index.equals(df.index)
    assert scores.dropna().between(0, 1).all()


def test_export_model_report_single_class():
    df = _fake_reviewers(60)
    df["is_churned_proxy"] = 0
    report, scores = export_model_report(df)
    assert report["models"] == {}
    assert scores is None


# ---------- build_all + writers (integration) ----------

def test_build_all_fills_real_contents(tmp_path, monkeypatch):
    from src import html_export
    monkeypatch.setattr(html_export, "OUTPUT_DIR", tmp_path)
    data = {
        "reviews": _fake_reviews(60),
        "reviewers": _fake_reviewers(120, seed=9),
        "status": {"mode": "live", "fetched_at": "2026-10-19T10:00:00"},
        "churn_days": 3,
        "window_days": 5,
    }
    payload = build_all(data)
    assert "let ALL=[{" in payload["html"]
    assert MARK_SRC not in payload["html"]
    assert "Live Play Store reviews" in payload["src"]
    assert len(payload["rows"]) == 60
    assert payload["n_models"] in (0, 3)
    p = write_filled_html(payload["html"], tmp_path / "filled.html")
    assert p.exists() and p.stat().st_size > 50_000


def test_ensure_static_assets(tmp_path, monkeypatch):
    from src import html_export
    monkeypatch.setattr(html_export, "STATIC_DATA_DIR", tmp_path / "data")
    rows = build_rows(_fake_reviews(20))
    model = {"models": {}, "odds": {}, "coefs": {}, "features": []}
    csv_p, json_p = ensure_static_assets(rows, model)
    assert csv_p.exists() and json_p.exists()
    back = pd.read_csv(csv_p)
    assert list(back.columns) == ROW_SCHEMA
    assert json.loads(json_p.read_text(encoding="utf-8"))["models"] == {}


def test_build_all_static_fallback_has_heuristic_scores():
    data = {
        "reviews": _fake_reviews(30),
        "reviewers": pd.DataFrame(),  # no reviewer features (static mode)
        "status": {"mode": "static", "fetched_at": None},
        "churn_days": 0,
        "window_days": 0,
    }
    payload = build_all(data)
    assert payload["n_models"] == 0
    assert payload["used_model_scores"] is False
    assert "heuristic" in payload["model_note"].lower()
    assert payload["rows"]["churn_score"].between(0.02, 0.99).all()
