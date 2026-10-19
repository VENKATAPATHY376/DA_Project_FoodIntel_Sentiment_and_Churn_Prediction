import streamlit as st

from src.pipeline import run_pipeline
from src.html_export import build_all, write_filled_html, ensure_static_assets

st.set_page_config(page_title="FoodIntel HTML Dashboard", layout="wide")
st.title("FoodIntel AI - Embedded HTML Dashboard")
st.caption(
    "The standalone `FoodIntel AI.html` dashboard, embedded in the app and filled with "
    "**this project's real pipeline contents** (live Play Store reviews, VADER sentiment, "
    "trained model metrics) - no sample data, no hardcoded numbers."
)


@st.cache_data(show_spinner=False)
def _load(per_app):
    return run_pipeline(per_app=per_app)


@st.cache_data(show_spinner=False)
def _build(data):
    return build_all(data)


data = _load(1500)

with st.spinner("Preparing embedded HTML dashboard (trains models for real metrics)..."):
    try:
        payload = _build(data)
    except FileNotFoundError:
        st.error("Template `FoodIntel AI.html` not found in the project root.")
        st.stop()
    except ValueError as e:
        st.error(f"HTML template injection failed: {e}")
        st.stop()

# Persist artifacts (self-contained copy + fetchable assets for the original template)
try:
    filled_path = write_filled_html(payload["html"])
    csv_path, json_path = ensure_static_assets(payload["rows"], payload["model_json"])
except OSError as e:
    filled_path = csv_path = json_path = None
    st.warning(f"Could not write export files: {e}")

status = data.get("status", {})
c1, c2, c3 = st.columns(3)
c1.metric("Reviews embedded", f"{len(payload['rows']):,}")
c2.metric("Models with real metrics", f"{payload['n_models']} / 3")
c3.metric("Data source", "Live" if status.get("mode") == "live" else "Static")

st.caption(f"{payload['mode_note']} {payload['model_note']}")

# The dashboard itself (st.iframe = non-deprecated replacement for components.html)
st.iframe(payload["html"], height=1000)

st.divider()
st.subheader("Downloads")
d1, d2, d3 = st.columns(3)
with d1:
    if filled_path:
        st.download_button(
            "Download filled HTML (self-contained)",
            filled_path.read_bytes(), file_name="FoodIntel_AI_filled.html",
            mime="text/html", width="stretch",
        )
with d2:
    if csv_path and csv_path.exists():
        st.download_button(
            "Download foodintel_clean.csv",
            csv_path.read_bytes(), file_name="foodintel_clean.csv",
            mime="text/csv", width="stretch",
        )
with d3:
    if json_path and json_path.exists():
        st.download_button(
            "Download model_results.json",
            json_path.read_bytes(), file_name="model_results.json",
            mime="application/json", width="stretch",
        )

with st.expander("Data & methodology disclosures for this HTML dashboard"):
    st.markdown(
        """
- **Data:** public Google Play Store reviews for Swiggy / Zomato / Zepto via `google-play-scraper`
  (app-store listing reviews, not internal platform data). Static Zomato CSV is the offline fallback.
- **`sentiment` / `sentiment_label`:** VADER compound score on English-like reviews.
- **`asp_*` theme flags & `churn_intent`:** approximate KEYWORD matches on review text (not manual labels).
- **`hour` / `dayofweek` / `is_weekend` / `daypart`:** derived from the real review timestamp.
- **`period`:** recent vs earlier split at the dataset's MEDIAN timestamp (live window is a few days - the
  template's former "30 days" wording was corrected to "earlier period").
- **`churn_score` / `churn_risk`:** churn probability from the best trained classifier (LogReg / SVM / RF,
  leakage-safe features); if training is impossible for the window, a disclosed heuristic is used instead.
- **Model metrics (accuracy / precision / recall / F1 / ROC-AUC / CV F1 / confusion matrices / ROC points):
  REAL training runs** from `src/churn_prediction.py` - the template's former hardcoded numbers
  (ROC-AUC ~0.96) were removed. Expect honest, moderate values (~0.56 ROC-AUC): churn is hard to predict
  from review behavior alone; treat scores as risk ranking, not certainty.
- **Correlation does not imply causation** - associations shown here are descriptive.
- The dashboard loads **Plotly and Tailwind from self-hosted `vendor/` files first**
  (inside the app: `/app/static/vendor/…`; next to the downloaded HTML: `./vendor/…`),
  with jsdelivr/unpkg as CDN fallbacks - if every source fails, a visible error banner
  appears instead of blank charts.
- `static/data/*.csv|json` are written so the original `FoodIntel AI.html` served from this app also
  loads real data instead of its sample fallback.
"""
    )
