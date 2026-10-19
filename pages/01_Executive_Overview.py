import streamlit as st
from src.pipeline import run_pipeline


@st.cache_data(show_spinner=False)
def load_data(refresh_key=0, per_app=1500):
    return run_pipeline(per_app=per_app)


st.title("Executive Overview")

# Refresh control (Section 11: refresh button + last successful update time)
c1, c2, c3 = st.columns([2, 2, 6])
refreshed = c1.button("🔄 Refresh live data", help="Force a new fetch from Google Play Store (public reviews)")
per_app = c2.number_input("Reviews per app", min_value=200, max_value=5000, value=1500, step=500)

if refreshed:
    st.cache_data.clear()

with st.spinner("Loading reviews (live fetch on first run / after refresh)..."):
    try:
        data = load_data(refresh_key=1 if refreshed else 0, per_app=per_app)
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

df_rev, df_cust = data["reviews"], data["reviewers"]
status = data["status"]

# ---- Data freshness banner ----
src = status.get("source", "?")
fetched = status.get("fetched_at", "n/a")
mode_note = {
    "live": "✅ Live fetch from Google Play Store (public app reviews)",
    "cached": "🕐 Cached live data (TTL 24h) - refresh to fetch newer",
    "stale_cache": "⚠️ Live fetch failed - using last saved file",
    "zomato_static": "📁 Static Zomato Bangalore dataset (no live fetch, no timestamps)",
}.get(src, src)
st.info(f"**Data source:** {mode_note}  |  **Last successful update:** {fetched}")
if status.get("errors"):
    with st.expander("Data source log"):
        for e in status["errors"]:
            st.write("-", e)

# ---- KPIs ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total reviews (English-like, scored)", len(df_rev))
apps = sorted(df_rev["app"].unique()) if len(df_rev) else []
c2.metric("Platforms", ", ".join(apps) if apps else "n/a")
if len(df_rev):
    c3.metric("Avg score", f"{df_rev['score'].mean():.2f}")
    pos = (df_rev["sent_label"] == "Positive").mean() * 100
    neg = (df_rev["sent_label"] == "Negative").mean() * 100
else:
    c3.metric("Avg score", "n/a"); pos = neg = 0.0
c4.metric("Sentiment (+ / -)", f"{pos:.0f}% / {neg:.0f}%")

c5, c6, c7 = st.columns(3)
if not df_cust.empty:
    c5.metric("Reviewer units", len(df_cust))
    c6.metric("Time-based churn rate (proxy)", f"{df_cust['is_churned_proxy'].mean()*100:.1f}%")
    c7.metric("Inactivity rule", f">= {data['churn_days']}d since last review")
else:
    c5.metric("Reviewer units", "n/a (no timestamps)")
    c6.metric("Time-based churn", "n/a")
    c7.metric("Observation window", f"{data['window_days']} days")

st.caption(
    "Honesty notes: reviews are PUBLIC Google Play Store app reviews (not Swiggy/Zomato internal "
    "customer or order data). Churn is a time-based INACTIVITY proxy among reviewers, derived from "
    "real review timestamps relative to the observation window - not verified subscription churn."
)

st.subheader("Reviews by platform")
if len(df_rev):
    import plotly.express as px
    cnt = df_rev["app"].value_counts().reset_index()
    cnt.columns = ["app", "reviews"]
    st.plotly_chart(px.bar(cnt, x="app", y="reviews", title="Scored reviews per platform"), width="stretch")

    st.subheader("Sample reviews")
    cols = [c for c in ["app", "reviewer", "score", "review_date", "review_text", "sent_label"] if c in df_rev.columns]
    st.dataframe(df_rev[cols].head(20), width="stretch")
