import streamlit as st
import pandas as pd
import plotly.express as px
from src.pipeline import run_pipeline


@st.cache_data(show_spinner=False)
def load_data():
    return run_pipeline()


st.title("Review Intelligence")
data = load_data()
df_rev = data["reviews"]

if df_rev.empty:
    st.warning("No scored reviews available.")
    st.stop()

# ---- Filters ----
st.subheader("Filters")
f1, f2, f3, f4 = st.columns(4)
apps = sorted(df_rev["app"].unique())
sel_apps = f1.multiselect("Platform(s)", apps, default=apps)
sents = ["All"] + sorted(df_rev["sent_label"].unique())
sel_sent = f2.selectbox("Sentiment", sents)
if df_rev["review_date"].notna().any():
    dmin = df_rev["review_date"].min().date()
    dmax = df_rev["review_date"].max().date()
    d1, d2 = f3.slider("Date range", dmin, dmax, (dmin, dmax))
    date_filter = (pd.Timestamp(d1), pd.Timestamp(d2) + pd.Timedelta(days=1))
else:
    f3.caption("No timestamps (static data)")
    date_filter = None
q = f4.text_input("Search text")

view = df_rev.copy()
if sel_apps:
    view = view[view["app"].isin(sel_apps)]
else:
    view = view.iloc[0:0]
if sel_sent != "All":
    view = view[view["sent_label"] == sel_sent]
if date_filter is not None and "review_date" in view:
    view = view[view["review_date"].between(*date_filter)]
if q:
    view = view[view["review_text"].str.contains(q, case=False, na=False)]

st.caption(f"{len(view)} of {len(df_rev)} reviews match filters")

# ---- Charts ----
c1, c2 = st.columns(2)
with c1:
    sent = view["sent_label"].value_counts().reset_index()
    sent.columns = ["Sentiment", "Count"]
    st.plotly_chart(px.pie(sent, names="Sentiment", values="Count", title="Sentiment Distribution"), width="stretch")
with c2:
    st.plotly_chart(px.histogram(view, x="score", nbins=5, color="app", barmode="group",
                                 title="Rating Distribution by Platform"), width="stretch")

st.subheader("Sentiment over time (real review timestamps)")
if view["review_date"].notna().any():
    t = view.copy()
    span_days = (t["review_date"].max() - t["review_date"].min()).total_seconds() / 86400
    if span_days < 2:
        t["period"] = t["review_date"].dt.floor("h")
        gran = "hour"
    else:
        t["period"] = t["review_date"].dt.date
        gran = "day"
    trend = t.groupby(["period", "app", "sent_label"]).size().reset_index(name="count")
    st.plotly_chart(px.line(trend, x="period", y="count", color="app", line_dash="sent_label",
                            title=f"Reviews per {gran} by platform & sentiment"), width="stretch")
else:
    st.info("Timestamps unavailable in static fallback mode.")

st.subheader("Sentiment vs Rating")
if len(view):
    ct = view.groupby("score")["sent_label"].value_counts().unstack(fill_value=0)
    st.plotly_chart(px.bar(ct, barmode="stack", title="Sentiment by rating score"), width="stretch")

st.subheader("Aspect-based sentiment")
asp_rows = []
for _, r in view.iterrows():
    for a in str(r.get("aspects", "")).split(","):
        a = a.strip()
        if a:
            asp_rows.append({"aspect": a, "sent_label": r["sent_label"], "app": r["app"]})
if asp_rows:
    adf = pd.DataFrame(asp_rows)
    c1, c2 = st.columns(2)
    with c1:
        at = adf.groupby(["aspect", "sent_label"]).size().reset_index(name="count")
        st.plotly_chart(px.bar(at, x="aspect", y="count", color="sent_label", barmode="stack",
                               title="Aspect mentions by sentiment"), width="stretch")
    with c2:
        aa = adf.groupby(["aspect", "app"]).size().reset_index(name="count")
        st.plotly_chart(px.bar(aa, x="aspect", y="count", color="app", barmode="group",
                               title="Aspect mentions by platform"), width="stretch")
    st.caption("Aspects detected by keyword matching (rule-based; may miss synonyms).")
else:
    st.info("No aspects detected in filtered reviews.")

st.subheader("Reviews (filtered table)")
cols = [c for c in ["app", "reviewer", "score", "review_date", "sent_label", "aspects", "review_text"] if c in view.columns]
st.dataframe(view[cols].head(200), width="stretch")
