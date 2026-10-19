import streamlit as st
import pandas as pd
import plotly.express as px
from src.pipeline import run_pipeline
from src.statistical_analysis import ttest_ind, anova


@st.cache_data(show_spinner=False)
def load_data():
    return run_pipeline()


st.title("Platform Comparison: Swiggy vs Zomato vs Zepto")
st.caption("All figures from live public Play Store reviews fetched by the app.")

data = load_data()
df_rev, df_cust = data["reviews"], data["reviewers"]

if df_rev.empty:
    st.error("No reviews available.")
    st.stop()

apps = sorted(df_rev["app"].unique())
if len(apps) < 2:
    st.warning("Only one platform present; comparison needs >= 2.")
    st.stop()

# ---- KPI comparison table ----
rows = []
for a in apps:
    sub = df_rev[df_rev["app"] == a]
    cust = df_cust[df_cust["app"] == a] if not df_cust.empty else pd.DataFrame()
    rows.append({
        "Platform": a,
        "Reviews": len(sub),
        "Avg score": round(sub["score"].mean(), 2) if sub["score"].notna().any() else None,
        "% Positive": round((sub["sent_label"] == "Positive").mean() * 100, 1),
        "% Negative": round((sub["sent_label"] == "Negative").mean() * 100, 1),
        "Reviewers": len(cust) if not cust.empty else None,
        "Churn rate % (proxy)": round(cust["is_churned_proxy"].mean() * 100, 1) if not cust.empty else None,
    })
kpi = pd.DataFrame(rows)
st.subheader("Head-to-head KPIs")
st.dataframe(kpi, width="stretch")

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.bar(kpi, x="Platform", y=["% Positive", "% Negative"], barmode="group",
                           title="Sentiment split by platform"), width="stretch")
with c2:
    st.plotly_chart(px.violin(df_rev, x="app", y="score", color="app", box=True, points=False,
                              title="Score distribution by platform (violin)"), width="stretch")

st.plotly_chart(px.box(df_rev, x="app", y="score", color="app",
                       title="Score distribution by platform (boxplot)"), width="stretch")

st.subheader("Review volume over time")
if df_rev["review_date"].notna().any():
    t = df_rev.copy()
    span_days = (t["review_date"].max() - t["review_date"].min()).total_seconds() / 86400
    if span_days < 2:
        t["period"] = t["review_date"].dt.floor("h")
        gran = "hour"
    else:
        t["period"] = t["review_date"].dt.date
        gran = "day"
    vol = t.groupby(["period", "app"]).size().reset_index(name="reviews")
    st.plotly_chart(px.line(vol, x="period", y="reviews", color="app",
                            title=f"Reviews per {gran} (window: {span_days:.1f} days)"),
                    width="stretch")

st.subheader("Common complaint aspects by platform")
asp_rows = []
for _, r in df_rev.iterrows():
    for a in str(r.get("aspects", "")).split(","):
        a = a.strip()
        if a and r["sent_label"] == "Negative":
            asp_rows.append({"aspect": a, "app": r["app"]})
if asp_rows:
    adf = pd.DataFrame(asp_rows).groupby(["aspect", "app"]).size().reset_index(name="count")
    st.plotly_chart(px.bar(adf, x="aspect", y="count", color="app", barmode="group",
                           title="Negative-review aspects by platform"), width="stretch")
else:
    st.info("No negative aspects detected.")

st.subheader("Statistical comparison")
if not df_cust.empty and "avg_score" in df_cust.columns and len(apps) >= 3:
    groups = [g["avg_score"].dropna() for _, g in df_cust.groupby("app") if len(g) >= 3]
    if len(groups) >= 2:
        f, p, sizes = anova(groups)
        st.write(f"One-way ANOVA on reviewer avg_score across {len(groups)} platforms: "
                 f"F = {f:.3f}, p = {p:.4f}, sizes = {sizes}")
        st.write(f"**{'Significant differences exist between platforms' if p < 0.05 else 'No significant difference'}** "
                 f"(alpha=0.05). Association only - not causal.")
elif not df_cust.empty and {"Swiggy", "Zomato"}.issubset(set(apps)):
    g1 = df_cust.loc[df_cust["app"] == "Swiggy", "avg_score"].dropna()
    g2 = df_cust.loc[df_cust["app"] == "Zomato", "avg_score"].dropna()
    res = ttest_ind(g1, g2)
    if res:
        t, p, n1, n2 = res
        st.write(f"T-test Swiggy vs Zomato: t = {t:.3f}, p = {p:.4f}, n = {n1}/{n2}")
        st.write(f"**{'Significant' if p < 0.05 else 'Not significant'}** difference (alpha=0.05).")
