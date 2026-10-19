import streamlit as st
import pandas as pd
import plotly.express as px
from src.pipeline import run_pipeline
from src.customer_segmentation import rule_based_segment, kmeans_segment


@st.cache_data(show_spinner=False)
def load_data():
    return run_pipeline()


st.title("Reviewer Analytics")
st.caption("Reviewers from public Play Store reviews. 'Customer' unit = (platform, reviewer name).")

data = load_data()
df_cust = data["reviewers"]
churn_days = data["churn_days"]

if df_cust.empty:
    st.error("No timestamped reviewer data (static fallback). Time-based analytics unavailable.")
    st.stop()

df_cust = df_cust.copy()
df_cust["segment"] = df_cust.apply(lambda r: rule_based_segment(r, churn_days=churn_days), axis=1)

c1, c2 = st.columns(2)
with c1:
    seg = df_cust["segment"].value_counts().reset_index()
    seg.columns = ["Segment", "Count"]
    st.plotly_chart(px.bar(seg, x="Segment", y="Count", title="Rule-based Segments (time-based rules)"), width="stretch")
with c2:
    st.plotly_chart(px.histogram(df_cust, x="days_since_last_review", color="app", nbins=30,
                                 title="Recency: days since last review"), width="stretch")

c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(px.histogram(df_cust, x="total_reviews", color="app", nbins=10,
                                 barmode="group", title="Review frequency per reviewer"), width="stretch")
with c4:
    st.plotly_chart(px.histogram(df_cust, x="avg_score", color="app", nbins=5,
                                 barmode="group", title="Avg score per reviewer"), width="stretch")

st.subheader("Churn rate by platform")
cr = df_cust.groupby("app")["is_churned_proxy"].mean().reset_index()
cr["churn_rate %"] = (cr["is_churned_proxy"] * 100).round(1)
fig_cr = px.bar(cr, x="app", y="churn_rate %",
                title=f"Time-based churn proxy (inactive >= {churn_days}d)",
                text=cr["churn_rate %"].apply(lambda v: f"{v:.1f}%"))
fig_cr.update_traces(textposition="outside")
fig_cr.update_yaxes(range=[0, max(10, cr["churn_rate %"].max() * 1.2)])
st.plotly_chart(fig_cr, width="stretch")

st.subheader("K-Means Clustering (unsupervised)")
try:
    dfk, km, _ = kmeans_segment(df_cust)
    f1, f2 = st.columns(2)
    with f1:
        st.plotly_chart(px.scatter(dfk, x="days_since_last_review", y="avg_score", color="cluster",
                                   hover_data=["app", "total_reviews"],
                                   title="Clusters: recency vs score"), width="stretch")
    with f2:
        num_cols = [c for c in ["total_reviews", "avg_score", "days_since_last_review", "avg_gap_days"] if c in dfk.columns]
        centers = dfk.groupby("cluster")[num_cols].mean().reset_index()
        st.plotly_chart(px.bar(centers.melt(id_vars="cluster"), x="variable", y="value", color="cluster",
                               title="Cluster feature means"), width="stretch")
    st.caption("Clusters reflect review-engagement patterns; no business meaning assigned without inspection.")
except Exception as e:
    st.warning(f"K-Means unavailable: {e}")

st.subheader("Hierarchical Clustering Dendrogram")
try:
    from src.customer_segmentation import hierarchical_linkage, dendrogram_fig
    Z, sample, hcols = hierarchical_linkage(df_cust)
    fig = dendrogram_fig(Z)
    st.pyplot(fig, clear_figure=True)
    st.caption(
        "Ward-linkage dendrogram on scaled reviewer features "
        f"({', '.join(hcols)}; random sample of {len(sample)} for readability). "
        "Large jumps in merge height suggest natural cluster counts - used here to "
        "justify the k=3 choice for K-Means above."
    )
except Exception as e:
    st.warning(f"Dendrogram unavailable: {e}")

st.subheader("Reviewer table")
cols = [c for c in ["app", "reviewer", "total_reviews", "avg_score", "negative_ratio",
                    "days_since_last_review", "tenure_days", "is_churned_proxy", "segment"] if c in df_cust.columns]
st.dataframe(df_cust[cols].sort_values("days_since_last_review", ascending=False).head(50).round(2), width="stretch")
