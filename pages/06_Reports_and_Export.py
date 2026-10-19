import streamlit as st
import pandas as pd
import io
from src.pipeline import run_pipeline
from src.utils import get_report_file


@st.cache_data(show_spinner=False)
def load_data():
    return run_pipeline()


def df_csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


st.title("Reports and Data Export")
data = load_data()
df_rev, df_cust, status = data["reviews"], data["reviewers"], data["status"]

st.subheader("Dataset Downloads")
c1, c2, c3 = st.columns(3)
c1.download_button("Reviews + sentiment (CSV)", df_csv_bytes(df_rev),
                   file_name="cleaned_reviews.csv", mime="text/csv")
if not df_cust.empty:
    c2.download_button("Reviewer features (CSV)", df_csv_bytes(df_cust),
                       file_name="reviewer_features.csv", mime="text/csv")
dqr = get_report_file("data_quality_report.md")
if dqr.exists():
    c3.download_button("Data quality report (MD)", dqr.read_bytes(),
                       file_name="data_quality_report.md", mime="text/markdown")

st.subheader("Static Seaborn/Matplotlib Figures (PNG)")
try:
    from src.plotting_static import generate_all_static
    figs = generate_all_static(df_rev, df_cust)
    if figs:
        c = st.columns(min(3, len(figs)))
        for i, (fname, fpath) in enumerate(sorted(figs.items())):
            with c[i % len(c)]:
                st.image(str(fpath), caption=fname, width=280)
                st.download_button(f"Download {fname}", fpath.read_bytes(),
                                   file_name=fname, mime="image/png", key=f"dl_{fname}")
        st.caption("Generated with Seaborn/Matplotlib (static artifacts for reports; "
                   "interactive charts use Plotly).")
    else:
        st.info("Static figures unavailable (need timestamped reviewer features).")
except Exception as e:
    st.warning(f"Static figure generation failed: {e}")

st.subheader("Filtered Review Export")
if not df_rev.empty:
    c1, c2 = st.columns(2)
    sel = c1.multiselect("Platforms", sorted(df_rev["app"].unique()), default=sorted(df_rev["app"].unique()))
    sentsel = c2.multiselect("Sentiments", sorted(df_rev["sent_label"].unique()),
                             default=sorted(df_rev["sent_label"].unique()))
    view = df_rev[df_rev["app"].isin(sel) & df_rev["sent_label"].isin(sentsel)]
    st.write(f"{len(view)} rows selected")
    st.download_button("Download filtered reviews (CSV)", df_csv_bytes(view),
                       file_name="filtered_reviews.csv", mime="text/csv")

st.subheader("Model Evaluation Summary")
try:
    from src.churn_prediction import train_models
    res, feats = train_models(df_cust)
    summary = res.copy()
    summary["CM"] = summary["CM"].apply(str)
    st.dataframe(summary.round(3), width="stretch")
    st.download_button("Download model summary (CSV)", df_csv_bytes(summary.round(3)),
                       file_name="model_evaluation_summary.csv", mime="text/csv")
    st.caption(f"Features: {', '.join(feats)} | Time-based churn proxy: inactive >= {data['churn_days']}d")
except Exception as e:
    st.warning(f"Could not compute model summary: {e}")

st.subheader("Analysis Report")
apps = ", ".join(sorted(df_rev["app"].unique())) if len(df_rev) else "n/a"
report = f"""# FoodIntel AI - Analysis Summary

- **Data source:** {status.get('source')} (Google Play Store public reviews)
- **Last successful update:** {status.get('fetched_at')}
- **Platforms:** {apps}
- **Reviews analyzed:** {len(df_rev)}
- **Reviewer units:** {len(df_cust)}
- **Observation window:** {data['window_days']} days
- **Time-based churn rule:** inactive >= {data['churn_days']} days before newest review (proxy, ~P80 recency)
- **Positive sentiment %:** {round((df_rev['sent_label']=='Positive').mean()*100,1) if len(df_rev) else 0}%
- **Negative sentiment %:** {round((df_rev['sent_label']=='Negative').mean()*100,1) if len(df_rev) else 0}%

## Limitations
- Reviews are public Google Play Store app reviews - not order/transaction data from the platforms.
- Churn is reviewer INACTIVITY on a short live window - a proxy, not subscription churn.
- Sentiment from VADER (rule-based); aspects by keywords; English-like filter via ASCII heuristic.
"""
st.download_button("Download analysis report (MD)", report.encode("utf-8"),
                   file_name="analysis_report.md", mime="text/markdown")
st.markdown(report)
