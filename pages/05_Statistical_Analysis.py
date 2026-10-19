import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.pipeline import run_pipeline
from src.statistical_analysis import desc_stats, corr, ci_mean, ttest_ind, anova, chi2
from src.eda import iqr_outliers, z_outliers


@st.cache_data(show_spinner=False)
def load_data():
    return run_pipeline()


st.title("Statistical Analysis")
data = load_data()
df_rev, df_cust = data["reviews"], data["reviewers"]

if df_cust.empty:
    st.warning("No reviewer features available (static mode).")

numeric = df_cust.select_dtypes(include=[np.number]) if not df_cust.empty else pd.DataFrame()

st.subheader("1. Descriptive Statistics")
if not numeric.empty:
    st.dataframe(numeric.describe().round(3), width="stretch")
else:
    st.info("No numeric reviewer features.")

st.subheader("2. Correlation Analysis")
if numeric.shape[1] >= 2:
    c = corr(numeric)
    st.plotly_chart(px.imshow(c.round(2), text_auto=True, title="Correlation Matrix (reviewer features)",
                              color_continuous_scale="RdBu_r", zmin=-1, zmax=1), width="stretch")
    strong = c.abs().unstack()
    strong = strong[(strong > 0.5) & (strong < 1.0)].drop_duplicates().sort_values(ascending=False)
    st.caption(f"Strong correlations (|r|>0.5): {len(strong)} pairs. Correlation != causation.")

st.subheader("3. Outlier Summary (IQR / Z-score)")
if not numeric.empty:
    o_rows = []
    for col in numeric.columns[:8]:
        o_rows.append({"feature": col,
                       "IQR outliers": int(iqr_outliers(numeric[col])),
                       "Z-score outliers (|z|>3)": int(z_outliers(numeric[col])),
                       "n": int(numeric[col].notna().sum())})
    st.dataframe(pd.DataFrame(o_rows), width="stretch")

st.subheader("4. Confidence Intervals")
if not numeric.empty:
    target = st.selectbox("Feature for CI of mean", numeric.columns.tolist(), index=0)
    ci = ci_mean(numeric[target].dropna())
    if ci:
        m, lo, hi, n = ci
        st.write(f"**{target}**: mean = {m:.3f}, 95% CI = [{lo:.3f}, {hi:.3f}], n = {n}")
        st.caption("t-based interval; CLT makes this reasonable for n>30.")
    else:
        st.info("Not enough data.")

st.subheader("5. Hypothesis Testing")
tab1, tab2, tab3 = st.tabs(["T-test (Swiggy vs Zomato)", "One-way ANOVA (3 platforms)", "Chi-square"])

with tab1:
    st.markdown("**H0:** mean reviewer score equal for Swiggy & Zomato. **H1:** differ. alpha=0.05.")
    if not df_cust.empty and {"app", "avg_score"}.issubset(df_cust.columns):
        g1 = df_cust.loc[df_cust["app"] == "Swiggy", "avg_score"].dropna()
        g2 = df_cust.loc[df_cust["app"] == "Zomato", "avg_score"].dropna()
        res = ttest_ind(g1, g2)
        if res:
            t, p, n1, n2 = res
            st.write(f"t = {t:.3f}, p = {p:.4f}, n1(Swiggy) = {n1}, n2(Zomato) = {n2}")
            st.write(f"**{'Reject H0' if p < 0.05 else 'Fail to reject H0'}** - "
                     f"{'statistically significant' if p < 0.05 else 'no significant'} difference in mean "
                     f"reviewer score. Association, not causation.")
        else:
            st.info("Insufficient data in one/both groups.")

with tab2:
    st.markdown("**H0:** all platform means equal. **H1:** at least one differs. alpha=0.05.")
    if not df_cust.empty and {"app", "avg_score"}.issubset(df_cust.columns):
        groups = [g["avg_score"].dropna() for _, g in df_cust.groupby("app") if len(g) >= 3]
        if len(groups) >= 2:
            f, p, sizes = anova(groups)
            st.write(f"F = {f:.3f}, p = {p:.4f}, group sizes = {sizes}")
            st.write(f"**{'Reject H0' if p < 0.05 else 'Fail to reject H0'}** - "
                     f"{'at least one platform mean differs' if p < 0.05 else 'no significant difference across platforms'}.")
        else:
            st.info("Need >=2 platforms with n>=3.")

with tab3:
    st.markdown("**H0:** sentiment and rating group independent. **H1:** associated. alpha=0.05.")
    if not df_rev.empty and "score" in df_rev.columns:
        tmp = df_rev.dropna(subset=["score"]).copy()
        tmp["rating_group"] = pd.cut(tmp["score"], [0, 2, 3, 5], labels=["Low(<=2)", "Mid(3)", "High(>3)"])
        table = pd.crosstab(tmp["rating_group"], tmp["sent_label"])
        st.dataframe(table, width="content")
        r = chi2(table)
        if r:
            chi2v, p, dof = r
            st.write(f"chi2 = {chi2v:.3f}, dof = {dof}, p = {p:.6f}")
            st.write(f"**{'Reject H0' if p < 0.05 else 'Fail to reject H0'}** - "
                     f"{'sentiment and rating group are associated' if p < 0.05 else 'no evidence of association'}.")
        else:
            st.info("Table unsuitable for chi-square.")

st.caption("All tests computed from the loaded real dataset. Significance != causation.")
