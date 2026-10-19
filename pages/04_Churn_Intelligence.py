import streamlit as st
import pandas as pd
import plotly.express as px
from src.pipeline import run_pipeline
from src.churn_prediction import train_models
from src.preprocessing import pick_churn_features
from sklearn.ensemble import RandomForestClassifier


@st.cache_data(show_spinner=False)
def load_data():
    return run_pipeline()


st.title("Churn Intelligence")
data = load_data()
df_cust = data["reviewers"]
churn_days = data["churn_days"]
status = data["status"]

if status.get("mode") == "static":
    st.warning("Static fallback - no timestamps, time-based churn unavailable.")
    st.stop()

st.warning(
    f"**Time-based churn proxy:** a reviewer is marked churned if their last review is "
    f"**>= {churn_days} days** before the newest review in the dataset (threshold = 80th percentile "
    f"of recency, derived from data). This measures REVIEWER inactivity, not subscription churn. "
    f"Observation window: {data['window_days']} days."
)

if df_cust.empty or "is_churned_proxy" not in df_cust:
    st.error("No reviewer features.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Churn rate (proxy)", f"{df_cust['is_churned_proxy'].mean()*100:.1f}%")
c2.metric("Churned units", int(df_cust["is_churned_proxy"].sum()))
c3.metric("Total units", len(df_cust))

class_counts = df_cust["is_churned_proxy"].value_counts()
if len(class_counts) < 2:
    st.warning(
        "⚠️ **Insufficient observation window for churn modelling.** The current dataset window "
        f"({data['window_days']} day(s)) is too short to separate active vs inactive reviewers "
        "(only one class present). Press **Refresh live data** on the Executive Overview with a "
        "larger 'Reviews per app' value to widen the window."
    )
    st.stop()

ld = df_cust["is_churned_proxy"].value_counts().reset_index()
ld.columns = ["is_churned_proxy (proxy)", "count"]
st.plotly_chart(px.bar(ld, x="is_churned_proxy (proxy)", y="count", title="Proxy Label Distribution"), width="stretch")

feats = pick_churn_features(df_cust)
st.caption(f"Model features (leakage-safe; `days_since_last_review` defines the label so it is EXCLUDED): {', '.join(feats)}")

try:
    res, feats = train_models(df_cust, features=feats)
    st.subheader("Model Comparison")
    st.dataframe(res[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].round(3), width="stretch")
    st.caption(
        "Models trained: **Logistic Regression** (StandardScaler + LR, max_iter=1000), "
        "**Random Forest** (100 trees), **SVM** (RBF kernel, probability=True, scaled). "
        "All use class_weight=balanced. Stratified 80/20 split. Reviewer churn is hard to predict from "
        "behavior alone - expect moderate ROC-AUC; interpret as risk ranking, not certainty.")

    best = res.loc[res["F1"].idxmax()]
    st.subheader(f"Confusion Matrix - {best['Model']} (best F1)")
    cm = pd.DataFrame(best["CM"], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    st.dataframe(cm, width="content")

    st.plotly_chart(px.bar(res, x="Model", y="ROC-AUC", title="ROC-AUC comparison", range_y=[0, 1]), width="stretch")

    st.subheader("Feature Importance (Random Forest)")
    X = df_cust[feats].fillna(0)
    y = df_cust["is_churned_proxy"]
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X, y)
    imp = pd.DataFrame({"feature": feats, "importance": rf.feature_importances_}).sort_values("importance")
    st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                           title="Feature Importances"), width="stretch")

    st.subheader("Predicted Risk Groups")
    proba = rf.predict_proba(X)[:, 1]
    risk = df_cust.copy()
    risk["churn_risk"] = proba
    risk["risk_group"] = pd.cut(risk["churn_risk"], [0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])
    rg = risk["risk_group"].value_counts().reset_index()
    rg.columns = ["Risk group", "count"]
    st.plotly_chart(px.bar(rg, x="Risk group", y="count", title="Predicted Risk Groups"), width="stretch")

    st.plotly_chart(px.violin(risk, x="risk_group", y="churn_risk", color="risk_group",
                              box=True, points=False,
                              title="Churn-risk score distribution by predicted risk group (violin)"),
                    width="stretch")
    st.caption("Violin width shows density of predicted risk scores within each group.")

    st.subheader("Reviewer-level Risk Table")
    show = [c for c in ["app", "reviewer", "total_reviews", "avg_score", "negative_ratio",
                        "days_since_last_review", "is_churned_proxy", "churn_risk", "risk_group"] if c in risk.columns]
    st.dataframe(risk[show].sort_values("churn_risk", ascending=False).head(50).round(3), width="stretch")
except Exception as e:
    st.error(f"Model training failed: {e}")
