import streamlit as st
from src.pipeline import run_pipeline

st.set_page_config(page_title="FoodIntel AI", page_icon="🍛", layout="wide")

st.title("FoodIntel AI 🍛")
st.markdown("**AI-Powered Customer Intelligence, Sentiment Analytics and Churn Prediction "
            "for Food Delivery Platforms (India) - Swiggy / Zomato / Zepto**")

st.markdown("""
Navigate the sidebar:
1. **Executive Overview** - KPIs, live data status, refresh button
2. **Review Intelligence** - sentiment, ratings, aspects, platform filters, real time-trends
3. **Reviewer Analytics** - time-based segments, recency/frequency, K-Means clusters
4. **Churn Intelligence** - TIME-BASED inactivity churn, model metrics, risk groups
5. **Statistical Analysis** - descriptive, correlations, outliers, CI, t-test/ANOVA/chi-square
6. **Reports and Export** - downloads of data, model summary, report
7. **Project README** - unit mapping, usage, limitations
8. **Platform Comparison** - Swiggy vs Zomato vs Zepto head-to-head
""")

st.subheader("Data source")
st.markdown("""
- **Live source:** public Google Play Store reviews for Swiggy / Zomato / Zepto (via `google-play-scraper`)
- **Cached:** 24h TTL in `data/raw/live/` (rate-limit friendly)
- **Fallback:** static Zomato Bangalore CSV if live fetch fails
""")
st.info("Reviews are PUBLIC app-store reviews - not internal platform order data. "
        "Churn is a time-based reviewer-inactivity PROXY, clearly labeled as such.")
