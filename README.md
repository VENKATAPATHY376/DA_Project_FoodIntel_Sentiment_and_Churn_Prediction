# FoodIntel AI 🍛

**AI-Powered Customer Intelligence, Sentiment Analytics and Churn Prediction Platform for Food Delivery Applications (India)**

Academic Data Analytics project (AIM23DLU01) built on **live public Google Play Store reviews** for Swiggy, Zomato & Zepto, with a static Zomato Bangalore CSV as offline fallback.

## Table of Contents
- [1. Dataset & Limitations](#1-dataset--limitations)
- [2. Setup and Run](#2-setup-and-run)
- [3. Tests](#3-tests)
- [4. Architecture](#4-architecture)
- [5. Dashboard Pages](#5-dashboard-pages)
- [6. Detailed Mapping to the 5 Units](#6-detailed-mapping-to-the-5-units)
- [7. Methodology Notes](#7-methodology-notes)
- [8. Deployment & Disclaimers](#8-deployment--disclaimers)

## 1. Data Sources & Limitations

### Primary: LIVE Google Play Store reviews (Swiggy / Zomato / Zepto)

- **Source:** public app-store reviews via `google-play-scraper` (no auth, no access-control bypass)
- **Retrieved:** fetched at app runtime; cached with 24h TTL in `data/raw/live/`
- **Refresh:** "Refresh live data" button on Executive Overview triggers a new fetch (respects rate limits via paging + sleep, caches result, records last-success timestamp)
- **Platforms:** `in.swiggy.android`, `com.application.zomato`, `com.zeptoconsumerapp`

| Required Field | Available | Handling |
|---|---|---|
| Customer identity | Reviewer name + platform | Unit = `(app, reviewer)` (disclosed; not a real account ID) |
| Review text + ratings | Yes (`content`, `score`) | NLP (VADER), sentiment, aspects, EDA |
| Review timestamps | Yes (`at`) | Enables **time-based churn** and real time-trends |
| Order value / history | No | Not present in app-store reviews |
| Churn label | No | **Time-based proxy:** reviewer inactive ≥ threshold before newest review (derived from data, disclosed everywhere) |

### Fallback: static Zomato Bangalore CSV
If live fetch fails (offline / network blocked), the app transparently falls back to the Kaggle Zomato Bangalore CSV (`data/raw/zomato-bangalore-restaurants.csv`). Timestamps are absent there, so time-based churn is marked unavailable and a dissatisfaction proxy is disclosed instead.

### Honesty notes
- Reviews are **public Play Store app reviews**, NOT internal Swiggy/Zomato/Zepto order or customer databases.
- Churn is a **reviewer-inactivity proxy**, not verified subscription churn.
- No authentication or access controls are bypassed; only publicly visible reviews are read.

## 2. Setup and Run

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

No dataset download needed — the app fetches **live public Play Store reviews** on first run (needs internet). A cached copy is stored under `data/raw/live/` (24h TTL). Optional offline fallback: place `zomato-bangalore-restaurants.csv` in `data/raw/`.

App: http://localhost:8501

## 3. Tests

```bat
python -m pytest -v
```

**41/41 passed** (structure, preprocessing, sentiment, churn, pipeline, time-based churn, dendrogram, static plots, robustness guards, HTML export, asset-cascade guard)

## 4. Architecture

```
FoodIntel-AI/
├── app.py                      # Home page
├── FoodIntel AI.html            # standalone HTML dashboard template (filled with real data by src/html_export.py)
├── pages/                      # 9 Streamlit pages
│   ├── 01_Executive_Overview.py    # KPIs, live status, refresh button
│   ├── 02_Review_Intelligence.py   # sentiment, ratings, aspects, filters, time-trends
│   ├── 03_Customer_Analytics.py    # reviewer segments, recency/frequency, clusters
│   ├── 04_Churn_Intelligence.py    # TIME-BASED churn, model metrics, risk groups
│   ├── 05_Statistical_Analysis.py  # descriptive, corr, outliers, CI, t/ANOVA/chi2
│   ├── 06_Reports_and_Export.py    # downloads (data, model summary, report)
│   ├── 07_Project_README.py        # renders this README
│   ├── 08_Platform_Comparison.py   # Swiggy vs Zomato vs Zepto
│   └── 09_FoodIntel_HTML.py        # embeds the filled HTML dashboard (st.iframe)
├── src/
│   ├── live_fetch.py           # Play Store fetch + 24h cache + fallback chain
│   ├── data_ingestion.py       # auto-detect CSV, load (fallback dataset)
│   ├── preprocessing.py        # cleaning, normalization, features, churn labels
│   ├── sentiment_analysis.py   # VADER + aspect keywords
│   ├── customer_segmentation.py# rule-based + K-Means
│   ├── churn_prediction.py     # LogReg / RF / SVM, leakage-safe features
│   ├── statistical_analysis.py # CI, t-test, ANOVA, chi-square
│   ├── eda.py                  # descriptive, IQR/Z-score outliers
│   ├── visualization.py        # Plotly helpers
│   ├── plotting_static.py      # Seaborn/Matplotlib static PNGs (heatmap, pairplot, violin, countplot)
│   ├── html_export.py          # fills FoodIntel AI.html with real rows + real model metrics
│   ├── pipeline.py             # orchestrator (live → cache → static)
│   └── utils.py                # paths
├── data/
│   ├── raw/live/               # cached live fetches (TTL 24h)
│   └── processed/              # cleaned outputs
├── static/data/                # foodintel_clean.csv + model_results.json (served for the original HTML template)
├── static/vendor/              # self-hosted plotly.min.js + tailwind.js (HTML dashboard asset cascade)
├── output/                     # self-contained FoodIntel_AI_filled.html (generated)
├── .streamlit/config.toml      # server.enableStaticServing = true
├── models/, reports/, tests/
├── requirements.txt
└── run_local.bat
```

## 5. Dashboard Pages

1. Executive Overview — KPIs, live/cached data status, **refresh button**, review counts, sentiment %, churn rate
2. Review Intelligence — sentiment dist, rating dist, sentiment-by-rating, aspects, real time-trends, filters (platform/sentiment/date/text)
3. Reviewer Analytics — time-based segments, recency/frequency distributions, K-Means clusters, **hierarchical clustering dendrogram**
4. Churn Intelligence — **time-based inactivity proxy**, model comparison (LogReg/RF/**SVM**), confusion matrix, ROC-AUC, RF importance, risk groups + **violin plot**, risk table
5. Statistical Analysis — descriptive, correlation heatmap, IQR/Z, 95% CI, t-test/ANOVA/chi-square with H0/H1, p, interpretation
6. Reports and Export — CSV/MD downloads + **Seaborn static figures** (heatmap, pair plot, violin, countplot) as PNGs
7. Project README — this file rendered in-app
8. Platform Comparison — Swiggy vs Zomato vs Zepto head-to-head KPIs, **violin + box plots**, ANOVA/t-test
9. FoodIntel HTML Dashboard — the standalone `FoodIntel AI.html` embedded via `st.iframe`, **filled with this project's real contents**: 23-field review rows (live timestamps, VADER sentiment, keyword theme flags, reviewer churn probabilities) + **REAL model metrics** (LogReg/SVM/RF: acc/precision/recall/F1/ROC-AUC/5-fold CV F1/confusion matrices/real ROC points, LR odds ratios & standardised coefficients). Downloads the self-contained HTML, `foodintel_clean.csv` and `model_results.json`; also writes them to `static/data/` so the original template (served at `/app/static/data/…`) loads real data instead of its sample fallback. The template's fabricated hardcoded metrics (ROC-AUC ~0.96) were removed and its "previous 30 days" wording corrected to "earlier period" (live window is a few days). Its originally pinned cdnjs Plotly URL **404'd (every chart was blank)** - replaced by an asset cascade that loads **self-hosted `static/vendor/` copies first** (`/app/static/vendor/…` in-app, `./vendor/…` from disk), then jsdelivr/unpkg CDN fallbacks, with content validation (an app server's SPA fallback returns index.html with HTTP 200 for missing paths - trusting that status was the second blank-chart failure mode) and a visible error banner if all sources fail; verified with a headless-browser check (charts render, zero page errors).

## 6. Detailed Mapping to the 5 Units

### Unit I — Data Analysis Fundamentals

| Topic | Where Used | Description |
|---|---|---|
| NumPy arrays & Pandas Series/DataFrames | `src/*.py`, `pages/*.py` | All ops via pandas; numpy for percentiles/std/outliers/scaling |
| Data cleaning & preprocessing | `src/preprocessing.py` | Text cleaning, rating extraction, NA handling, duplicates |
| Handling missing values & duplicates | `src/preprocessing.py`, `src/eda.py` | Drop/flag; missing % in report |
| Label encoding / One-hot encoding | `src/preprocessing.py`, models | Encode sentiment labels; binary churn target; legacy `online_order`/`book_table` in fallback path |
| Normalization/Standardization & Feature Engineering | `src/preprocessing.py`, `src/churn_prediction.py`, `src/customer_segmentation.py` | Engineer real features; StandardScaler in LR/SVM/K-Means |
| Reading & exporting CSV files | `src/live_fetch.py`, `src/data_ingestion.py`, `pages/06_Reports_and_Export.py` | Live fetch → CSV cache; auto-detect fallback CSV; export cleaned/filtered/model summaries |

### Unit II — Exploratory Data Analysis (EDA)

| Topic | Where Used | Description |
|---|---|---|
| Mean, median, mode & descriptive stats | `src/statistical_analysis.py`, Page 5 | `.describe()` (mean/median/std/min/max/count) |
| Univariate/bivariate/multivariate | Pages 2–5 | Hist/pie (uni), sentiment-by-rating/votes (bi), corr/clusters (multi) |
| Correlation & correlation matrices | `src/statistical_analysis.corr`, Page 5 | Pearson on numeric features (heatmap + strong pairs) |
| Histograms, scatter plots, pair plots & heatmaps | `src/visualization.py`, `src/plotting_static.py`, Pages 2–5, 6 | Plotly heatmap (Page 5) + **Seaborn heatmap & pair plot** (static PNGs, Page 6) |
| Outlier detection (IQR & Z-score) | `src/eda.py`, Page 5 | Both methods; flagged, not auto-removed |

### Unit III — Data Visualization and Dashboards

| Topic | Where Used | Description |
|---|---|---|
| Line/bar/pie/histogram | Pages 1–9 (Plotly) | Used for storytelling/insights; Page 9 embeds the same Plotly charts inside the standalone HTML dashboard |
| Box plots & heatmaps | Pages 5, 6, 8 | Plotly correlation heatmap; **Seaborn heatmap** PNG (Page 6); boxplot on Page 8 |
| Violin plots | Pages 4, 8 | `px.violin` - risk-score density by risk group (Page 4); score distribution by platform (Page 8); plus **Seaborn violin** PNG on Page 6 |
| Matplotlib/Seaborn | `src/plotting_static.py`, Page 6 | Seaborn heatmap, pair plot, violin, countplot generated as PNGs for download (interactive dashboard stays Plotly) |
| Interactive charts using Plotly | All pages | Hover/zoom enabled |
| Dendrograms (hierarchical clustering) | `src/customer_segmentation.py`, Page 3 | scipy Ward linkage + truncated dendrogram next to K-Means; merge-height jumps justify k choice |
| Dashboard design & data storytelling | `app.py` + pages + `FoodIntel AI.html` | Multi-page, filters, KPIs, disclosures, exports; Page 9 delivers the single-file HTML dashboard (Plotly + JS) filled with real pipeline data via `src/html_export.py` |

### Unit IV — Inferential Statistics and Hypothesis Testing

| Topic | Where Used | Description |
|---|---|---|
| Sample mean & confidence intervals | `src/statistical_analysis.ci_mean`, Page 5 | 95% t-interval; CLT note |
| Z-test & T-test | Page 5 (Welch t-test) | Welch t-test comparing Swiggy vs Zomato reviewer avg_score (no pop std → t correct; Z not used) |
| One-way ANOVA | Page 5, Page 8 | F, p across the 3 platforms (n>=3 per group) |
| P-values & statistical significance | Page 5 | alpha=0.05; reject/fail to reject stated |
| Comparing groups & interpreting results | Page 5 | H0/H1, n, method, stat, p, plain English; notes power/association≠causation |

### Unit V — Machine Learning and Model Evaluation

| Topic | Where Used | Description |
|---|---|---|
| Linear Regression | **Not used** | No defensible real continuous target; must not use for binary churn (justified) |
| Logistic Regression | `src/churn_prediction.py`, Page 4 | Binary on time-based inactivity proxy; Pipeline+scaler; class_weight balanced |
| Support Vector Machine (SVM) | `src/churn_prediction.py`, Page 4 | **Actively used** as churn classifier. RBF kernel, `probability=True`, scaled, `class_weight=balanced`; shown in model-comparison table + ROC-AUC chart + model caption |
| Training and testing datasets | `src/churn_prediction.py` | Stratified 80/20 (random_state=42) |
| MAE, MSE, RMSE, R² score and accuracy | `src/churn_prediction.py` (classification eval) | Accuracy/Precision/Recall/F1/ROC-AUC reported; regression metrics not computed (no LR for regression / binary task) |
| Comparing models & selecting the best model | Page 4 | F1-based selection; ROC-AUC, CM, RF importance shown |

## 7. Methodology Notes

- **Time-based churn:** a reviewer is marked churned if inactive ≥ threshold before the newest review in the dataset. Threshold **derived from data** (80th percentile of recency). Observation window & threshold both shown in the UI.
- **Leakage guard:** `days_since_last_review` (defines the label) is excluded from model features; `pick_churn_features()` enforces this and a test asserts it.
- **Live data & rate limits:** Play Store fetch pages reviews (200/request) with sleep between pages; result cached with 24h TTL; refresh button forces re-fetch; last-success timestamp + source (live/cached/stale/static) surfaced on Overview.
- **Fallback:** live → cached → stale-cache → static Zomato CSV; the active source is always disclosed.
- **Class imbalance:** `class_weight="balanced"`.
- **NLP:** VADER; ASCII heuristic for English-like (fast); ratings ≠ ground truth.
- **Aspects:** keyword-based (rule-limited) — disclosed.

## 8. Deployment & Disclaimers

- **Local execution** (Windows). Streamlit Cloud: keep requirements, set `app.py`; live fetch needs internet at runtime (cache warms automatically).
- **Data ethics:** only publicly visible Play Store reviews are read; no auth/access-control bypass; 24h cache respects rate limits.
- **Estimates only:** proxy labels → model estimates, not future certainty.
- **Statistics:** significance reported; correlation ≠ causation.
- **Freshness:** live by default; refresh button re-fetches and updates timestamp/charts.
