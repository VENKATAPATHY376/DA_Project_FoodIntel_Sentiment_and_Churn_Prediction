import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def rule_based_segment(row, churn_days=7, votes_median=200):
    """Rule-based segments.

    TIME-BASED schema (Play Store reviewers, preferred):
    - At-risk:          inactive >= churn_days (time-based churn proxy)
    - Recent:           active within half the churn threshold
    - Frequent:         >= 2 reviews and short avg gap
    - One-off:          1 review only
    - Engaged:          rest

    LEGACY schema (Zomato static, no dates): rating-based rules.
    """
    # Time-based branch
    if "days_since_last_review" in row.index and pd.notna(row.get("days_since_last_review")):
        rec = row["days_since_last_review"]
        n = row.get("total_reviews", 1)
        gap = row.get("avg_gap_days", 0)
        if rec >= churn_days:
            return "At-risk (inactive)"
        if rec <= churn_days / 2:
            return "Recently active"
        if n >= 2 and gap <= 1:
            return "Frequent"
        if n == 1:
            return "One-off"
        return "Engaged"

    # Legacy branch (rating-based; no time data available)
    r = row.get("avg_rating", 3)
    v = row.get("avg_votes", 0)
    if r is None or pd.isna(r):
        return "Unclassified"
    if r <= 2.5:
        return "At-risk"
    if r < 3.0:
        return "Low-rated"
    if r < 4.0:
        return "Neutral"
    if v >= votes_median:
        return "High-value"
    return "Satisfied"


def kmeans_segment(df, cols=None, k=3):
    if cols is None:
        candidates = ["total_reviews", "avg_score", "tenure_days", "avg_gap_days", "days_since_last_review"]
        cols = [c for c in candidates if c in df.columns]
        if len(cols) < 2:
            cols = [c for c in ["avg_rating", "avg_votes", "avg_cost", "review_count"] if c in df.columns]
    X = df[cols].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    df = df.copy()
    df["cluster"] = km.fit_predict(Xs)
    return df, km, scaler


def hierarchical_linkage(df, cols=None, sample_n=300):
    """Ward linkage on scaled features for dendrogram plotting.

    Returns (linkage_matrix, sampled_scaled_df, feature_names).
    Sampled for tractability: dendrograms on thousands of points are unreadable.
    """
    from scipy.cluster.hierarchy import linkage
    if cols is None:
        candidates = ["total_reviews", "avg_score", "tenure_days", "avg_gap_days", "days_since_last_review"]
        cols = [c for c in candidates if c in df.columns]
        if len(cols) < 2:
            cols = [c for c in ["avg_rating", "avg_votes", "avg_cost", "review_count"] if c in df.columns]
    sample = df[cols].fillna(0).sample(min(sample_n, len(df)), random_state=42)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(sample)
    Z = linkage(Xs, method="ward")
    return Z, sample, cols


def dendrogram_fig(Z, max_d=None, n_clusters=3):
    """Render a scipy dendrogram and return the matplotlib figure (caller closes it)."""
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90,
               color_threshold=max_d, ax=ax)
    ax.set_title(f"Hierarchical Clustering Dendrogram (Ward, truncated) - suggested k ~ {n_clusters}")
    ax.set_xlabel("Cluster size (increasing merge distance = more dissimilar)")
    ax.set_ylabel("Ward distance")
    return fig
