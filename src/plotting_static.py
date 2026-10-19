"""Static Matplotlib/Seaborn figures (Unit III requirement).

Plotly drives the interactive dashboard; these Seaborn figures are generated
as PNGs in reports/ for download on the Reports page (data storytelling +
static report artifacts).
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.utils import REPORTS

warnings.filterwarnings("ignore", category=FutureWarning)


def _save(fig, name):
    path = REPORTS / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def seaborn_corr_heatmap(df, name="seaborn_corr_heatmap.png"):
    """Seaborn correlation heatmap of numeric reviewer features."""
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Seaborn)")
    return _save(fig, name)


def seaborn_pairplot(df, name="seaborn_pairplot.png", max_n=400):
    """Seaborn pairplot on a sample, colored by churn proxy label."""
    cols = [c for c in ["total_reviews", "avg_score", "days_since_last_review",
                        "avg_gap_days", "is_churned_proxy"] if c in df.columns]
    if len(cols) < 3 or "is_churned_proxy" not in cols:
        return None
    sample = df[cols].dropna().sample(min(max_n, len(df)), random_state=42)
    g = sns.pairplot(sample, hue="is_churned_proxy", palette="Set1", diag_kind="hist")
    g.figure.suptitle("Pair Plot by Churn Proxy (sample)", y=1.02)
    return _save(g.figure, name)


def seaborn_violin(df, x="app", y="score", name="seaborn_violin_score.png"):
    """Seaborn violin plot of score distribution per platform."""
    sub = df.dropna(subset=[y])
    if sub.empty or x not in sub.columns:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=sub, x=x, y=y, hue=x, palette="muted", legend=False,
                   ax=ax, inner="quart")
    ax.set_title(f"{y} distribution by {x} (Seaborn violin)")
    return _save(fig, name)


def seaborn_score_dist(df, name="seaborn_score_dist.png"):
    """Seaborn countplot of rating scores."""
    sub = df.dropna(subset=["score"])
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=sub, x="score", hue="app", ax=ax)
    ax.set_title("Review score counts by platform (Seaborn)")
    return _save(fig, name)


def generate_all_static(df_reviews, df_reviewers):
    """Generate every static figure; returns dict name->path (skips failures)."""
    out = {}
    try:
        if df_reviewers is not None and not df_reviewers.empty:
            p = seaborn_corr_heatmap(df_reviewers)
            if p:
                out[p.name] = p
            p = seaborn_pairplot(df_reviewers)
            if p:
                out[p.name] = p
        if df_reviews is not None and not df_reviews.empty:
            p = seaborn_violin(df_reviews)
            if p:
                out[p.name] = p
            p = seaborn_score_dist(df_reviews)
            if p:
                out[p.name] = p
    except Exception:
        pass
    return out
