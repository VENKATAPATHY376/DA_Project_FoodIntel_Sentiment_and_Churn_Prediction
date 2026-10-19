from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib
from src.utils import get_model_file
from src.preprocessing import pick_churn_features


def make_models(probability=True):
    """Single source of truth for the three churn classifiers.

    Shared by train_models (dashboard metrics) and export_model_report
    (HTML/CSV exports) so exported numbers always match the app.

    probability only toggles libsvm's Platt calibration for SVM - it does NOT
    change predicted labels (identical accuracy/precision/recall/F1/confusion
    matrices); exports set it False to skip libsvm's very slow internal CV,
    and score ROC/AUC from the decision_function instead (Platt scaling is a
    monotone sigmoid of that value, so the ROC/AUC are unchanged).
    """
    return {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "SVM": Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(probability=probability, random_state=42, class_weight="balanced"))]),
    }


def train_models(df, target="is_churned_proxy", features=None):
    if features is None:
        # Leakage-safe selection: never includes the label-defining column
        features = pick_churn_features(df)
    if not features or len(features) == 0:
        raise ValueError("No usable prediction-time features found in dataframe.")
    df = df.dropna(subset=features + [target])
    if df[target].nunique() < 2:
        raise ValueError("Target has a single class; cannot train classifiers.")
    X = df[features]
    y = df[target]
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=stratify, random_state=42)
    models = make_models()
    results = []
    best = None
    best_score = -1
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        try:
            roc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
        except ValueError:
            roc = 0.0  # single-class test fold
        res = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc,
            "CM": confusion_matrix(y_test, y_pred).tolist()
        }
        results.append(res)
        if res["F1"] > best_score:
            best_score = res["F1"]
            best = pipe
    if best is not None:
        joblib.dump(best, get_model_file("churn_model.joblib"))
        joblib.dump(features, get_model_file("churn_features.joblib"))
    return pd.DataFrame(results), features


# Display names for export charts (real feature -> readable label)
FEATURE_DISPLAY = {
    "total_reviews": "Total reviews",
    "avg_score": "Avg rating",
    "score_std": "Rating variability",
    "negative_ratio": "Low-rating share (<=2)",
    "tenure_days": "Tenure (days)",
    "avg_gap_days": "Avg gap between reviews",
    "avg_thumbs": "Avg thumbs up",
    "neg_sent_ratio": "Negative sentiment share",
    "pos_sent_ratio": "Positive sentiment share",
    "avg_sent_comp": "Avg VADER compound",
    "avg_votes": "Avg votes",
    "avg_cost": "Avg cost",
    "online_order": "Online ordering",
    "book_table": "Table booking",
    "review_count": "Review count",
}


def export_model_report(df, target="is_churned_proxy", features=None):
    """Full, REAL model report for the embedded HTML dashboard / model_results.json.

    Returns (report, churn_scores):
      report = {
        "models": {name: {acc, prec, rec, f1, auc, cvf1, cm, fpr, tpr}},
        "odds":   {display_name: odds_ratio},   # Logistic Regression, per 1 SD (standardised)
        "coefs":  {display_name: std_coef},
        "features": [...],
      }
      churn_scores: pd.Series aligned to df.index with the churn probability from
      the best-F1 model refit on all usable rows (None if training impossible).

    Metrics use the SAME stratified 80/20 split as train_models (random_state=42);
    cvf1 = stratified K-fold CV F1 on the full data (folds reduced when the
    minority class is small). Everything is computed from real training runs -
    no hardcoded numbers.
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import roc_curve
    import numpy as np

    if features is None:
        features = pick_churn_features(df)
    report = {"models": {}, "odds": {}, "coefs": {}, "features": list(features)}
    if not features or df.empty:
        return report, None
    clean = df.dropna(subset=features + [target])
    if clean.empty or clean[target].nunique() < 2:
        return report, None  # caller falls back to a disclosed heuristic

    X, y = clean[features], clean[target]
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=stratify, random_state=42)

    # HTML renders confusion-matrix cards for the first two entries -> LR, SVM first
    for name in ["Logistic Regression", "SVM", "Random Forest"]:
        # probability=False: identical predicted labels, skips libsvm's slow Platt CV
        pipe = make_models(probability=False)[name]
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        # ROC/AUC signal: predict_proba when available, else decision_function
        # (Platt calibration is a monotone sigmoid of it -> identical ROC/AUC)
        score_test = None
        try:
            if hasattr(pipe, "predict_proba"):
                score_test = pipe.predict_proba(X_test)[:, 1]
            elif hasattr(pipe, "decision_function"):
                score_test = pipe.decision_function(X_test)
        except Exception:
            score_test = None
        try:
            auc = roc_auc_score(y_test, score_test) if score_test is not None else 0.0
        except ValueError:
            auc = 0.0
        fpr, tpr = [], []
        if score_test is not None:
            try:
                fpr_l, tpr_l, _ = roc_curve(y_test, score_test)
                fpr, tpr = [round(float(v), 4) for v in fpr_l], [round(float(v), 4) for v in tpr_l]
            except ValueError:
                pass
        f1_val = f1_score(y_test, y_pred, zero_division=0)
        n_min = int(y.value_counts().min())
        folds = max(2, min(5, n_min))
        try:
            cvf1 = float(cross_val_score(
                make_models(probability=False)[name], X, y,
                cv=StratifiedKFold(folds, shuffle=True, random_state=42),
                scoring="f1", n_jobs=-1).mean())
        except Exception:
            cvf1 = float(f1_val)
        report["models"][name] = {
            "acc": round(float(accuracy_score(y_test, y_pred)), 4),
            "prec": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "rec": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_val), 4),
            "auc": round(float(auc), 4),
            "cvf1": round(cvf1, 4),
            "cm": confusion_matrix(y_test, y_pred).tolist(),
            "fpr": fpr, "tpr": tpr,
        }

    # Logistic Regression standardised coefficients + odds ratios (per 1 SD)
    try:
        lr = make_models()["Logistic Regression"]
        lr.fit(X, y)
        coefs = lr.named_steps["clf"].coef_[0]
        for fname, c in zip(features, coefs):
            disp = FEATURE_DISPLAY.get(fname, fname.replace("_", " ").title())
            report["coefs"][disp] = round(float(c), 4)
            report["odds"][disp] = round(float(np.exp(c)), 4)
    except Exception:
        pass

    # Churn probability for every reviewer: best-F1 model refit on all usable rows
    # (probability=True here: churn_score must be a real calibrated probability)
    churn_scores = None
    try:
        best_name = max(report["models"], key=lambda n: report["models"][n]["f1"])
        scorer = make_models(probability=True)[best_name]
        scorer.fit(X, y)
        churn_scores = pd.Series(
            scorer.predict_proba(X)[:, 1], index=clean.index).reindex(df.index)
        report["best_model"] = best_name
    except Exception:
        pass
    return report, churn_scores
