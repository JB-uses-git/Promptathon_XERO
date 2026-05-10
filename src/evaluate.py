"""
Model evaluation for Telecom Churn Prediction.
Computes the full evaluation suite from the PRD:
  - ROC-AUC, PR-AUC
  - Top-decile lift
  - Precision & Recall at top 10%
  - Expected Calibration Error (ECE)
  - Calibration slope
  - Brier score
Outputs a leaderboard table and saves results.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.linear_model import LogisticRegression as _LR
from train import HeuristicModel

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


# ──────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────

def top_decile_lift(y_true, y_prob):
    """Lift in the top 10% of scored records."""
    n = len(y_true)
    k = max(int(n * 0.10), 1)
    order = np.argsort(-y_prob)
    top_k_churn_rate = y_true.values[order[:k]].mean()
    base_rate = y_true.mean()
    return top_k_churn_rate / base_rate if base_rate > 0 else 0.0


def precision_at_top_k(y_true, y_prob, k_frac=0.10):
    """Precision in the top k% of scored records."""
    n = len(y_true)
    k = max(int(n * k_frac), 1)
    order = np.argsort(-y_prob)
    return y_true.values[order[:k]].mean()


def recall_at_top_k(y_true, y_prob, k_frac=0.10):
    """Recall captured in the top k% of scored records."""
    n = len(y_true)
    k = max(int(n * k_frac), 1)
    order = np.argsort(-y_prob)
    churners_captured = y_true.values[order[:k]].sum()
    total_churners = y_true.sum()
    return churners_captured / total_churners if total_churners > 0 else 0.0


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(y_true)


def calibration_slope(y_true, y_prob):
    """
    Calibration slope: regress y_true on logit(y_prob).
    Perfect calibration → slope ≈ 1.0.
    """
    eps = 1e-7
    logit = np.log((y_prob + eps) / (1 - y_prob + eps)).reshape(-1, 1)
    lr = _LR(solver="lbfgs", max_iter=500)
    lr.fit(logit, y_true)
    return lr.coef_[0][0]


# ──────────────────────────────────────────────
# Full evaluation
# ──────────────────────────────────────────────

def evaluate_model(name: str, y_true: pd.Series, y_prob: np.ndarray) -> dict:
    """Compute all PRD evaluation metrics for one model."""
    base_rate = y_true.mean()
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    lift = top_decile_lift(y_true, y_prob)
    prec10 = precision_at_top_k(y_true, y_prob, 0.10)
    rec10 = recall_at_top_k(y_true, y_prob, 0.10)
    ece = expected_calibration_error(y_true.values, y_prob)
    cal_slope = calibration_slope(y_true.values, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    metrics = {
        "model": name,
        "ROC-AUC": round(roc, 4),
        "PR-AUC": round(pr, 4),
        "Top-Decile Lift": round(lift, 4),
        "Precision@10%": round(prec10, 4),
        "Recall@10%": round(rec10, 4),
        "ECE": round(ece, 4),
        "Cal. Slope": round(cal_slope, 4),
        "Brier": round(brier, 4),
        "Base Churn Rate": round(base_rate, 4),
    }
    return metrics


def run_evaluation():
    """Load all models and splits, evaluate on the TEST set, print leaderboard."""
    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(
        os.path.join(ARTIFACTS_DIR, "splits.joblib")
    )

    model_names = ["heuristic", "logistic", "xgboost", "catboost"]
    results = []

    print("=" * 64)
    print("|            MODEL EVALUATION  -  TEST SET                    |")
    print("=" * 64 + "\n")

    for name in model_names:
        model_path = os.path.join(ARTIFACTS_DIR, f"model_{name}.joblib")
        if not os.path.exists(model_path):
            print(f"  ⚠ {name} not found, skipping.")
            continue

        model = joblib.load(model_path)

        # Get probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test).astype(float)

        metrics = evaluate_model(name, y_test, y_prob)
        results.append(metrics)

    # Build leaderboard
    leaderboard = pd.DataFrame(results).set_index("model")

    # Add PRD gate pass/fail markers
    leaderboard["ROC-AUC >=0.75"] = leaderboard["ROC-AUC"].apply(lambda x: "PASS" if x >= 0.75 else "FAIL")
    leaderboard["PR-AUC >=0.40"] = leaderboard["PR-AUC"].apply(lambda x: "PASS" if x >= 0.40 else "FAIL")
    leaderboard["Lift >=2.5"] = leaderboard["Top-Decile Lift"].apply(lambda x: "PASS" if x >= 2.5 else "FAIL")
    leaderboard["ECE <=0.05"] = leaderboard["ECE"].apply(lambda x: "PASS" if x <= 0.05 else "FAIL")

    print(leaderboard.to_string())
    print()

    # Save
    csv_path = os.path.join(ARTIFACTS_DIR, "evaluation_leaderboard.csv")
    leaderboard.to_csv(csv_path)
    print(f"[INFO] Leaderboard saved to {csv_path}")

    return leaderboard


if __name__ == "__main__":
    run_evaluation()
