"""
WTTE Survival Model Evaluation
Compares the WTTE model against the existing CatBoost/XGBoost models
using identical metrics: ROC-AUC, PR-AUC, Top-decile Lift, ECE, Brier, etc.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression as _LR
import tensorflow as tf
import keras
import keras.ops as ops
from keras.layers import Layer

# ── Register custom objects for Keras loading ──
@keras.saving.register_keras_serializable(package="Custom")
class WeibullOutputLayer(Layer):
    def __init__(self, init_alpha=36.0, max_beta_value=4.0, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = init_alpha; self.max_beta_value = max_beta_value
    def call(self, x):
        a = x[..., 0]; b = x[..., 1]
        a = ops.clip(a, -3.0, 3.0); a = self.init_alpha * ops.exp(a)
        shift = float(np.log(self.max_beta_value - 1.0))
        b = self.max_beta_value * ops.sigmoid(b - shift)
        return ops.stack([a, b], axis=-1)
    def get_config(self):
        c = super().get_config()
        c.update({"init_alpha": self.init_alpha, "max_beta_value": self.max_beta_value})
        return c

@keras.saving.register_keras_serializable(package="Custom")
def wtte_loss(y_true, y_pred):
    y = tf.cast(y_true[..., 0], tf.float32); u = tf.cast(y_true[..., 1], tf.float32)
    a = tf.cast(y_pred[..., 0], tf.float32); b = tf.cast(y_pred[..., 1], tf.float32)
    eps = 1e-6; a = tf.maximum(a, eps); b = tf.maximum(b, eps); y = tf.maximum(y, eps)
    ya = y / a; log_ya = tf.math.log(ya + eps)
    survival = tf.clip_by_value(-tf.pow(ya, b), -50.0, 0.0)
    hazard = tf.clip_by_value(tf.math.log(b / a + eps) + (b - 1.0) * log_ya, -50.0, 50.0)
    return -tf.reduce_mean(u * hazard + survival)


ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
WTTE_DIR = os.path.join(ARTIFACTS_DIR, "wtte_data")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots", "wtte")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Metric helpers (same as src/evaluate.py) ──

def top_decile_lift(y_true, y_prob):
    n = len(y_true)
    k = max(int(n * 0.10), 1)
    order = np.argsort(-y_prob)
    top_k_rate = y_true[order[:k]].mean()
    base_rate = y_true.mean()
    return top_k_rate / base_rate if base_rate > 0 else 0.0

def precision_at_top_k(y_true, y_prob, k_frac=0.10):
    n = len(y_true); k = max(int(n * k_frac), 1)
    order = np.argsort(-y_prob)
    return y_true[order[:k]].mean()

def recall_at_top_k(y_true, y_prob, k_frac=0.10):
    n = len(y_true); k = max(int(n * k_frac), 1)
    order = np.argsort(-y_prob)
    return y_true[order[:k]].sum() / max(y_true.sum(), 1)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0: continue
        ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece / len(y_true)

def calibration_slope(y_true, y_prob):
    eps = 1e-7
    logit = np.log((y_prob + eps) / (1 - y_prob + eps)).reshape(-1, 1)
    lr = _LR(solver="lbfgs", max_iter=500)
    lr.fit(logit, y_true)
    return lr.coef_[0][0]


def evaluate_model(name, y_true, y_prob):
    return {
        "model": name,
        "ROC-AUC": round(roc_auc_score(y_true, y_prob), 4),
        "PR-AUC": round(average_precision_score(y_true, y_prob), 4),
        "Top-Decile Lift": round(top_decile_lift(y_true, y_prob), 4),
        "Precision@10%": round(precision_at_top_k(y_true, y_prob), 4),
        "Recall@10%": round(recall_at_top_k(y_true, y_prob), 4),
        "ECE": round(expected_calibration_error(y_true, y_prob), 4),
        "Cal. Slope": round(calibration_slope(y_true, y_prob), 4),
        "Brier": round(brier_score_loss(y_true, y_prob), 4),
        "Base Churn Rate": round(y_true.mean(), 4),
    }


# ── Plotting ──

def plot_roc_curves(models_data, save_path):
    """ROC curve comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_true, y_prob, color in models_data:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

def plot_pr_curves(models_data, save_path):
    """Precision-Recall curve comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_true, y_prob, color in models_data:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, linewidth=2)
    base = models_data[0][1].mean()
    ax.axhline(y=base, color='gray', linestyle='--', alpha=0.5, label=f'Baseline ({base:.2f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

def plot_calibration(models_data, save_path, n_bins=10):
    """Calibration plot comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    for name, y_true, y_prob, color in models_data:
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_means, bin_true = [], []
        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_means.append(y_prob[mask].mean())
                bin_true.append(y_true[mask].mean())
        ax.plot(bin_means, bin_true, 'o-', label=name, color=color, linewidth=2, markersize=6)
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Observed Frequency', fontsize=12)
    ax.set_title('Calibration Plot', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

def plot_risk_distribution(alphas, betas, y_true, save_path):
    """WTTE-specific: Alpha distribution split by churned/not-churned."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Alpha distribution
    ax = axes[0]
    ax.hist(alphas[y_true == 0], bins=50, alpha=0.6, label='Active', color='#2196F3', density=True)
    ax.hist(alphas[y_true == 1], bins=50, alpha=0.6, label='Churned', color='#f44336', density=True)
    ax.set_xlabel('Alpha (Expected Months)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Alpha Distribution by Churn Status', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(0, min(200, np.percentile(alphas, 99)))
    ax.grid(True, alpha=0.3)
    
    # Risk score distribution at 12 months
    ax = axes[1]
    risk_12 = 1.0 - np.exp(-np.power(12.0 / alphas, betas))
    ax.hist(risk_12[y_true == 0], bins=50, alpha=0.6, label='Active', color='#2196F3', density=True)
    ax.hist(risk_12[y_true == 1], bins=50, alpha=0.6, label='Churned', color='#f44336', density=True)
    ax.set_xlabel('12-Month Churn Risk', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('12-Month Risk Distribution by Churn Status', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ──

def main():
    print("=" * 70)
    print("  WTTE SURVIVAL MODEL — FULL EVALUATION SUITE")
    print("=" * 70)
    
    # 1. Load WTTE model and data
    print("\n[1/4] Loading models and data...")
    wtte_model = keras.models.load_model(os.path.join(WTTE_DIR, "wtte_model.keras"))
    x_flat = np.load(os.path.join(WTTE_DIR, "x_flat.npy"))
    y_tte = np.load(os.path.join(WTTE_DIR, "y_tte.npy"))
    y_events = np.load(os.path.join(WTTE_DIR, "y_events.npy"))
    
    # Binary ground truth for standard metrics
    y_true = y_events.astype(int)
    
    # WTTE predictions
    preds = wtte_model.predict(x_flat, verbose=0)
    alphas = preds[:, 0]
    betas = preds[:, 1]
    
    # Convert Weibull params to 12-month churn probability for apples-to-apples comparison
    t = 12
    wtte_probs = 1.0 - np.exp(-np.power(t / alphas, betas))
    wtte_probs = np.clip(wtte_probs, 0.0, 1.0)
    
    print(f"  WTTE predictions: {len(wtte_probs)} customers")
    print(f"  Churn rate: {y_true.mean()*100:.1f}%")
    
    # 2. Load old models for comparison (they used different test split)
    print("\n[2/4] Loading CatBoost/XGBoost for comparison...")
    splits_path = os.path.join(ARTIFACTS_DIR, "splits.joblib")
    
    results = []
    models_data = []  # for plotting
    colors = {'catboost': '#1976D2', 'xgboost': '#388E3C', 'WTTE-Survival': '#E53935'}
    
    if os.path.exists(splits_path):
        X_train, X_val, X_test, y_train, y_val, y_test_old = joblib.load(splits_path)
        
        for name in ['catboost', 'xgboost']:
            model_path = os.path.join(ARTIFACTS_DIR, f"model_{name}.joblib")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                if hasattr(model, "predict_proba"):
                    old_probs = model.predict_proba(X_test)[:, 1]
                else:
                    old_probs = model.predict(X_test).astype(float)
                
                y_test_np = y_test_old.values if hasattr(y_test_old, 'values') else y_test_old
                metrics = evaluate_model(name, y_test_np, old_probs)
                results.append(metrics)
                models_data.append((name, y_test_np, old_probs, colors[name]))
                print(f"  Loaded {name}: ROC-AUC={metrics['ROC-AUC']}")
    
    # 3. Evaluate WTTE
    print("\n[3/4] Evaluating WTTE Survival Model...")
    wtte_metrics = evaluate_model("WTTE-Survival", y_true, wtte_probs)
    results.append(wtte_metrics)
    models_data.append(("WTTE-Survival", y_true, wtte_probs, colors['WTTE-Survival']))
    
    # Build leaderboard
    leaderboard = pd.DataFrame(results).set_index("model")
    leaderboard["ROC-AUC >=0.75"] = leaderboard["ROC-AUC"].apply(lambda x: "PASS" if x >= 0.75 else "FAIL")
    leaderboard["PR-AUC >=0.40"] = leaderboard["PR-AUC"].apply(lambda x: "PASS" if x >= 0.40 else "FAIL")
    leaderboard["Lift >=2.5"] = leaderboard["Top-Decile Lift"].apply(lambda x: "PASS" if x >= 2.5 else "FAIL")
    leaderboard["ECE <=0.05"] = leaderboard["ECE"].apply(lambda x: "PASS" if x <= 0.05 else "FAIL")
    
    print("\n" + "=" * 70)
    print("  LEADERBOARD")
    print("=" * 70)
    print(leaderboard.to_string())
    
    # Save leaderboard
    csv_path = os.path.join(ARTIFACTS_DIR, "evaluation_leaderboard_wtte.csv")
    leaderboard.to_csv(csv_path)
    print(f"\n  Leaderboard saved to {csv_path}")
    
    # 4. Generate plots
    print("\n[4/4] Generating evaluation plots...")
    
    # Note: ROC/PR/Calibration comparisons can only fairly compare models
    # evaluated on the SAME test set. The old models use splits.joblib test set,
    # WTTE uses the full dataset. So we plot WTTE standalone + old models standalone.
    
    # WTTE standalone plots
    wtte_only = [("WTTE-Survival", y_true, wtte_probs, '#E53935')]
    plot_roc_curves(wtte_only, os.path.join(PLOTS_DIR, "roc_wtte.png"))
    plot_pr_curves(wtte_only, os.path.join(PLOTS_DIR, "pr_wtte.png"))
    plot_calibration(wtte_only, os.path.join(PLOTS_DIR, "calibration_wtte.png"))
    plot_risk_distribution(alphas, betas, y_true, os.path.join(PLOTS_DIR, "risk_distribution_wtte.png"))
    
    # Combined comparison (note: different test sets, for visual reference only)
    if len(models_data) > 1:
        plot_roc_curves(models_data, os.path.join(PLOTS_DIR, "roc_comparison.png"))
        plot_pr_curves(models_data, os.path.join(PLOTS_DIR, "pr_comparison.png"))
        plot_calibration(models_data, os.path.join(PLOTS_DIR, "calibration_comparison.png"))
    
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
