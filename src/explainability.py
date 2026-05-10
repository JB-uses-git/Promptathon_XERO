"""
Explainability module for Telecom Churn Prediction.
Uses SHAP to produce:
  - Global feature importance (summary bar plot + beeswarm)
  - Local explanations (waterfall plots for sample high-risk & low-risk users)
All plots are saved as PNG files in the artifacts directory.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")


def run_explainability():
    """Generate SHAP explanations for the best tree-based model (XGBoost)."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load data and model
    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(
        os.path.join(ARTIFACTS_DIR, "splits.joblib")
    )

    model_path = os.path.join(ARTIFACTS_DIR, "model_xgboost.joblib")
    if not os.path.exists(model_path):
        print("[ERROR] XGBoost model not found. Run train.py first.")
        return

    model = joblib.load(model_path)
    feature_names = list(X_test.columns)

    # Unwrap CalibratedClassifierCV to get the base tree model for SHAP
    from sklearn.calibration import CalibratedClassifierCV
    if isinstance(model, CalibratedClassifierCV):
        base_model = model.calibrated_classifiers_[0].estimator
        print(f"[INFO] Unwrapped calibration wrapper -> {type(base_model).__name__}")
    else:
        base_model = model

    print("[INFO] Computing SHAP values (this may take a minute)...")

    # Use a sample for speed on large datasets
    sample_size = min(5000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    y_sample = y_test.loc[X_sample.index]

    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X_sample)

    # ── 1. Global: Summary bar plot ──
    print("[INFO] Generating global feature importance bar plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False,
                      feature_names=feature_names)
    plt.title("SHAP Global Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_global_bar.png"), dpi=150)
    plt.close()

    # ── 2. Global: Beeswarm plot ──
    print("[INFO] Generating beeswarm plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False,
                      feature_names=feature_names)
    plt.title("SHAP Beeswarm – Feature Impact on Churn Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_beeswarm.png"), dpi=150)
    plt.close()

    # ── 3. Local: Waterfall for highest-risk user ──
    print("[INFO] Generating local waterfall plots...")
    y_prob = model.predict_proba(X_sample)[:, 1]
    high_risk_idx = np.argmax(y_prob)
    low_risk_idx = np.argmin(y_prob)

    # Create Explanation objects for waterfall
    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_sample.values,
        feature_names=feature_names,
    )

    # High-risk waterfall
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation[high_risk_idx], show=False)
    plt.title(f"High-Risk User (P(churn)={y_prob[high_risk_idx]:.3f})",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_waterfall_high_risk.png"), dpi=150)
    plt.close()

    # Low-risk waterfall
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation[low_risk_idx], show=False)
    plt.title(f"Low-Risk User (P(churn)={y_prob[low_risk_idx]:.3f})",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_waterfall_low_risk.png"), dpi=150)
    plt.close()

    # ── 4. Feature importance table (saved as CSV) ──
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(ARTIFACTS_DIR, "shap_feature_importance.csv")
    importance_df.to_csv(csv_path, index=False)

    print(f"\n[INFO] SHAP Feature Importance (top 10):")
    print(importance_df.head(10).to_string(index=False))
    print(f"\n[INFO] All plots saved to {os.path.abspath(PLOTS_DIR)}")
    print(f"[INFO] Importance CSV saved to {csv_path}")

    return importance_df


if __name__ == "__main__":
    run_explainability()
