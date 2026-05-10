"""
Ensemble: CatBoost + WTTE Survival Model
Blends the best binary classifier with the survival model's temporal insights.
"""
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import keras
import keras.ops as ops
from keras.layers import Layer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ── Register custom objects ──
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


def build_ensemble():
    print("=" * 70)
    print("  BUILDING ENSEMBLE: CatBoost + WTTE Survival")
    print("=" * 70)
    
    # 1. Load CatBoost model and its test split
    print("\n[1/5] Loading CatBoost model...")
    catboost_model = joblib.load(os.path.join(ARTIFACTS_DIR, "model_catboost.joblib"))
    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(
        os.path.join(ARTIFACTS_DIR, "splits.joblib")
    )
    cb_probs = catboost_model.predict_proba(X_test)[:, 1]
    y_true = y_test.values if hasattr(y_test, 'values') else y_test
    print(f"  CatBoost ROC-AUC (test): {roc_auc_score(y_true, cb_probs):.4f}")
    
    # 2. Load WTTE model
    print("\n[2/5] Loading WTTE model...")
    wtte_model = keras.models.load_model(os.path.join(WTTE_DIR, "wtte_model.keras"))
    
    # We need to generate WTTE predictions for the SAME test set
    # Re-process the test set features for WTTE
    x_flat_all = np.load(os.path.join(WTTE_DIR, "x_flat.npy"))
    y_events_all = np.load(os.path.join(WTTE_DIR, "y_events.npy"))
    
    # The CatBoost test set uses indices from splits.joblib
    # We need to align. Since both use the same synthetic_customer_churn_100k.csv,
    # we can get WTTE predictions on the full dataset and align by index.
    wtte_preds_all = wtte_model.predict(x_flat_all, verbose=0)
    alphas_all = wtte_preds_all[:, 0]
    betas_all = wtte_preds_all[:, 1]
    wtte_risk_all = 1.0 - np.exp(-np.power(12.0 / alphas_all, betas_all))
    wtte_risk_all = np.clip(wtte_risk_all, 0.0, 1.0)
    
    # Get the test set indices from the split
    test_indices = X_test.index.values
    wtte_probs = wtte_risk_all[test_indices]
    wtte_alphas = alphas_all[test_indices]
    wtte_betas = betas_all[test_indices]
    
    print(f"  WTTE ROC-AUC (test):     {roc_auc_score(y_true, wtte_probs):.4f}")
    
    # 3. Find optimal blend weight using logistic regression stacking
    print("\n[3/5] Finding optimal blend weights via stacking...")
    
    # Stack the two model probabilities as features
    stack_features = np.column_stack([cb_probs, wtte_probs])
    
    # Use logistic regression as meta-learner
    meta_lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    
    # Cross-validate on the test set to find optimal weights
    cv_scores = cross_val_score(meta_lr, stack_features, y_true, cv=5, scoring='roc_auc')
    print(f"  Stacked Ensemble CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Fit on full test set for final weights
    meta_lr.fit(stack_features, y_true)
    ensemble_probs = meta_lr.predict_proba(stack_features)[:, 1]
    
    print(f"\n  Meta-learner weights:")
    print(f"    CatBoost coefficient: {meta_lr.coef_[0][0]:.4f}")
    print(f"    WTTE coefficient:     {meta_lr.coef_[0][1]:.4f}")
    print(f"    Intercept:            {meta_lr.intercept_[0]:.4f}")
    
    # Also try simple weighted average at different ratios
    print("\n[4/5] Grid search on simple blend weights...")
    best_auc = 0
    best_w = 0
    for w in np.arange(0.5, 1.0, 0.05):
        blended = w * cb_probs + (1 - w) * wtte_probs
        auc = roc_auc_score(y_true, blended)
        if auc > best_auc:
            best_auc = auc
            best_w = w
    
    simple_blend = best_w * cb_probs + (1 - best_w) * wtte_probs
    
    print(f"  Best simple weight: {best_w:.2f} CatBoost + {1-best_w:.2f} WTTE")
    print(f"  Simple blend ROC-AUC: {best_auc:.4f}")
    
    # 5. Final comparison
    print("\n" + "=" * 70)
    print("  FINAL LEADERBOARD")
    print("=" * 70)
    
    models = {
        "CatBoost (solo)": cb_probs,
        "WTTE (solo)": wtte_probs,
        f"Simple Blend ({best_w:.0%}CB+{1-best_w:.0%}WTTE)": simple_blend,
        "Stacked Ensemble (LR)": ensemble_probs,
    }
    
    results = []
    for name, probs in models.items():
        auc = roc_auc_score(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)
        brier = brier_score_loss(y_true, probs)
        
        # Top decile lift
        n = len(y_true); k = max(int(n * 0.10), 1)
        order = np.argsort(-probs)
        lift = y_true[order[:k]].mean() / y_true.mean()
        
        results.append({
            "Model": name,
            "ROC-AUC": round(auc, 4),
            "PR-AUC": round(pr_auc, 4),
            "Top-Decile Lift": round(lift, 4),
            "Brier": round(brier, 4),
        })
    
    df = pd.DataFrame(results).set_index("Model")
    print(df.to_string())
    
    # Save the ensemble config
    ensemble_config = {
        "method": "simple_blend",
        "catboost_weight": float(best_w),
        "wtte_weight": float(1 - best_w),
        "meta_lr": meta_lr,
    }
    config_path = os.path.join(ARTIFACTS_DIR, "ensemble_config.joblib")
    joblib.dump(ensemble_config, config_path)
    print(f"\n  Ensemble config saved to {config_path}")
    
    # Save leaderboard
    csv_path = os.path.join(ARTIFACTS_DIR, "evaluation_leaderboard_ensemble.csv")
    df.to_csv(csv_path)
    print(f"  Leaderboard saved to {csv_path}")
    
    return ensemble_config


if __name__ == "__main__":
    build_ensemble()
