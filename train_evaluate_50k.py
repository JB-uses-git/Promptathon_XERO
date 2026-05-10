"""
Full Train + Evaluate pipeline on the expanded Telco 50K dataset.
Trains CatBoost, XGBoost, WTTE, and Ensemble — then compares all results.
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# TensorFlow / Keras for WTTE
import tensorflow as tf
import keras
import keras.ops as ops
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Dropout, Layer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── WTTE custom objects ──
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

# ── Directories ──
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts", "telco50k")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Metric helpers ──
def top_decile_lift(y_true, y_prob):
    n = len(y_true); k = max(int(n * 0.10), 1)
    order = np.argsort(-y_prob)
    return y_true[order[:k]].mean() / max(y_true.mean(), 1e-7)

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

def evaluate_model(name, y_true, y_prob):
    return {
        "Model": name,
        "ROC-AUC": round(roc_auc_score(y_true, y_prob), 4),
        "PR-AUC": round(average_precision_score(y_true, y_prob), 4),
        "Top-Decile Lift": round(top_decile_lift(y_true, y_prob), 4),
        "Precision@10%": round(precision_at_top_k(y_true, y_prob), 4),
        "Recall@10%": round(recall_at_top_k(y_true, y_prob), 4),
        "ECE": round(expected_calibration_error(y_true, y_prob), 4),
        "Brier": round(brier_score_loss(y_true, y_prob), 4),
    }


def prepare_data():
    """Load expanded dataset, engineer features, split."""
    print("\n[DATA] Loading Telco_churn_50k.csv...")
    df = pd.read_csv("Telco_churn_50k.csv")
    
    # Clean TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
    
    y = (df['Churn'] == 'Yes').astype(int).values
    tenure = df['tenure'].values.astype(np.float32)
    
    # Feature engineering
    feat_df = df.drop(columns=['customerID', 'Churn']).copy()
    feat_df['charges_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
    feat_df['is_new'] = (df['tenure'] <= 6).astype(float)
    feat_df['is_high_spender'] = (df['MonthlyCharges'] > 80).astype(float)
    feat_df['is_mtm'] = (df['Contract'] == 'Month-to-month').astype(float)
    feat_df['mtm_new'] = feat_df['is_mtm'] * feat_df['is_new']
    feat_df['mtm_high_charges'] = feat_df['is_mtm'] * feat_df['is_high_spender']
    
    # Encode categoricals
    cat_cols = feat_df.select_dtypes(include='object').columns.tolist()
    feat_encoded = pd.get_dummies(feat_df, columns=cat_cols, drop_first=True)
    
    # Normalize numeric for WTTE
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_per_tenure']
    norm_stats = {}
    for col in numeric_cols:
        mean = feat_encoded[col].mean()
        std = feat_encoded[col].std()
        norm_stats[col] = {'mean': float(mean), 'std': float(std)}
    
    # WTTE needs normalized features
    feat_norm = feat_encoded.copy()
    for col in numeric_cols:
        feat_norm[col] = (feat_norm[col] - norm_stats[col]['mean']) / (norm_stats[col]['std'] + 1e-7)
    
    X = feat_encoded.values.astype(np.float32)
    X_norm = feat_norm.values.astype(np.float32)
    
    # Stratified split: 70/15/15
    X_trainval, X_test, Xn_trainval, Xn_test, y_trainval, y_test, t_trainval, t_test = \
        train_test_split(X, X_norm, y, tenure, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, Xn_train, Xn_val, y_train, y_val, t_train, t_val = \
        train_test_split(X_trainval, Xn_trainval, y_trainval, t_trainval, test_size=0.176, random_state=42, stratify=y_trainval)
    
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Features: {X.shape[1]} | Churn rate: {y.mean()*100:.1f}%")
    
    data = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'Xn_train': Xn_train, 'Xn_val': Xn_val, 'Xn_test': Xn_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        't_train': t_train, 't_val': t_val, 't_test': t_test,
        'feature_names': feat_encoded.columns.tolist(),
        'norm_stats': norm_stats,
    }
    return data


def train_catboost(data):
    print("\n[CATBOOST] Training...")
    model = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        eval_metric='AUC', random_seed=42, verbose=50,
        early_stopping_rounds=30
    )
    model.fit(data['X_train'], data['y_train'],
              eval_set=(data['X_val'], data['y_val']),
              verbose=50)
    probs = model.predict_proba(data['X_test'])[:, 1]
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "model_catboost_50k.joblib"))
    return probs


def train_xgboost(data):
    print("\n[XGBOOST] Training...")
    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        eval_metric='auc', random_state=42, verbosity=0,
        early_stopping_rounds=30
    )
    model.fit(data['X_train'], data['y_train'],
              eval_set=[(data['X_val'], data['y_val'])],
              verbose=False)
    probs = model.predict_proba(data['X_test'])[:, 1]
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "model_xgboost_50k.joblib"))
    return probs


def train_wtte(data):
    print("\n[WTTE] Training survival model...")
    num_features = data['Xn_train'].shape[1]
    
    # Build model
    inputs = Input(shape=(num_features,), name="features")
    h = Dense(64, activation='relu')(inputs)
    h = BatchNormalization()(h)
    h = Dropout(0.2)(h)
    h = Dense(32, activation='relu')(h)
    h = BatchNormalization()(h)
    raw_ab = Dense(2, name="raw_params")(h)
    ab = WeibullOutputLayer(init_alpha=36.0, max_beta_value=4.0, name="weibull_output")(raw_ab)
    
    model = Model(inputs=inputs, outputs=ab)
    optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(loss=wtte_loss, optimizer=optimizer)
    
    # Prepare WTTE targets: [tenure, churn_event]
    y_train_wtte = np.stack([data['t_train'], data['y_train'].astype(np.float32)], axis=-1)
    y_val_wtte = np.stack([data['t_val'], data['y_val'].astype(np.float32)], axis=-1)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    model.fit(
        data['Xn_train'], y_train_wtte,
        epochs=100, batch_size=1024,
        validation_data=(data['Xn_val'], y_val_wtte),
        callbacks=callbacks, verbose=1
    )
    
    # Predict on test set
    preds = model.predict(data['Xn_test'], verbose=0)
    alphas = preds[:, 0]
    betas = preds[:, 1]
    wtte_probs = np.clip(1.0 - np.exp(-np.power(12.0 / alphas, betas)), 0, 1)
    
    model.save(os.path.join(ARTIFACTS_DIR, "wtte_model_50k.keras"))
    return wtte_probs, alphas, betas


def plot_roc(models_data, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_true, y_prob, color in models_data:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Telco 50K Dataset', fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")

def plot_pr(models_data, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_true, y_prob, color in models_data:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, linewidth=2)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('PR Curve - Telco 50K Dataset', fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("=" * 70)
    print("  FULL PIPELINE: Train + Evaluate on Telco 50K (Expanded)")
    print("=" * 70)
    
    # 1. Prepare data
    data = prepare_data()
    
    # 2. Train models
    cb_probs = train_catboost(data)
    xgb_probs = train_xgboost(data)
    wtte_probs, alphas, betas = train_wtte(data)
    
    # 3. Build ensemble
    print("\n[ENSEMBLE] Blending CatBoost + WTTE...")
    best_auc, best_w = 0, 0.95
    for w in np.arange(0.5, 1.0, 0.05):
        blended = w * cb_probs + (1 - w) * wtte_probs
        auc = roc_auc_score(data['y_test'], blended)
        if auc > best_auc:
            best_auc = auc; best_w = w
    
    ensemble_probs = best_w * cb_probs + (1 - best_w) * wtte_probs
    print(f"  Best blend: {best_w:.0%} CatBoost + {1-best_w:.0%} WTTE")
    
    # 4. Evaluate all
    print("\n" + "=" * 70)
    print("  TELCO 50K LEADERBOARD")
    print("=" * 70)
    
    y_test = data['y_test']
    results = [
        evaluate_model("CatBoost", y_test, cb_probs),
        evaluate_model("XGBoost", y_test, xgb_probs),
        evaluate_model("WTTE-Survival", y_test, wtte_probs),
        evaluate_model(f"Ensemble ({best_w:.0%}CB+{1-best_w:.0%}WTTE)", y_test, ensemble_probs),
    ]
    
    leaderboard = pd.DataFrame(results).set_index("Model")
    leaderboard["ROC>=0.75"] = leaderboard["ROC-AUC"].apply(lambda x: "PASS" if x >= 0.75 else "FAIL")
    leaderboard["PR>=0.40"] = leaderboard["PR-AUC"].apply(lambda x: "PASS" if x >= 0.40 else "FAIL")
    leaderboard["ECE<=0.05"] = leaderboard["ECE"].apply(lambda x: "PASS" if x <= 0.05 else "FAIL")
    
    print(leaderboard.to_string())
    
    # Save
    csv_path = os.path.join(ARTIFACTS_DIR, "leaderboard_telco50k.csv")
    leaderboard.to_csv(csv_path)
    print(f"\n  Saved: {csv_path}")
    
    # 5. Cross-branch comparison
    print("\n" + "=" * 70)
    print("  CROSS-BRANCH COMPARISON")
    print("=" * 70)
    
    # Load results from other branches
    old_lb_path = os.path.join("artifacts", "evaluation_leaderboard.csv")
    wtte_lb_path = os.path.join("artifacts", "evaluation_leaderboard_wtte.csv")
    
    comparison = []
    
    if os.path.exists(old_lb_path):
        old_lb = pd.read_csv(old_lb_path, index_col=0)
        if 'catboost' in old_lb.index:
            comparison.append({
                "Branch": "main",
                "Dataset": "Synthetic 100K",
                "Best Model": "CatBoost",
                "ROC-AUC": old_lb.loc['catboost', 'ROC-AUC'],
                "PR-AUC": old_lb.loc['catboost', 'PR-AUC'],
                "Top-Decile Lift": old_lb.loc['catboost', 'Top-Decile Lift'],
                "Brier": old_lb.loc['catboost', 'Brier'],
            })
    
    if os.path.exists(wtte_lb_path):
        wtte_lb = pd.read_csv(wtte_lb_path, index_col=0)
        if 'WTTE-Survival' in wtte_lb.index:
            comparison.append({
                "Branch": "rnn-extension",
                "Dataset": "Synthetic 100K",
                "Best Model": "WTTE-Survival",
                "ROC-AUC": wtte_lb.loc['WTTE-Survival', 'ROC-AUC'],
                "PR-AUC": wtte_lb.loc['WTTE-Survival', 'PR-AUC'],
                "Top-Decile Lift": wtte_lb.loc['WTTE-Survival', 'Top-Decile Lift'],
                "Brier": wtte_lb.loc['WTTE-Survival', 'Brier'],
            })
    
    # Add ensemble from rnn-extension branch
    ens_lb_path = os.path.join("artifacts", "evaluation_leaderboard_ensemble.csv")
    if os.path.exists(ens_lb_path):
        ens_lb = pd.read_csv(ens_lb_path, index_col=0)
        best_ens = ens_lb['ROC-AUC'].idxmax()
        comparison.append({
            "Branch": "rnn-extension",
            "Dataset": "Synthetic 100K",
            "Best Model": best_ens,
            "ROC-AUC": ens_lb.loc[best_ens, 'ROC-AUC'],
            "PR-AUC": ens_lb.loc[best_ens, 'PR-AUC'],
            "Top-Decile Lift": ens_lb.loc[best_ens, 'Top-Decile Lift'],
            "Brier": ens_lb.loc[best_ens, 'Brier'],
        })
    
    # This branch's best
    best_here = leaderboard['ROC-AUC'].idxmax()
    comparison.append({
        "Branch": "dataset-expansion",
        "Dataset": "Telco 50K (expanded)",
        "Best Model": best_here,
        "ROC-AUC": leaderboard.loc[best_here, 'ROC-AUC'],
        "PR-AUC": leaderboard.loc[best_here, 'PR-AUC'],
        "Top-Decile Lift": leaderboard.loc[best_here, 'Top-Decile Lift'],
        "Brier": leaderboard.loc[best_here, 'Brier'],
    })
    
    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))
    comp_df.to_csv(os.path.join(ARTIFACTS_DIR, "cross_branch_comparison.csv"), index=False)
    
    # 6. Generate plots
    print("\n[PLOTS] Generating evaluation plots...")
    colors = {'CatBoost': '#1976D2', 'XGBoost': '#388E3C', 
              'WTTE-Survival': '#E53935', f"Ensemble ({best_w:.0%}CB+{1-best_w:.0%}WTTE)": '#9C27B0'}
    
    models_data = [
        ("CatBoost", y_test, cb_probs, '#1976D2'),
        ("XGBoost", y_test, xgb_probs, '#388E3C'),
        ("WTTE-Survival", y_test, wtte_probs, '#E53935'),
        ("Ensemble", y_test, ensemble_probs, '#9C27B0'),
    ]
    
    plot_roc(models_data, os.path.join(PLOTS_DIR, "roc_telco50k.png"))
    plot_pr(models_data, os.path.join(PLOTS_DIR, "pr_telco50k.png"))
    
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
