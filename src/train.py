"""
Model training for Customer Churn Prediction.
Implements the champion-challenger strategy from the PRD:
  - Baseline 0: Heuristic rules
  - Baseline 1: Regularised logistic regression
  - Champion:   XGBoost (with calibration)
  - Champion:   CatBoost (with calibration)
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
RANDOM_STATE = 42


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def load_splits():
    """Load pre-computed train/val/test splits."""
    path = os.path.join(ARTIFACTS_DIR, "splits.joblib")
    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(path)
    return X_train, X_val, X_test, y_train, y_val, y_test


def _save_model(model, name: str):
    """Persist a trained model."""
    path = os.path.join(ARTIFACTS_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"  -> Saved model to {path}")


# --------------------------------------------------
# Baseline 0 - Heuristic
# --------------------------------------------------

class HeuristicModel:
    """
    Rule-based churn score based on EDA findings:
    - Month-to-month contract -> higher risk
    - New customer (tenure <= 12) -> higher risk
    - High monthly charges (> $120) -> higher risk
    Returns a probability-like score in [0, 1].
    """

    def __init__(self):
        self.rules = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Define heuristic rules from EDA findings
        self.rules = []
        if "contract_Month-to-month" in X.columns:
            self.rules.append("contract_Month-to-month")
        if "is_new_customer" in X.columns:
            self.rules.append("is_new_customer")
        if "is_high_spender" in X.columns:
            self.rules.append("is_high_spender")
        if "mtm_high_charges" in X.columns:
            self.rules.append("mtm_high_charges")
        print(f"  Rules based on: {self.rules}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score = np.zeros(len(X))
        weights = {
            "contract_Month-to-month": 0.40,
            "is_new_customer": 0.30,
            "is_high_spender": 0.15,
            "mtm_high_charges": 0.15,
        }
        for col in self.rules:
            if col in X.columns:
                w = weights.get(col, 0.25)
                score += w * X[col].values.astype(float)
        score = np.clip(score, 0, 1)
        return np.column_stack([1 - score, score])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------
# Training functions
# --------------------------------------------------

def train_heuristic(X_train, y_train):
    """Baseline 0: Heuristic rule model."""
    print("\n" + "="*38)
    print("  BASELINE 0 - Heuristic Rules")
    print("="*38)
    model = HeuristicModel()
    model.fit(X_train, y_train)
    _save_model(model, "model_heuristic")
    return model


def train_logistic(X_train, y_train):
    """Baseline 1: Regularised logistic regression with standard scaling."""
    print("\n" + "="*38)
    print("  BASELINE 1 - Logistic Regression")
    print("="*38)
    t0 = time.time()

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight={0: 1.0, 1: pos_weight},
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )),
    ])
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")
    _save_model(model, "model_logistic")
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """Champion candidate: XGBoost with Platt calibration."""
    print("\n" + "="*38)
    print("  CHAMPION - XGBoost")
    print("="*38)
    t0 = time.time()

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    base_model = XGBClassifier(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=0.5,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=RANDOM_STATE,
        verbosity=0,
        use_label_encoder=False,
    )
    base_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"  Base model iterations: {base_model.best_iteration}")

    # Calibrate for better ECE
    calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    calibrated.fit(X_val, y_val)

    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s (incl. calibration)")
    _save_model(calibrated, "model_xgboost")
    return calibrated


def train_catboost(X_train, y_train, X_val, y_val):
    """Champion candidate: CatBoost with Platt calibration."""
    print("\n" + "="*38)
    print("  CHAMPION - CatBoost")
    print("="*38)
    t0 = time.time()

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    base_model = CatBoostClassifier(
        iterations=1000,
        depth=7,
        learning_rate=0.03,
        subsample=0.8,
        l2_leaf_reg=1.0,
        min_data_in_leaf=10,
        scale_pos_weight=pos_weight,
        eval_metric="Logloss",
        early_stopping_rounds=50,
        random_seed=RANDOM_STATE,
        verbose=0,
    )
    base_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
    )
    best_iter = base_model.get_best_iteration() if hasattr(base_model, "get_best_iteration") else "N/A"
    print(f"  Base model iterations: {best_iter}")

    # Calibrate for better ECE
    calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    calibrated.fit(X_val, y_val)

    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s (incl. calibration)")
    _save_model(calibrated, "model_catboost")
    return calibrated


# --------------------------------------------------
# Main
# --------------------------------------------------

def run_training():
    """Train all models and return them in a dict."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    print(f"[INFO] Training features: {list(X_train.columns)}")
    print(f"[INFO] Train size: {len(X_train)}  |  Churn rate: {y_train.mean():.4f}")

    models = {}
    models["heuristic"] = train_heuristic(X_train, y_train)
    models["logistic"] = train_logistic(X_train, y_train)
    models["xgboost"] = train_xgboost(X_train, y_train, X_val, y_val)
    models["catboost"] = train_catboost(X_train, y_train, X_val, y_val)

    print("\n[DONE] All models trained and saved.\n")
    return models


if __name__ == "__main__":
    run_training()
