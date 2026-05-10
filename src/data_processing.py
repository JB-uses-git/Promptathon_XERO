"""
Data preprocessing and feature engineering for Customer Churn Prediction.
Loads raw CSV (synthetic_customer_churn_100k.csv), cleans anomalies,
engineers features, and produces stratified train/validation/test splits.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# Configuration
# --------------------------------------------------
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "synthetic_customer_churn_100k.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
RANDOM_STATE = 42

TARGET = "Churn"
DROP_COLS = ["CustomerID"]  # non-predictive identifiers


def load_and_clean(path: str) -> pd.DataFrame:
    """Load the CSV and handle anomalies."""
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns.")

    # Encode target: Yes -> 1, No -> 0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    print(f"[INFO] Churn distribution: {df['Churn'].mean():.4f} (positive rate)")

    # Clip negative TotalCharges to 0 (265 anomalous records found in EDA)
    n_neg = (df["TotalCharges"] < 0).sum()
    df["TotalCharges"] = df["TotalCharges"].clip(lower=0)
    print(f"[INFO] Clipped {n_neg} negative TotalCharges values to 0.")

    # Verify no nulls
    assert df.isnull().sum().sum() == 0, "Unexpected null values found."

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features and encode categoricals."""

    # --- Tenure-derived features ---
    # Tenure is in months (range 1-72)
    df["tenure_years"] = df["Tenure"] / 12.0

    # New customer flag (first 12 months have 64% churn rate per EDA)
    df["is_new_customer"] = (df["Tenure"] <= 12).astype(int)

    # Tenure buckets (non-linear effect: 0-12 very different from 12+)
    df["tenure_bucket"] = pd.cut(
        df["Tenure"],
        bins=[0, 6, 12, 24, 48, 72],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    # --- Charges-derived features ---
    # Average charges per month of tenure (captures billing consistency)
    df["avg_charge_per_month"] = df["TotalCharges"] / (df["Tenure"] + 1)

    # High spender flag (>$120/month has 52% churn per EDA)
    df["is_high_spender"] = (df["MonthlyCharges"] > 120).astype(int)

    # Medium-high spender flag (>$90/month has ~42% churn)
    df["is_med_high_spender"] = (df["MonthlyCharges"] > 90).astype(int)

    # Charges deviation from expected total
    df["expected_total"] = df["Tenure"] * df["MonthlyCharges"]
    df["charges_deviation"] = df["TotalCharges"] - df["expected_total"]

    # Monthly charges squared (captures non-linear effect)
    df["monthly_charges_sq"] = df["MonthlyCharges"] ** 2

    # --- Interaction features ---
    # Contract x MonthlyCharges: Month-to-month + high charges = highest churn risk
    df["mtm_high_charges"] = ((df["Contract"] == "Month-to-month") &
                               (df["MonthlyCharges"] > 90)).astype(int)

    # Month-to-month + new customer (the most at-risk segment)
    df["mtm_new_customer"] = ((df["Contract"] == "Month-to-month") &
                               (df["Tenure"] <= 12)).astype(int)

    # Triple interaction: Month-to-month + new + high charges
    df["mtm_new_high"] = ((df["Contract"] == "Month-to-month") &
                           (df["Tenure"] <= 12) &
                           (df["MonthlyCharges"] > 90)).astype(int)

    # --- Age bins ---
    df["age_group"] = pd.cut(
        df["Age"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
    )

    # --- Encode categoricals ---
    label_encoders = {}
    cat_cols = ["Gender", "Contract", "PaymentMethod", "age_group"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # --- One-Hot encode Contract (strongest predictor) ---
    contract_dummies = pd.get_dummies(df["Contract"], prefix="contract", dtype=int)
    df = pd.concat([df, contract_dummies], axis=1)

    # Drop original categorical columns and non-predictive columns
    cols_to_drop = cat_cols + DROP_COLS + ["expected_total"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    return df, label_encoders


def split_data(df: pd.DataFrame, target: str = TARGET):
    """Stratified 70/15/15 train/validation/test split."""
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"[INFO] Split sizes -> Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")
    print(f"[INFO] Churn rates -> Train: {y_train.mean():.4f}  |  Val: {y_val.mean():.4f}  |  Test: {y_test.mean():.4f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def run_pipeline():
    """Execute the full data processing pipeline and save artifacts."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load & clean
    df = load_and_clean(RAW_DATA_PATH)

    # 2. Feature engineering
    df, label_encoders = engineer_features(df)

    # 3. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # 4. Save
    joblib.dump((X_train, X_val, X_test, y_train, y_val, y_test),
                os.path.join(OUTPUT_DIR, "splits.joblib"))
    joblib.dump(label_encoders, os.path.join(OUTPUT_DIR, "label_encoders.joblib"))
    joblib.dump(list(X_train.columns), os.path.join(OUTPUT_DIR, "feature_names.joblib"))

    print(f"\n[INFO] Artifacts saved to {os.path.abspath(OUTPUT_DIR)}")
    print(f"[INFO] Feature columns ({len(X_train.columns)}): {list(X_train.columns)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    run_pipeline()
