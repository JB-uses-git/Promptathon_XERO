import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, 
    precision_score, recall_score, classification_report
)
import xgboost as xgb
import shap
import joblib
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')

def run_pipeline():
    print("=" * 60)
    print("AMC Churn Prediction Pipeline")
    print("=" * 60)

    # ----------------------------------------------------------------
    # PHASE 1 — Data Prep & Feature Engineering
    # ----------------------------------------------------------------
    print("\n[Phase 1] Loading & Engineering Features...")
    df = pd.read_csv('amc_synthetic_data.csv')
    print(f"  Loaded {len(df)} rows, churn rate = {df['churn'].mean():.2%}")

    # Convert dates
    date_cols = ['contract_start_date', 'contract_end_date', 'last_service_date', 'estimated_churn_date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    today = pd.to_datetime('2025-05-12')

    # Derived features
    df['contract_duration_days'] = (df['contract_end_date'] - df['contract_start_date']).dt.days
    df['contract_duration_days'] = df['contract_duration_days'].clip(lower=1)
    df['expiry_month'] = df['contract_end_date'].dt.month
    
    df['service_call_rate'] = df['total_service_calls'] / df['contract_duration_days']
    
    df['days_since_last_service'] = (today - df['last_service_date']).dt.days
    df['days_since_last_service'] = df['days_since_last_service'].fillna(
        df['days_since_last_service'].max() if df['days_since_last_service'].notna().any() else 999
    ).clip(lower=0)
    
    df['renewal_loyalty_score'] = df['previous_renewals'] * df['contract_value_inr']

    # Encode categoricals
    cat_cols = ['city', 'equipment_brand', 'contract_tier', 'equipment_type']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    joblib.dump(encoders, 'encoders.joblib')

    # Define feature set
    features = [
        'contract_value_inr', 'equipment_age_years', 'total_service_calls',
        'avg_resolution_time_days', 'unresolved_complaints', 'missed_scheduled_visits',
        'repeat_complaints', 'previous_renewals', 'days_to_expiry', 
        'renewal_reminder_sent', 'last_renewal_delay_days',
        'contract_duration_days', 'service_call_rate', 'days_since_last_service',
        'renewal_loyalty_score', 'expiry_month',
        'city_encoded', 'equipment_brand_encoded', 'contract_tier_encoded', 'equipment_type_encoded',
        'renewal_rate'
    ]
    
    # Convert booleans to int for XGBoost
    df['repeat_complaints'] = df['repeat_complaints'].astype(int)
    df['renewal_reminder_sent'] = df['renewal_reminder_sent'].astype(int)
    
    X = df[features].copy()
    y = df['churn'].copy()

    print(f"  Engineered {len(features)} features")
    print(f"  contract_duration_days range: {df['contract_duration_days'].min()} - {df['contract_duration_days'].max()}")
    print(f"  days_since_last_service range: {df['days_since_last_service'].min():.0f} - {df['days_since_last_service'].max():.0f}")

    # ----------------------------------------------------------------
    # PHASE 2 — XGBoost Churn Classification
    # ----------------------------------------------------------------
    print("\n[Phase 2] Training XGBoost Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    ratio = float(np.sum(y_train == 0)) / max(np.sum(y_train == 1), 1)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='auc',
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print(f"\n  XGBoost Results:")
    print(f"  {'Accuracy':<12} {acc:.3f}")
    print(f"  {'Precision':<12} {prec:.3f}")
    print(f"  {'Recall':<12} {rec:.3f}")
    print(f"  {'F1-Score':<12} {f1:.3f}")
    print(f"  {'AUC-ROC':<12} {auc:.3f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

    joblib.dump(xgb_model, 'xgboost_churn_model.joblib')
    joblib.dump(features, 'feature_names.joblib')
    X_train.to_csv('X_train_reference.csv', index=False)

    # ----------------------------------------------------------------
    # PHASE 3 — Cox Proportional Hazards (Survival Analysis)
    # ----------------------------------------------------------------
    print("[Phase 3] Training Cox Proportional Hazards Model...")
    df_cox = df.copy()
    
    # Duration = days until churn event (for churners) or days to expiry (for non-churners / censored)
    churn_days = (df_cox['estimated_churn_date'] - today).dt.days
    df_cox['duration'] = np.where(
        df_cox['churn'] == 1,
        churn_days,
        df_cox['days_to_expiry']
    )
    df_cox['duration'] = df_cox['duration'].clip(lower=1)
    
    cox_features = [
        'duration', 'churn',
        'unresolved_complaints', 'avg_resolution_time_days', 'equipment_age_years',
        'previous_renewals', 'contract_value_inr', 'missed_scheduled_visits',
        'days_since_last_service'
    ]
    df_cph = df_cox[cox_features].dropna()
    
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_cph, duration_col='duration', event_col='churn')
    
    print(f"  Concordance Index: {cph.concordance_index_:.4f}")
    print(f"  (>0.5 = better than random, >0.7 = good)")
    
    joblib.dump(cph, 'cox_model.joblib')

    # ----------------------------------------------------------------
    # Generate predictions for dashboard
    # ----------------------------------------------------------------
    print("\n[Final] Generating predictions for all customers...")
    df['churn_probability'] = xgb_model.predict_proba(X)[:, 1]
    
    # Generate Estimated Days to Churn using the Cox Survival Model
    # First, handle missing values for the Cox prediction matching what we did in training
    cox_pred_df = df_cox[cox_features].drop(columns=['duration', 'churn']).fillna(0)
    expected_duration = cph.predict_expectation(cox_pred_df)
    
    # The expected duration is the total days from contract start to expected churn.
    # To get days from TODAY until expected churn, we subtract the days passed.
    days_passed = (today - df['contract_start_date']).dt.days
    estimated_days_left = expected_duration - days_passed
    df['estimated_days_to_churn'] = estimated_days_left.clip(lower=1).round()
    
    df.to_csv('processed_amc_data.csv', index=False)
    
    print(f"\n{'=' * 60}")
    print(f"Pipeline Complete!")
    print(f"  Saved: xgboost_churn_model.joblib, cox_model.joblib")
    print(f"  Saved: processed_amc_data.csv, encoders.joblib")
    print(f"  Run:   streamlit run app.py")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    run_pipeline()
