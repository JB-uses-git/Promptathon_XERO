import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import xgboost as xgb
import shap
import joblib
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')

def run_pipeline():
    print("Loading data...")
    df = pd.read_csv('amc_synthetic_data.csv')

    print("Phase 1: Feature Engineering & Preprocessing...")
    # Convert dates
    date_cols = ['contract_start_date', 'contract_end_date', 'last_service_date', 'estimated_churn_date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    today = pd.to_datetime('2025-05-12')

    # Derived columns
    df['contract_duration_days'] = (df['contract_end_date'] - df['contract_start_date']).dt.days
    df['contract_duration_days'] = df['contract_duration_days'].replace(0, 1) # avoid div by zero
    
    df['service_call_rate'] = df['total_service_calls'] / df['contract_duration_days']
    df['days_since_last_service'] = (today - df['last_service_date']).dt.days.fillna(999) 
    df['renewal_loyalty_score'] = df['previous_renewals'] * df['contract_value_inr']

    # Encoding Categoricals
    cat_cols = ['city', 'equipment_brand', 'contract_tier', 'equipment_type']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        encoders[col] = le
    
    joblib.dump(encoders, 'encoders.joblib')

    features = [
        'contract_value_inr', 'equipment_age_years', 'total_service_calls',
        'avg_resolution_time_days', 'unresolved_complaints', 'missed_scheduled_visits',
        'repeat_complaints', 'previous_renewals', 'days_to_expiry', 
        'renewal_reminder_sent', 'last_renewal_delay_days',
        'contract_duration_days', 'service_call_rate', 'days_since_last_service',
        'renewal_loyalty_score',
        'city_encoded', 'equipment_brand_encoded', 'contract_tier_encoded', 'equipment_type_encoded'
    ]
    
    X = df[features]
    y = df['churn']

    print("Phase 2: Churn Classification Model (XGBoost)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='auc'
    )
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]

    print(f"XGBoost Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
    print(f"AUC-ROC:  {roc_auc_score(y_test, y_prob):.3f}")

    joblib.dump(xgb_model, 'xgboost_churn_model.joblib')
    X_train.to_csv('X_train_reference.csv', index=False)
    
    print("Phase 3: Churn Timing Model (Cox Proportional Hazards)...")
    df_cox = df.copy()
    
    churn_days = (df_cox['estimated_churn_date'] - today).dt.days
    df_cox['duration'] = np.where(df_cox['churn'] == 1, churn_days, df_cox['days_to_expiry'])
    df_cox['duration'] = df_cox['duration'].clip(lower=1)
    
    cox_features = [
        'duration', 'churn',
        'unresolved_complaints', 'avg_resolution_time_days', 'equipment_age_years',
        'previous_renewals', 'contract_value_inr'
    ]
    df_cph = df_cox[cox_features].dropna()
    
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_cph, duration_col='duration', event_col='churn')
    
    print("CoxPH Concordance Index:", cph.concordance_index_)
    joblib.dump(cph, 'cox_model.joblib')

    # Ensure all predictions are generated for the app
    df['churn_probability'] = xgb_model.predict_proba(X)[:, 1]
    
    df.to_csv('processed_amc_data.csv', index=False)
    print("Pipeline Complete! Saved models and processed data.")

if __name__ == '__main__':
    run_pipeline()
