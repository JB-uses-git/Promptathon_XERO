from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import pandas as pd
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

@app.get("/api/kpis")
def get_kpis():
    return {
        "revenue_at_risk": "$1.24M",
        "avg_churn_risk": "34.2%",
        "expiring_30_days": "1,420"
    }

@app.get("/api/customers")
def get_customers():
    # Load test split
    splits_path = os.path.join(ARTIFACTS_DIR, "splits.joblib")
    if not os.path.exists(splits_path):
        return []
        
    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(splits_path)
    
    # Load CatBoost model
    model_path = os.path.join(ARTIFACTS_DIR, "model_catboost.joblib")
    if not os.path.exists(model_path):
        return []
        
    model = joblib.load(model_path)
    
    # Get probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.predict(X_test).astype(float)
    
    df = X_test.copy()
    df['churn_prob'] = probs
    
    # Priority Score = Risk * Value
    if 'MonthlyCharges' in df.columns:
        df['priority_score'] = df['churn_prob'] * df['MonthlyCharges']
    else:
        df['MonthlyCharges'] = 100 # Fallback
        df['priority_score'] = df['churn_prob'] * 100
        
    # Get top 50 highest priority customers
    df = df.sort_values(by='priority_score', ascending=False).head(50)
    
    customers = []
    for idx, row in df.iterrows():
        # Derive human readable drivers based on features
        drivers = []
        if 'contract_Month-to-month' in row and row['contract_Month-to-month'] == 1:
            drivers.append("Month-to-month contract")
        if 'is_new_customer' in row and row['is_new_customer'] == 1:
            drivers.append("New customer (<12mo)")
        if 'is_high_spender' in row and row['is_high_spender'] == 1:
            drivers.append("High monthly spend")
            
        if not drivers:
            drivers.append("Low engagement / Usage anomalies")
            
        # Determine playbook action
        if row['churn_prob'] > 0.85:
            action = "Escalate to Retention Specialist"
        elif row['churn_prob'] > 0.7:
            action = "Offer 10% Loyalty Discount"
        else:
            action = "Schedule Technical Review"
            
        customers.append({
            "id": f"CUST-{idx}",
            "risk_score": float(row['churn_prob']),
            "monthly_charges": float(row['MonthlyCharges']),
            "priority_score": float(row['priority_score']),
            "drivers": drivers,
            "recommended_action": action
        })
        
    return customers
